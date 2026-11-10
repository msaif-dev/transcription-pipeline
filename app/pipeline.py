"""The pipeline: validate -> stream-decode & normalize -> chunk -> transcribe -> map timestamps -> post-process."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from . import audio as audio_mod
from .audio import TARGET_SAMPLE_RATE, AudioTooLongError
from .chunking import silence_aware_chunks
from .config import Settings
from .engines import TranscriptionEngine
from .models import Segment, TranscriptionResult, Word
from .postprocess import finalize_segments

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float | None, int], None]  # (fraction 0..1 or None if unknown, chunks done)


class TranscriptionPipeline:
    def __init__(self, engine: TranscriptionEngine, settings: Settings):
        self.engine = engine
        self.settings = settings

    def run(
        self,
        path: Path,
        *,
        filename: str | None = None,
        language: str | None = None,
        word_timestamps: bool | None = None,
        progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        s = self.settings
        started = time.perf_counter()
        word_ts = s.word_timestamps if word_timestamps is None else word_timestamps

        info = audio_mod.probe(Path(path), filename)
        if info.duration_seconds is not None and info.duration_seconds > s.max_duration_seconds:
            raise AudioTooLongError(
                f"Audio is {info.duration_seconds:.0f}s; the limit is {s.max_duration_seconds:.0f}s"
            )

        max_samples = int(s.max_duration_seconds * TARGET_SAMPLE_RATE)
        lang = language  # None -> auto-detect, then lock it for the remaining chunks
        lang_prob: float | None = 1.0 if language else None
        segments: list[Segment] = []
        decoded = 0
        n_chunks = 0

        chunks = silence_aware_chunks(
            audio_mod.stream_pcm(Path(path), TARGET_SAMPLE_RATE),
            TARGET_SAMPLE_RATE,
            s.chunk_seconds,
            s.chunk_search_seconds,
        )
        for chunk in chunks:
            decoded += len(chunk.samples)
            if decoded > max_samples:  # header had no/incorrect duration
                raise AudioTooLongError(f"Audio exceeds the {s.max_duration_seconds:.0f}s limit")

            out = self.engine.transcribe(
                chunk.samples, sample_rate=TARGET_SAMPLE_RATE, language=lang, word_timestamps=word_ts
            )
            # Lock the language once a chunk with real speech has been seen, so a multi-chunk file
            # isn't transcribed in a different language per chunk.
            if lang is None and out.segments:
                lang, lang_prob = out.language, out.language_probability

            for seg in out.segments:
                start = chunk.start + max(0.0, seg.start)
                end = chunk.start + min(seg.end, chunk.duration)  # clamp to chunk bounds
                words = (
                    [Word(start=chunk.start + w.start, end=chunk.start + w.end, word=w.word, probability=w.probability)
                     for w in seg.words]
                    if seg.words
                    else None
                )
                segments.append(
                    Segment(
                        id=0, start=start, end=max(start, end), text=seg.text,
                        avg_logprob=seg.avg_logprob, no_speech_prob=seg.no_speech_prob, words=words,
                    )
                )

            n_chunks += 1
            log.info("chunk %d done: %.1fs-%.1fs, %d segments", chunk.index, chunk.start, chunk.end, len(out.segments))
            if progress:
                total = info.duration_seconds
                progress(min(chunk.end / total, 1.0) if total else None, n_chunks)

        duration = decoded / TARGET_SAMPLE_RATE
        final = finalize_segments(segments)
        elapsed = time.perf_counter() - started
        return TranscriptionResult(
            text=" ".join(seg.text for seg in final),
            language=lang,
            language_probability=round(lang_prob, 4) if lang_prob is not None else None,
            duration_seconds=round(duration, 3),
            segments=final,
            audio=info,
            engine=self.engine.name,
            model=self.engine.model_name,
            chunks=n_chunks,
            processing_seconds=round(elapsed, 3),
            real_time_factor=round(elapsed / duration, 4) if duration else 0.0,
        )
