"""faster-whisper backend (OpenAI Whisper weights re-implemented on CTranslate2).

Chosen because it is open source (MIT), runs fully offline, is ~4x faster than the reference
openai-whisper implementation with lower memory, supports int8 quantization on CPU, and has
built-in Silero VAD plus segment- and word-level timestamps.
"""

from __future__ import annotations

import threading

import numpy as np

from ..config import Settings
from .base import EngineOutput, EngineSegment, EngineWord, TranscriptionEngine


class FasterWhisperEngine(TranscriptionEngine):
    name = "faster-whisper"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._settings.model_size

    def _get_model(self):
        # Lazy, thread-safe load: weights are downloaded/loaded once and shared by all workers.
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from faster_whisper import WhisperModel

                    s = self._settings
                    self._model = WhisperModel(
                        s.model_size,
                        device=s.device,
                        compute_type=s.compute_type,
                        cpu_threads=s.cpu_threads,
                        num_workers=max(1, s.job_workers),
                        download_root=str(s.model_dir) if s.model_dir else None,
                    )
        return self._model

    def warmup(self) -> None:
        self._get_model()

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> EngineOutput:
        if sample_rate != 16_000:
            raise ValueError("faster-whisper expects 16 kHz audio")
        model = self._get_model()
        segments_iter, info = model.transcribe(
            audio,
            language=language,
            beam_size=self._settings.beam_size,
            vad_filter=self._settings.vad_filter,
            word_timestamps=word_timestamps,
        )
        segments = []
        for seg in segments_iter:  # generator: decoding actually happens here
            words = None
            if word_timestamps and seg.words:
                words = [EngineWord(w.start, w.end, w.word, w.probability) for w in seg.words]
            segments.append(
                EngineSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    avg_logprob=seg.avg_logprob,
                    no_speech_prob=seg.no_speech_prob,
                    words=words,
                )
            )
        return EngineOutput(segments, info.language, info.language_probability)
