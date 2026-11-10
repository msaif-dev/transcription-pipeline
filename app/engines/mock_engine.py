"""Deterministic fake engine: used by the test suite and for local development without model weights.

It emits one segment per 5 s window that contains non-silent audio, so tests can assert on
timestamps, chunk offsets and silence handling without downloading a model.
"""

from __future__ import annotations

import numpy as np

from .base import EngineOutput, EngineSegment, EngineWord, TranscriptionEngine

_WINDOW_SECONDS = 5.0
_SILENCE_RMS = 1e-3


class MockEngine(TranscriptionEngine):
    name = "mock"

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "mock"

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> EngineOutput:
        self.calls += 1
        window = int(_WINDOW_SECONDS * sample_rate)
        segments = []
        for i, start in enumerate(range(0, len(audio), window)):
            piece = audio[start : start + window]
            if piece.size == 0 or float(np.sqrt(np.mean(piece**2))) < _SILENCE_RMS:
                continue
            s, e = start / sample_rate, (start + len(piece)) / sample_rate
            text = f" mock segment {i}"
            words = [EngineWord(s, e, text.strip(), 0.99)] if word_timestamps else None
            segments.append(EngineSegment(s, e, text, avg_logprob=-0.1, no_speech_prob=0.01, words=words))
        return EngineOutput(segments, language or "en", 1.0)
