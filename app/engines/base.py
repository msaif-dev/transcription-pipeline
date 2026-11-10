"""Engine abstraction: the pipeline depends on this interface, not on a specific STT library.

Swapping Whisper for another backend (OpenAI API, Deepgram, AWS Transcribe, NVIDIA Parakeet...)
means adding one class here; chunking, validation, timestamps, exports and the API are unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EngineWord:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class EngineSegment:
    """Timestamps are relative to the start of the audio passed to the engine."""

    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    words: list[EngineWord] | None = None


@dataclass
class EngineOutput:
    segments: list[EngineSegment] = field(default_factory=list)
    language: str | None = None
    language_probability: float | None = None


class TranscriptionEngine(ABC):
    name: str = "base"

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    def warmup(self) -> None:
        """Load model weights ahead of the first request (optional)."""

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> EngineOutput:
        """Transcribe mono float32 PCM and return segments with chunk-relative timestamps."""
