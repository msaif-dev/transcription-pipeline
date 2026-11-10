"""Public data contracts (API responses and pipeline output)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Word(BaseModel):
    start: float
    end: float
    word: str
    probability: float


class Segment(BaseModel):
    id: int
    start: float = Field(description="Segment start, seconds from the beginning of the file")
    end: float = Field(description="Segment end, seconds from the beginning of the file")
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    low_confidence: bool = False
    words: list[Word] | None = None


class AudioInfo(BaseModel):
    filename: str
    container: str
    codec: str
    sample_rate: int
    channels: int
    duration_seconds: float | None
    size_bytes: int


class TranscriptionResult(BaseModel):
    text: str
    language: str | None
    language_probability: float | None
    duration_seconds: float
    segments: list[Segment]
    audio: AudioInfo
    engine: str
    model: str
    chunks: int
    processing_seconds: float
    real_time_factor: float = Field(description="processing time / audio duration (lower is faster)")


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Job(BaseModel):
    id: str
    status: JobStatus
    filename: str
    created_at: datetime
    updated_at: datetime
    progress: float = 0.0
    error: str | None = None
    result: TranscriptionResult | None = None
