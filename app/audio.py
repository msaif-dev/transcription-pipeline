"""Audio validation and normalization.

Every input format (WAV, MP3, M4A, FLAC, OGG/Opus, WebM, even the audio track of MP4 video)
is decoded with PyAV (FFmpeg's libraries, bundled in the wheel, so no system ffmpeg is needed)
and normalized to the one representation the STT model expects:

    16 kHz, mono, float32 PCM in [-1, 1]

Decoding is *streamed*: frames are resampled and yielded as small blocks, so a multi-hour
file never has to sit in memory as one array.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np

from .models import AudioInfo

TARGET_SAMPLE_RATE = 16_000
_MAX_CORRUPT_PACKETS = 50


class AudioError(Exception):
    """Base class for input problems that are the caller's fault (mapped to 4xx)."""


class UnsupportedFormatError(AudioError):
    pass


class InvalidAudioError(AudioError):
    pass


class AudioTooLongError(AudioError):
    pass


def check_extension(filename: str, allowed: frozenset[str]) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise UnsupportedFormatError(
            f"Unsupported file type '{ext or '(none)'}'. Allowed: {', '.join(sorted(allowed))}"
        )
    return ext


def _reason(exc: Exception) -> str:
    """FFmpeg error text without the server-side file path (don't leak internals to clients)."""
    return getattr(exc, "strerror", None) or type(exc).__name__


def probe(path: Path, filename: str | None = None) -> AudioInfo:
    """Open the file and read its header. Extensions can lie, so the content is what we trust."""
    path = Path(path)
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise InvalidAudioError("File contains no audio stream")
            stream = container.streams.audio[0]
            ctx = stream.codec_context

            duration: float | None = None
            if container.duration is not None:
                duration = container.duration / av.time_base
            elif stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)

            channels = getattr(getattr(ctx, "layout", None), "nb_channels", None) or getattr(ctx, "channels", 0)
            return AudioInfo(
                filename=filename or path.name,
                container=container.format.name,
                codec=ctx.name,
                sample_rate=ctx.sample_rate or 0,
                channels=int(channels or 0),
                duration_seconds=round(duration, 3) if duration is not None else None,
                size_bytes=path.stat().st_size,
            )
    except AudioError:
        raise
    except (av.error.FFmpegError, UnicodeDecodeError, ValueError) as exc:
        raise InvalidAudioError(f"Could not read audio file: {_reason(exc)}") from exc


def stream_pcm(path: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> Iterator[np.ndarray]:
    """Yield the decoded audio as consecutive float32 mono blocks at ``sample_rate``.

    A handful of corrupt packets (common in truncated MP3s / recordings) are skipped rather than
    failing the whole file; a file that is mostly corrupt is rejected.
    """
    resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
    corrupt = 0
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise InvalidAudioError("File contains no audio stream")
            stream = container.streams.audio[0]
            stream.thread_type = "AUTO"
            for packet in container.demux(stream):
                try:
                    frames = packet.decode()
                except av.error.InvalidDataError:
                    corrupt += 1
                    if corrupt > _MAX_CORRUPT_PACKETS:
                        raise InvalidAudioError("Audio stream is too corrupted to decode")
                    continue
                for frame in frames:
                    frame.pts = None  # avoid resampler errors on timestamp discontinuities
                    for out in resampler.resample(frame):
                        yield out.to_ndarray().reshape(-1).astype(np.float32, copy=False)
            for out in resampler.resample(None):  # flush
                yield out.to_ndarray().reshape(-1).astype(np.float32, copy=False)
    except AudioError:
        raise
    except av.error.FFmpegError as exc:
        raise InvalidAudioError(f"Failed to decode audio: {_reason(exc)}") from exc


def load_pcm(path: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Decode a whole file into memory (convenience for short files and tests)."""
    blocks = list(stream_pcm(path, sample_rate))
    return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)
