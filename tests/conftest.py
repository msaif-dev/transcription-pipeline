from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings


def write_wav(path: Path, samples: np.ndarray, sample_rate: int, channels: int = 1) -> Path:
    """Write float samples in [-1, 1] as 16-bit PCM WAV (interleaving if multi-channel)."""
    data = np.clip(samples, -1, 1)
    if channels > 1:
        data = np.repeat(data[:, None], channels, axis=1).reshape(-1)
    pcm = (data * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return path


def tone_with_gaps(sample_rate: int, pattern: list[tuple[str, float]]) -> np.ndarray:
    """Build audio from ('tone'|'silence', seconds) pieces."""
    parts = []
    for kind, seconds in pattern:
        n = int(seconds * sample_rate)
        if kind == "tone":
            t = np.arange(n) / sample_rate
            parts.append(0.5 * np.sin(2 * np.pi * 440 * t))
        else:
            parts.append(np.zeros(n))
    return np.concatenate(parts).astype(np.float32)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        engine="mock",
        chunk_seconds=10,
        chunk_search_seconds=3,
        sync_max_duration_seconds=30,
        max_duration_seconds=600,
        upload_dir=tmp_path / "uploads",
        _env_file=None,
    )


@pytest.fixture
def speechlike_wav(tmp_path: Path) -> Path:
    """25 s, 44.1 kHz stereo: tones (stand-in for speech) separated by silences near 10 s and 20 s."""
    audio = tone_with_gaps(
        44_100,
        [("tone", 4), ("silence", 1), ("tone", 4.5), ("silence", 1), ("tone", 8), ("silence", 1.5), ("tone", 5)],
    )
    return write_wav(tmp_path / "speech.wav", audio, 44_100, channels=2)
