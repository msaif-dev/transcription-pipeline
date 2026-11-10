"""Silence-aware chunking for long audio.

Why chunk at all? Whisper-family models work on 30 s windows internally, but handing a
2-hour file to the model in one call means: the whole waveform in RAM (~460 MB of float32),
no progress reporting, no way to resume/parallelize, and one failure loses everything.

Why *silence-aware*? Cutting at a fixed offset (e.g. exactly every 10 min) splits words in half,
which produces garbled or duplicated text at every boundary. Instead, around each target
boundary we search a small window for the quietest 100 ms frame (lowest RMS energy) and cut
there. Chunks therefore don't overlap, nothing needs de-duplicating, and timestamps are mapped
back to the original file by adding the chunk's start offset.

The chunker consumes the streaming decoder, so memory stays bounded by roughly one chunk
regardless of file length.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start_sample: int
    samples: np.ndarray
    sample_rate: int

    @property
    def start(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def end(self) -> float:
        return self.start + self.duration


def find_quiet_cut(audio: np.ndarray, lo: int, hi: int, frame: int) -> int:
    """Return a sample index in [lo, hi) at the centre of the lowest-energy frame."""
    lo = max(lo, 0)
    hi = min(hi, len(audio))
    window = audio[lo:hi]
    n_frames = len(window) // frame
    if n_frames <= 1:
        return (lo + hi) // 2
    frames = window[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    return lo + int(np.argmin(rms)) * frame + frame // 2


def silence_aware_chunks(
    blocks: Iterable[np.ndarray],
    sample_rate: int,
    chunk_seconds: float,
    search_seconds: float,
    frame_ms: int = 100,
) -> Iterator[AudioChunk]:
    target = int(chunk_seconds * sample_rate)
    search = int(search_seconds * sample_rate)
    frame = max(1, int(sample_rate * frame_ms / 1000))

    pending: list[np.ndarray] = []
    pending_len = 0
    offset = 0
    index = 0

    for block in blocks:
        if block.size == 0:
            continue
        pending.append(block)
        pending_len += len(block)

        # Only cut once we can see the whole search window past the target point.
        while pending_len >= target + search:
            audio = np.concatenate(pending)
            cut = find_quiet_cut(audio, target - search, target + search, frame) if search else target
            yield AudioChunk(index, offset, audio[:cut], sample_rate)
            index += 1
            offset += cut
            rest = audio[cut:]
            pending, pending_len = [rest], len(rest)

    if pending_len:
        yield AudioChunk(index, offset, np.concatenate(pending), sample_rate)
