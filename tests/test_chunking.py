import numpy as np

from app.chunking import find_quiet_cut, silence_aware_chunks
from tests.conftest import tone_with_gaps

SR = 16_000


def _blocks(audio: np.ndarray, size: int = 4000):
    for i in range(0, len(audio), size):
        yield audio[i : i + size]


def test_cut_lands_in_silence():
    audio = tone_with_gaps(SR, [("tone", 9), ("silence", 1), ("tone", 5)])
    cut = find_quiet_cut(audio, 7 * SR, 13 * SR, SR // 10)
    assert 9 * SR <= cut <= 10 * SR


def test_chunks_cut_at_silence_and_cover_all_audio():
    # silences at 9.0-10.0 s and 19.5-20.5 s; target 10 s, search +/-3 s
    audio = tone_with_gaps(
        SR, [("tone", 9), ("silence", 1), ("tone", 9.5), ("silence", 1), ("tone", 7)]
    )
    chunks = list(silence_aware_chunks(_blocks(audio), SR, chunk_seconds=10, search_seconds=3))

    assert len(chunks) == 3
    # contiguous, non-overlapping, lossless
    assert chunks[0].start_sample == 0
    for a, b in zip(chunks, chunks[1:]):
        assert a.start_sample + len(a.samples) == b.start_sample
    assert sum(len(c.samples) for c in chunks) == len(audio)
    np.testing.assert_array_equal(np.concatenate([c.samples for c in chunks]), audio)
    # boundaries fall inside the silent gaps, never mid-"word"
    assert 9.0 <= chunks[1].start <= 10.0
    assert 19.5 <= chunks[2].start <= 20.5


def test_short_audio_is_a_single_chunk():
    audio = tone_with_gaps(SR, [("tone", 4)])
    chunks = list(silence_aware_chunks(_blocks(audio), SR, chunk_seconds=10, search_seconds=3))
    assert len(chunks) == 1 and len(chunks[0].samples) == len(audio)


def test_memory_is_bounded_by_chunk_size():
    # 2 minutes streamed in small blocks: no chunk is ever larger than target + search window
    audio = tone_with_gaps(SR, [("tone", 120)])
    chunks = list(silence_aware_chunks(_blocks(audio, 1600), SR, chunk_seconds=10, search_seconds=3))
    assert max(len(c.samples) for c in chunks) <= 13 * SR
    assert sum(len(c.samples) for c in chunks) == len(audio)
