from pathlib import Path

import av
import numpy as np
import pytest

from app.audio import (
    InvalidAudioError,
    UnsupportedFormatError,
    check_extension,
    load_pcm,
    probe,
)
from tests.conftest import tone_with_gaps, write_wav


def test_probe_reads_header(speechlike_wav: Path):
    info = probe(speechlike_wav)
    assert info.container == "wav"
    assert info.sample_rate == 44_100
    assert info.channels == 2
    assert info.duration_seconds == pytest.approx(25.0, abs=0.01)


def test_decode_normalizes_to_16k_mono_float32(speechlike_wav: Path):
    pcm = load_pcm(speechlike_wav)
    assert pcm.dtype == np.float32
    assert pcm.ndim == 1
    assert len(pcm) / 16_000 == pytest.approx(25.0, abs=0.05)
    assert np.abs(pcm).max() <= 1.0


def _encode(tmp_path: Path, codec: str, ext: str, sr: int = 44_100) -> Path:
    """Encode a test tone with PyAV's bundled FFmpeg encoders."""
    samples = tone_with_gaps(sr, [("tone", 2), ("silence", 1), ("tone", 2)])
    path = tmp_path / f"tone{ext}"
    try:
        with av.open(str(path), "w") as out:
            stream = out.add_stream(codec, rate=sr)
            stream.layout = "mono"
            fmt = stream.codec_context.format.name
            frame_size = 1152
            for i in range(0, len(samples), frame_size):
                block = samples[i : i + frame_size]
                if fmt.startswith("s16"):
                    data = (block * 32767).astype(np.int16)[None, :]
                elif fmt.startswith("s32"):
                    data = (block * 2147483647).astype(np.int32)[None, :]
                else:
                    data = block[None, :]
                frame = av.AudioFrame.from_ndarray(data, format=fmt, layout="mono")
                frame.sample_rate = sr
                for pkt in stream.encode(frame):
                    out.mux(pkt)
            for pkt in stream.encode(None):
                out.mux(pkt)
    except (av.error.FFmpegError, ValueError) as exc:
        pytest.skip(f"encoder {codec} unavailable: {exc}")
    return path


@pytest.mark.parametrize(
    ("codec", "ext", "sr"),
    [("mp3", ".mp3", 44_100), ("flac", ".flac", 44_100), ("aac", ".m4a", 44_100), ("libopus", ".ogg", 48_000)],
)
def test_compressed_formats_decode_to_same_representation(tmp_path: Path, codec: str, ext: str, sr: int):
    path = _encode(tmp_path, codec, ext, sr)
    pcm = load_pcm(path)
    assert pcm.dtype == np.float32
    # encoders add a little padding/priming; duration must still be ~5 s
    assert len(pcm) / 16_000 == pytest.approx(5.0, abs=0.15)


def test_rejects_non_audio_content_even_with_audio_extension(tmp_path: Path):
    fake = tmp_path / "notes.wav"
    fake.write_text("this is not audio" * 100)
    with pytest.raises(InvalidAudioError):
        probe(fake)


def test_rejects_unknown_extension():
    with pytest.raises(UnsupportedFormatError):
        check_extension("report.pdf", frozenset({".wav", ".mp3"}))
    assert check_extension("Call.MP3", frozenset({".wav", ".mp3"})) == ".mp3"


def test_low_sample_rate_telephony_audio_is_upsampled(tmp_path: Path):
    path = write_wav(tmp_path / "phone.wav", tone_with_gaps(8_000, [("tone", 3)]), 8_000)
    pcm = load_pcm(path)
    assert len(pcm) / 16_000 == pytest.approx(3.0, abs=0.05)
