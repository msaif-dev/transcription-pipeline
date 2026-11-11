from pathlib import Path

import pytest

from app.audio import AudioTooLongError, InvalidOptionError
from app.engines.mock_engine import MockEngine
from app.models import Segment, TranscriptionResult
from app.pipeline import TranscriptionPipeline
from app.postprocess import export, finalize_segments


def test_timestamps_are_mapped_back_to_the_original_file(speechlike_wav: Path, settings):
    engine = MockEngine()
    progress = []
    result = TranscriptionPipeline(engine, settings).run(
        speechlike_wav, progress=lambda f, n: progress.append((f, n))
    )

    assert result.chunks == engine.calls == 3  # 25 s file, 10 s chunks
    assert result.duration_seconds == pytest.approx(25.0, abs=0.05)
    assert result.language == "en"
    starts = [s.start for s in result.segments]
    assert starts == sorted(starts)
    assert max(s.end for s in result.segments) <= result.duration_seconds + 1e-6
    # segments from later chunks have absolute (not chunk-relative) timestamps
    assert any(s.start >= 20 for s in result.segments)
    assert [s.id for s in result.segments] == list(range(len(result.segments)))
    assert progress[-1] == (1.0, 3)


def test_duration_limit_enforced(speechlike_wav: Path, settings):
    settings.max_duration_seconds = 10
    with pytest.raises(AudioTooLongError):
        TranscriptionPipeline(MockEngine(), settings).run(speechlike_wav)


def test_invalid_language_raises_clear_error(speechlike_wav: Path, settings):
    engine = MockEngine()
    with pytest.raises(InvalidOptionError, match="Unsupported language 'klingon'"):
        TranscriptionPipeline(engine, settings).run(speechlike_wav, language="klingon")
    assert engine.calls == 0  # rejected before any transcription work


def test_finalize_cleans_and_flags_low_confidence():
    raw = [
        Segment(id=9, start=5.0, end=6.0, text="  second   one ", avg_logprob=-0.2, no_speech_prob=0.1),
        Segment(id=9, start=0.0, end=1.0, text="first", avg_logprob=-1.4, no_speech_prob=0.1),
        Segment(id=9, start=2.0, end=3.0, text="   ", avg_logprob=-0.1, no_speech_prob=0.1),
    ]
    out = finalize_segments(raw)
    assert [s.text for s in out] == ["first", "second one"]
    assert [s.id for s in out] == [0, 1]
    assert out[0].low_confidence and not out[1].low_confidence


def test_repetition_hallucination_is_dropped():
    raw = [
        Segment(id=0, start=178.9, end=185.9, text="Record how long each stage takes."),
        Segment(id=1, start=185.9, end=186.4, text=" Record how long each stage takes."),
        Segment(id=2, start=300.0, end=302.0, text="Record how long each stage takes."),  # far later: legit
    ]
    out = finalize_segments(raw)
    assert [s.start for s in out] == [178.9, 300.0]


def test_subtitle_exports(speechlike_wav: Path, settings):
    result: TranscriptionResult = TranscriptionPipeline(MockEngine(), settings).run(speechlike_wav)
    srt = export(result, "srt")
    vtt = export(result, "vtt")
    assert srt.startswith("1\n00:00:00,000 --> ")
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> ")
    assert export(result, "txt").count("\n") == len(result.segments)
