"""Post-processing: turn raw engine segments into clean, downstream-ready output.

* whitespace normalization and empty-segment removal
* confidence flagging (Whisper's own thresholds), so downstream consumers / reviewers can
  find the parts most likely to be wrong
* export to JSON (machine use: search indexing, RAG chunking, LLM summarization),
  SRT / WebVTT (subtitles / players) and plain text
"""

from __future__ import annotations

import json

from .models import Segment, TranscriptionResult

# Same thresholds Whisper itself uses to decide a decode failed / a window is silence.
LOW_LOGPROB = -1.0
HIGH_NO_SPEECH = 0.6

EXPORT_FORMATS = ("json", "txt", "srt", "vtt")
MEDIA_TYPES = {
    "json": "application/json",
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
}


def clean_text(text: str) -> str:
    return " ".join(text.split())


def is_low_confidence(avg_logprob: float | None, no_speech_prob: float | None) -> bool:
    return (avg_logprob is not None and avg_logprob < LOW_LOGPROB) or (
        no_speech_prob is not None and no_speech_prob > HIGH_NO_SPEECH
    )


def _is_repetition(prev: Segment | None, text: str, start: float, max_gap: float = 2.0) -> bool:
    """Whisper sometimes re-emits the previous sentence on a short audio tail ("repetition
    hallucination"). An identical text immediately following itself is treated as one."""
    return prev is not None and prev.text.casefold() == text.casefold() and start - prev.end <= max_gap


def finalize_segments(segments: list[Segment]) -> list[Segment]:
    """Clean text, drop empties and back-to-back repetitions, enforce ordering and renumber ids."""
    out: list[Segment] = []
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        text = clean_text(seg.text)
        if not text or _is_repetition(out[-1] if out else None, text, seg.start):
            continue
        out.append(
            seg.model_copy(
                update={
                    "id": len(out),
                    "text": text,
                    "start": round(seg.start, 3),
                    "end": round(max(seg.end, seg.start), 3),
                    "low_confidence": is_low_confidence(seg.avg_logprob, seg.no_speech_prob),
                }
            )
        )
    return out


def _timestamp(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(result: TranscriptionResult) -> str:
    blocks = [
        f"{i}\n{_timestamp(s.start, ',')} --> {_timestamp(s.end, ',')}\n{s.text}\n"
        for i, s in enumerate(result.segments, start=1)
    ]
    return "\n".join(blocks)


def to_vtt(result: TranscriptionResult) -> str:
    blocks = [f"{_timestamp(s.start, '.')} --> {_timestamp(s.end, '.')}\n{s.text}\n" for s in result.segments]
    return "WEBVTT\n\n" + "\n".join(blocks)


def to_txt(result: TranscriptionResult) -> str:
    return "\n".join(f"[{_timestamp(s.start, '.')} - {_timestamp(s.end, '.')}] {s.text}" for s in result.segments) + "\n"


def to_json(result: TranscriptionResult) -> str:
    return json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False)


def export(result: TranscriptionResult, fmt: str) -> str:
    fmt = fmt.lower()
    if fmt == "json":
        return to_json(result)
    if fmt == "txt":
        return to_txt(result)
    if fmt == "srt":
        return to_srt(result)
    if fmt == "vtt":
        return to_vtt(result)
    raise ValueError(f"Unknown format '{fmt}'. Use one of {EXPORT_FORMATS}")
