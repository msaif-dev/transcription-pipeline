"""Command-line interface.

    python -m app.cli path/to/audio.mp3                      # JSON to stdout
    python -m app.cli talk.m4a -f srt -o talk.srt            # subtitles
    python -m app.cli call.wav --language en --words         # word-level timestamps
    python -m app.cli long.mp3 --model small --chunk-seconds 300
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .audio import AudioError, check_extension
from .config import Settings
from .engines import create_engine
from .pipeline import TranscriptionPipeline
from .postprocess import EXPORT_FORMATS, export


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="transcribe", description="Transcribe an audio file with timestamps per segment.")
    p.add_argument("input", type=Path, help="Audio file (WAV, MP3, M4A, FLAC, OGG, WebM, MP4, ...)")
    p.add_argument("-f", "--format", choices=EXPORT_FORMATS, default="json")
    p.add_argument("-o", "--output", type=Path, help="Write to this file instead of stdout")
    p.add_argument("--language", help="ISO-639-1 code (e.g. en), or 'auto' (default) to detect it")
    p.add_argument("--words", action="store_true", help="Include word-level timestamps")
    p.add_argument("--engine", choices=["faster-whisper", "mock"])
    p.add_argument("--model", help="Whisper model size, e.g. tiny, base, small, medium, large-v3")
    p.add_argument("--device", help="cpu | cuda | auto")
    p.add_argument("--chunk-seconds", type=float)
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s %(message)s")

    overrides = {
        k: v
        for k, v in {
            "engine": args.engine,
            "model_size": args.model,
            "device": args.device,
            "chunk_seconds": args.chunk_seconds,
        }.items()
        if v is not None
    }
    settings = Settings(**overrides)

    if not args.input.is_file():
        print(f"error: {args.input} not found", file=sys.stderr)
        return 2
    try:
        check_extension(args.input.name, settings.allowed_extensions)
        pipeline = TranscriptionPipeline(create_engine(settings), settings)

        def progress(fraction: float | None, chunks: int) -> None:
            if not args.quiet:
                pct = f"{fraction:.0%}" if fraction is not None else "?"
                print(f"  chunk {chunks} done ({pct})", file=sys.stderr)

        result = pipeline.run(args.input, language=args.language, word_timestamps=args.words, progress=progress)
    except AudioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    text = export(result, args.format)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        if not args.quiet:
            print(
                f"Wrote {args.output} - {len(result.segments)} segments, {result.duration_seconds:.1f}s audio, "
                f"language={result.language}, {result.processing_seconds:.1f}s (RTF {result.real_time_factor})",
                file=sys.stderr,
            )
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
