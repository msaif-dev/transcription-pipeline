from ..config import Settings
from .base import EngineOutput, EngineSegment, EngineWord, TranscriptionEngine


def create_engine(settings: Settings) -> TranscriptionEngine:
    if settings.engine == "faster-whisper":
        from .faster_whisper_engine import FasterWhisperEngine

        return FasterWhisperEngine(settings)
    if settings.engine == "mock":
        from .mock_engine import MockEngine

        return MockEngine()
    raise ValueError(f"Unknown engine '{settings.engine}'. Use 'faster-whisper' or 'mock'.")


__all__ = ["create_engine", "EngineOutput", "EngineSegment", "EngineWord", "TranscriptionEngine"]
