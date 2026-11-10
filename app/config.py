"""Runtime configuration, read from environment variables (prefix ``STT_``) or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STT_", env_file=".env", extra="ignore")

    # --- Speech-to-text engine -------------------------------------------------
    engine: str = Field("faster-whisper", description="'faster-whisper' or 'mock'")
    model_size: str = Field("base", description="tiny | base | small | medium | large-v3 | distil-large-v3 ...")
    device: str = Field("cpu", description="'cpu', 'cuda' or 'auto'")
    compute_type: str = Field("int8", description="int8 on CPU, float16 on GPU")
    cpu_threads: int = 0  # 0 = let CTranslate2 decide
    model_dir: Path | None = None  # where model weights are cached
    beam_size: int = 5
    vad_filter: bool = True  # Silero VAD: skip silence, reduces hallucinations
    word_timestamps: bool = False
    preload_model: bool = False  # load weights at API startup instead of on the first request

    # --- Long-audio handling ----------------------------------------------------
    chunk_seconds: float = 600.0  # target chunk length (10 min)
    chunk_search_seconds: float = 15.0  # look +/- this far from the target for a quiet cut point

    # --- Limits -----------------------------------------------------------------
    max_upload_mb: int = 500
    max_duration_seconds: float = 4 * 3600
    sync_max_duration_seconds: float = 60.0  # longer files are processed as background jobs
    job_workers: int = 1  # concurrent transcriptions (CPU bound)

    # --- Storage ----------------------------------------------------------------
    upload_dir: Path = Path("data/uploads")

    allowed_extensions: frozenset[str] = frozenset(
        {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".webm", ".wma", ".amr", ".mp4", ".mov", ".mkv"}
    )

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be > 0")
        if not 0 <= self.chunk_search_seconds < self.chunk_seconds / 2:
            raise ValueError("chunk_search_seconds must be >= 0 and < chunk_seconds / 2")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
