"""Background job execution for long files.

A bounded thread pool caps how many transcriptions run at once (inference is CPU/GPU bound;
unbounded concurrency just makes every request slower and can exhaust memory). Both the
synchronous endpoint and async jobs go through this same pool.

This in-memory store is deliberately simple for the exercise. In production the same interface
would sit on a durable queue + store (e.g. SQS/Celery/RQ + Redis/Postgres, uploads in S3) so jobs
survive restarts and workers scale horizontally.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .audio import AudioError
from .models import Job, JobStatus, TranscriptionResult
from .pipeline import TranscriptionPipeline

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobManager:
    def __init__(self, pipeline: TranscriptionPipeline, workers: int = 1):
        self._pipeline = pipeline
        self._executor = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="stt")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(
        self, path: Path, filename: str, *, language: str | None, word_timestamps: bool | None
    ) -> tuple[Job, Future]:
        now = _now()
        job = Job(id=uuid.uuid4().hex, status=JobStatus.queued, filename=filename, created_at=now, updated_at=now)
        with self._lock:
            self._jobs[job.id] = job
        future = self._executor.submit(self._run, job.id, path, filename, language, word_timestamps)
        return job, future

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy() if job else None

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs[job_id]
            self._jobs[job_id] = job.model_copy(update={**fields, "updated_at": _now()})

    def _run(
        self, job_id: str, path: Path, filename: str, language: str | None, word_timestamps: bool | None
    ) -> TranscriptionResult:
        self._update(job_id, status=JobStatus.processing)

        def on_progress(fraction: float | None, _chunks: int) -> None:
            if fraction is not None:
                self._update(job_id, progress=round(fraction, 4))

        try:
            result = self._pipeline.run(
                path, filename=filename, language=language, word_timestamps=word_timestamps, progress=on_progress
            )
            self._update(job_id, status=JobStatus.completed, progress=1.0, result=result)
            return result
        except AudioError as exc:
            self._update(job_id, status=JobStatus.failed, error=str(exc))
            raise
        except Exception as exc:  # noqa: BLE001 - record any failure on the job, then re-raise
            log.exception("job %s failed", job_id)
            self._update(job_id, status=JobStatus.failed, error=f"Internal error: {type(exc).__name__}")
            raise
        finally:
            Path(path).unlink(missing_ok=True)  # uploads are temporary; don't retain user audio

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
