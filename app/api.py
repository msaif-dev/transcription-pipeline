"""HTTP API.

POST /v1/transcriptions             upload audio; short files return the transcript (200),
                                    long files return a job (202 + Location header)
GET  /v1/jobs/{id}                  job status, progress and result
GET  /v1/jobs/{id}/transcript       result as json | txt | srt | vtt
GET  /health                        liveness + engine info

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from . import __version__
from .audio import AudioError, AudioTooLongError, UnsupportedFormatError, check_extension, probe
from .config import Settings, get_settings
from .engines import TranscriptionEngine, create_engine
from .jobs import JobManager
from .models import Job, JobStatus, TranscriptionResult
from .pipeline import TranscriptionPipeline
from .postprocess import MEDIA_TYPES, export

log = logging.getLogger(__name__)
Format = Literal["json", "txt", "srt", "vtt"]
_UPLOAD_BLOCK = 1024 * 1024


def create_app(settings: Settings | None = None, engine: TranscriptionEngine | None = None) -> FastAPI:
    settings = settings or get_settings()
    engine = engine or create_engine(settings)
    pipeline = TranscriptionPipeline(engine, settings)
    jobs = JobManager(pipeline, settings.job_workers)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        if settings.preload_model:
            await asyncio.to_thread(engine.warmup)
        yield
        jobs.shutdown()

    app = FastAPI(title="Transcription Pipeline", version=__version__, lifespan=lifespan)

    @app.exception_handler(AudioError)
    async def _audio_error(_: Request, exc: AudioError):
        if isinstance(exc, UnsupportedFormatError):
            code = 415
        elif isinstance(exc, AudioTooLongError):
            code = 413
        else:
            code = 422
        return JSONResponse(status_code=code, content={"detail": str(exc)})

    async def _save_upload(upload: UploadFile, ext: str) -> Path:
        """Stream the upload to disk in 1 MB blocks, enforcing the size limit as we go."""
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        dest = settings.upload_dir / f"{uuid.uuid4().hex}{ext}"
        written = 0
        try:
            with dest.open("wb") as fh:
                while block := await upload.read(_UPLOAD_BLOCK):
                    written += len(block)
                    if written > settings.max_upload_bytes:
                        raise HTTPException(413, f"File exceeds the {settings.max_upload_mb} MB limit")
                    fh.write(block)
            if written == 0:
                raise HTTPException(422, "Uploaded file is empty")
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
        return dest

    def _render(result: TranscriptionResult, fmt: str) -> Response:
        if fmt == "json":
            return JSONResponse(result.model_dump(mode="json"))
        return Response(export(result, fmt), media_type=MEDIA_TYPES[fmt])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "engine": engine.name, "model": engine.model_name, "version": __version__}

    @app.post(
        "/v1/transcriptions",
        responses={202: {"model": Job}, 413: {}, 415: {}, 422: {}},
        response_model=TranscriptionResult,
    )
    async def create_transcription(
        file: UploadFile = File(..., description="Audio file (WAV, MP3, M4A, FLAC, OGG, WebM, MP4...)"),
        language: str | None = Form(None, description="ISO-639-1 code, e.g. 'en'. Omit to auto-detect."),
        word_timestamps: bool | None = Form(None),
        mode: Literal["auto", "sync", "async"] = Query(
            "auto", description="auto: sync if duration <= STT_SYNC_MAX_DURATION_SECONDS, else async job"
        ),
        format: Format = Query("json", description="Response format for synchronous results"),
    ):
        filename = Path(file.filename or "upload").name
        ext = check_extension(filename, settings.allowed_extensions)
        path = await _save_upload(file, ext)
        try:
            info = await asyncio.to_thread(probe, path, filename)  # rejects non-audio before queueing
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        if info.duration_seconds is not None and info.duration_seconds > settings.max_duration_seconds:
            path.unlink(missing_ok=True)
            raise AudioTooLongError(f"Audio is {info.duration_seconds:.0f}s; the limit is {settings.max_duration_seconds:.0f}s")

        run_async = mode == "async" or (
            mode == "auto"
            and (info.duration_seconds is None or info.duration_seconds > settings.sync_max_duration_seconds)
        )
        job, future = jobs.submit(path, filename, language=language, word_timestamps=word_timestamps)

        if run_async:
            return JSONResponse(
                status_code=202,
                content=job.model_dump(mode="json"),
                headers={"Location": f"/v1/jobs/{job.id}"},
            )

        result = await asyncio.wrap_future(future)  # AudioError -> handler above; others -> 500
        return _render(result, format)

    @app.get("/v1/jobs/{job_id}", response_model=Job)
    def get_job(job_id: str) -> Job:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    @app.get("/v1/jobs/{job_id}/transcript")
    def get_transcript(job_id: str, format: Format = Query("json")) -> Response:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        if job.status == JobStatus.failed:
            raise HTTPException(409, f"Job failed: {job.error}")
        if job.status != JobStatus.completed or job.result is None:
            raise HTTPException(409, f"Job is {job.status.value}; try again later")
        return _render(job.result, format)

    return app
