# Transcription Pipeline

A small, production minded service that turns an audio file (or audio in a video file) into **timestamped text** that's ready for downstream use (search, subtitles, RAG/LLM processing).

- **Accepts** WAV, MP3, M4A/AAC, FLAC, OGG/Opus, WebM, WMA, AMR, and the audio track of MP4/MOV/MKV
- **Transcribes** with open source Whisper via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, int8 on CPU, Silero VAD)
- **Returns** segments with start/end timestamps (optionally word level), language, confidence flags, as JSON / SRT / WebVTT / TXT
- **Handles long audio** by streaming decode + silence aware chunking + async jobs with progress
- Exposed as a **FastAPI service** and a **CLI**; tested with a mock engine (no model download needed for tests)

Contents: [Architecture](#architecture) · [Quick start](#quick-start) · [API](#api) · [Measured results](#measured-results-laptop-cpu-intel-i7-8650u-no-gpu-base-model-int8) · [Key design decisions](#key-design-decisions) · [System design: production](#system-design-taking-this-to-production)

## Architecture

```
            ┌───────────────┐   ┌──────────────────────┐   ┌──────────────────────┐   ┌────────────────┐   ┌───────────────────┐
 upload ──► │ 1. Validate   │──►│ 2. Stream decode &   │──►│ 3. Silence aware     │──►│ 4. Transcribe  │──►│ 5. Postprocess    │──► JSON/SRT/VTT/TXT
 (API/CLI)  │ ext + size +  │   │ normalize (PyAV)     │   │ chunking (~10 min,   │   │ (engine        │   │ offset timestamps,│
            │ content probe │   │ any codec → 16 kHz   │   │ cut at quietest      │   │ interface:     │   │ clean, dedupe,    │
            │ + duration    │   │ mono float32 blocks  │   │ 100 ms near target)  │   │ faster-whisper │   │ confidence flags, │
            └───────────────┘   └──────────────────────┘   └──────────────────────┘   │ / mock / ...)  │   │ exports           │
                                                                                      └────────────────┘   └───────────────────┘
 Short files (≤ 60 s) → synchronous 200 response.  Longer files → 202 + job id → poll /v1/jobs/{id} (progress per chunk).
 Both paths share one bounded worker pool, so concurrent requests can't overload the CPU/GPU.
```

| Module | Responsibility |
|---|---|
| `app/audio.py` | Extension allowlist, content probing (don't trust the extension), streaming decode + resample to 16 kHz mono via PyAV (bundled FFmpeg, so no system install), tolerant of a few corrupt packets |
| `app/chunking.py` | Splits the decoded stream into chunks at the quietest point near each target boundary, so words are never cut in half. Memory stays bounded at about one chunk |
| `app/engines/` | `TranscriptionEngine` interface; `FasterWhisperEngine` (real) and `MockEngine` (tests/dev). Adding OpenAI / Deepgram / AWS Transcribe means adding one class |
| `app/pipeline.py` | Orchestration: limits, language lock after the first speech chunk, chunk to file timestamp mapping, progress, metrics (RTF) |
| `app/postprocess.py` | Whitespace cleanup, drop empty and consecutive duplicate (hallucinated) segments, low confidence flags, SRT/VTT/TXT/JSON export |
| `app/jobs.py` | Bounded thread pool job runner with status/progress; deletes uploaded audio after processing |
| `app/api.py` | FastAPI endpoints, streaming upload with size limit, sync/async routing, error to HTTP mapping |
| `app/cli.py` | Command line entry point |

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

# CLI (the first run downloads the Whisper "base" model, ~140 MB)
python -m app.cli samples/meeting.wav                     # JSON to stdout
python -m app.cli samples/long_talk.mp3 -f srt -o talk.srt
python -m app.cli call.wav --language en --words --model small

# API
uvicorn app.main:app --port 8000                          # interactive docs at http://localhost:8000/docs
curl -F "file=@samples/meeting.wav" "http://localhost:8000/v1/transcriptions"
curl -F "file=@samples/meeting.wav" "http://localhost:8000/v1/transcriptions?format=srt"
curl -F "file=@samples/long_talk.mp3" "http://localhost:8000/v1/transcriptions?mode=async"   # → 202 + Location
curl "http://localhost:8000/v1/jobs/<id>"                                                  # status + progress
curl "http://localhost:8000/v1/jobs/<id>/transcript?format=vtt"
python scripts/demo_api.py samples/long_talk.mp3                                           # scripted demo

# Tests (mock engine, no model download, a few seconds)
pytest

# Docker Compose (easiest): one command builds and starts the service, then open http://localhost:8000/docs
docker compose up -d --build        # logs: docker compose logs -f     stop: docker compose down
# Settings via env or .env file, e.g. PORT=8080, MODEL_SIZE=small (rebuild), STT_SYNC_MAX_DURATION_SECONDS=120

# Plain Docker: model baked into the image (~890 MB), runs as a non-root user, needs no internet at runtime,
# includes a HEALTHCHECK on /health
docker build --build-arg MODEL_SIZE=base -t transcriber . && docker run -p 8000:8000 transcriber
```

`samples/` contains mock speech generated with Windows' built in TTS (`scripts/make_samples.ps1`). `samples/output/` contains the real outputs produced by the pipeline.

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/transcriptions` | multipart `file`, optional form `language` (ISO-639-1 code, or `auto` = default; invalid codes get an immediate 422), `word_timestamps`; query `mode=auto\|sync\|async`, `format=json\|txt\|srt\|vtt`. `auto` = sync if ≤ `STT_SYNC_MAX_DURATION_SECONDS`, otherwise a job |
| `GET` | `/v1/jobs/{id}` | `queued → processing → completed/failed`, `progress` 0–1, `result` |
| `GET` | `/v1/jobs/{id}/transcript?format=` | 409 until completed |
| `GET` | `/health` | engine + model |

Errors: `415` unsupported type, `413` too large / too long, `422` not decodable / empty / no audio stream, `404` unknown job.

Example result (abridged, real output from `samples/meeting.wav`):

```json
{
  "text": "Good morning everyone, and thanks for joining the quarterly product review. Today we will cover ...",
  "language": "en", "language_probability": 0.9994, "duration_seconds": 42.473,
  "segments": [
    {"id": 0, "start": 0.0,  "end": 4.24,  "text": "Good morning everyone, and thanks for joining the quarterly product review.", "avg_logprob": -0.13, "no_speech_prob": 0.01, "low_confidence": false},
    {"id": 1, "start": 5.68, "end": 11.52, "text": "Today we will cover three topics, customer feedback, the release schedule, and hiring.", "...": "..."}
  ],
  "audio": {"filename": "meeting.wav", "container": "wav", "codec": "pcm_s16le", "sample_rate": 22050, "channels": 1, "duration_seconds": 42.473, "size_bytes": 1873094},
  "engine": "faster-whisper", "model": "base", "chunks": 1, "processing_seconds": 5.836, "real_time_factor": 0.1374
}
```

## Configuration

All settings live in `app/config.py` and can be overridden with `STT_*` env vars or `.env` (see `.env.example`): model size/device/compute type, VAD, word timestamps, chunk length and search window, upload/duration limits, sync threshold, worker count.

## Measured results (laptop CPU: Intel i7-8650U, no GPU, `base` model, int8)

| Input | Duration | Processing | RTF | Chunks |
|---|---|---|---|---|
| `meeting.wav` (22 kHz WAV) | 42.5 s | 5.8 s | 0.14 | 1 |
| `long_talk.mp3` (via API, async job, 60 s chunks) | 186.4 s | 19.9 s | 0.11 | 4, cut at 50.0 / 103.1 / 155.8 s, each in a pause between sentences |
| Real lecture **video** (`.mkv`, 72 MB, Indian English speaker, in Docker, word timestamps on) | 22 min 26 s | 3 min 19 s | 0.15 | 3 (default 10 min), cut at 601.2 / 1213.1 s, both in speaker pauses: no lost or duplicated words; 188 segments, 3,806 words, language `en` (p = 0.99) |

RTF = processing time ÷ audio length, so 0.11 is about 9x faster than real time. A 1 hour file is about 7 min on this laptop. On a GPU (`STT_DEVICE=cuda STT_COMPUTE_TYPE=float16`) it's much faster, even with `large-v3`.

## Key design decisions

1. **Normalize once, at the edge.** Every format is decoded and resampled into one representation (16 kHz mono float32). Everything after that has a single code path. Content is probed, so a renamed PDF is rejected even with a `.mp3` extension.
2. **Stream, don't load.** Decoding yields small blocks and the chunker holds at most about one chunk. Memory stays flat whether the file is 1 minute or 4 hours.
3. **Cut chunks in silence, not at fixed offsets.** Fixed cuts split words and cause errors or duplicates at every boundary. Overlap and merge is fragile. Cutting at the quietest 100 ms near the target keeps chunks disjoint and lossless, and each chunk's offset maps timestamps back to the original file.
4. **Lock the language** after the first chunk that contains speech, so a long file isn't transcribed in different languages chunk to chunk.
5. **Sync for short files, async jobs for long ones**, through one bounded worker pool. This prevents HTTP timeouts and keeps the CPU/GPU from being oversubscribed. Progress is reported per chunk.
6. **Engine behind an interface.** The STT backend is a config switch. A mock engine makes the test suite fast and deterministic.
7. **Output built for downstream use**: segment timestamps, confidence flags (Whisper's own `avg_logprob < -1` / `no_speech_prob > 0.6` thresholds), language, metrics, and standard subtitle formats.
8. **Privacy by default.** The prototype deletes uploaded audio after processing. In production, audio is kept in S3 under a retention policy so retries and reprocessing are possible (see System design). Errors don't leak server paths. The service runs fully offline once the model is cached, so audio never leaves your infrastructure.

## System design: taking this to production

The repo is a single process prototype. Its in memory job store and local temp files work for one machine, but they lose jobs on restart and can't scale out. The design below keeps the same pipeline code (`app/pipeline.py` and everything under it) and changes what surrounds it. Examples use AWS; the same shape works on GCP or Azure.

**What the prototype already does vs. what is design only** (prototype behavior verified against the running Docker service):

| Topic | ✅ Built and tested in this repo | 📐 Production design (below) |
|---|---|---|
| Concurrent uploads | Async endpoints; streamed uploads (1 MB blocks, size limit, unique names); one bounded worker pool for sync and async. 4 simultaneous uploads: all `200`, processed in order, `/health` stayed within 10 ms | Presigned direct to S3 uploads, SQS queue, autoscaled GPU workers, priority lane, rate limits, idempotency / content hash deduplication, backpressure (`429`) |
| Storage | Uploads stored temporarily and deleted after processing (verified: upload dir empty). Results kept in memory; **lost on restart** (verified: `404` after restart) | S3 for audio (with lifecycle) and versioned transcript JSON, Postgres for job/chunk state and segments, vector DB for RAG |
| Failure handling | Permanent input errors → immediate `413/415/422`; unexpected errors → job `failed` + logged traceback; tolerates a few corrupt packets; Docker automatically restarts the service if it exits (verified) | Retries with backoff + DLQ, heartbeats and a stuck job reaper, idempotent writes, per chunk checkpoints, model fallbacks, retry endpoint. Audio must be kept (the prototype deletes it even on failure, so jobs can't be retried) |
| API | `POST /v1/transcriptions` (auto sync/async, 4 output formats), `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/transcript`, `GET /health`, OpenAPI `/docs`, status codes 200/202/404/409/413/415/422 | Auth, rate limits, presigned uploads, webhooks, list/retry/delete, problem+json errors, `/v1/transcriptions/{id}` naming |

> **📐 Everything below in this section is design only. It is NOT implemented in this repo.**
> It answers the assessment's system design questions, which asked for thoughts rather than code. The endpoints that actually exist are the four in the [API](#api) section above (`GET /health`, `POST /v1/transcriptions`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/transcript`), and they are the ones listed at `/docs`.

```
 client ──HTTPS──► API Gateway / ALB ──► API service (FastAPI, stateless, autoscaled)
   │               auth, rate limits,      • POST /v1/uploads        → presigned S3 URL
   │               quotas, WAF             • POST /v1/transcriptions → job row + queue message
   │                                       • GET status / transcript / list, DELETE
   │                                             │                    │
   │                                             ▼                    ▼
   │                                     Postgres                SQS queue ──(3 failures)──► DLQ ──► alarm
   │                                     jobs, chunks, segments,      │
   │                                     idempotency keys             ▼
   │                                             ▲              GPU workers (ECS/EKS, spot, autoscale on queue depth)
   │                                             └───────────── validate → decode → chunk → STT → postprocess
   │                                                            (status + per chunk checkpoints)
   │                                                                  ▲   │
   └──PUT audio (direct, multipart)──► S3 raw/ ───────────────────────┘   │
                                       S3 transcripts/ ◄──────────────────┤
                                                                          ├──► webhook: job.completed / job.failed (HMAC signed)
                                                                          └──► optional: embeddings → vector DB (RAG, timestamp citations)
```

### 1. Concurrent uploads

- **Audio bytes never go through the API servers.** `POST /v1/uploads` returns a short lived **presigned S3 (multipart) URL**. The client uploads directly to S3, which handles thousands of parallel uploads and lets large uploads resume. API pods only handle small JSON requests, so they stay stateless and cheap to scale. (The prototype streams uploads to disk in 1 MB blocks, which is fine for one node.)
- **A queue separates intake from processing.** Creating a job is only a DB insert plus an SQS message, so a burst of 1,000 uploads is absorbed by the queue rather than by overloaded servers. GPU workers pull at their own pace (backpressure). Worker count **autoscales on queue depth / backlog per worker**, and spot instances keep costs down.
- **Bounded concurrency per worker** (1–2 jobs per GPU, as in `JobManager` today) avoids out of memory errors and thrashing. A long file is **split into chunk tasks** that several workers process in parallel, then reassembled in order.
- **Fairness:** short files go to a priority lane so a 30 s clip isn't stuck behind a 4 hour podcast. Per tenant rate limits and quotas at the gateway return `429` + `Retry-After`.
- **Duplicates:** an `Idempotency-Key` header makes client retries safe, and a **SHA-256 content hash** recognizes reuploads of the same file, returning the existing transcript for the same model and options instead of paying to transcribe it again.

### 2. Storing audio and transcripts

Each kind of data goes in the store that suits it:

| Data | Store | Why / details |
|---|---|---|
| Original audio | **S3** `raw/{tenant}/{job_id}/original.{ext}` | Large, immutable, cheap. Private bucket, SSE-KMS encryption, presigned access only. **Lifecycle policy**: kept while the job may need retries or reprocessing, then moved to cold storage or deleted (e.g. 30 days, set per tenant). `DELETE` endpoint for GDPR style erasure |
| Transcript (canonical) | **S3** `transcripts/{job_id}/{model_version}/transcript.json` | The JSON result is the contract. **Versioned by model**, so rerunning with a better model never overwrites the old result. SRT/VTT/TXT are rendered from it on request and cached |
| Job state and metadata | **Postgres** (RDS) | `jobs` (id, tenant, status, attempts, audio_key, sha256, duration, codec, model, language, error_code, timestamps) and `chunks` (job_id, index, start/end, status, attempts, result_key). Transactions make status changes safe; also supports listing, filtering and billing |
| Segments for querying | **Postgres** `segments` (+ full text index), or OpenSearch at scale | "Find where the customer said *refund*" → jump to the timestamp |
| AI / RAG | **Vector DB** (pgvector / Pinecone) | Segments grouped into ~30–60 s windows, embedded with `{job_id, start, end}` metadata, so an LLM answer can cite the exact moment in the recording |

Audio and transcripts can contain personal data. Everything is encrypted at rest and in transit, isolated per tenant, kept out of logs, and retained only as long as configured.

### 3. Retrying and recovering failed transcriptions

- **Separate permanent from transient failures.** Bad input (not audio, corrupt, too long) is **permanent**: fail fast, never retry, and return a clear error code. This is already how `AudioError` works. Worker crashes, out of memory, spot interruptions, S3 or third party timeouts, and `429`/`5xx` errors are **transient**: retry them.
- **Retries with exponential backoff + jitter**, up to 3 attempts, then the message goes to a **dead letter queue** and raises an alarm. Once the cause is fixed, DLQ messages are inspected and redriven.
- **Crash recovery:** SQS is at least once delivery. Workers **heartbeat** to extend the message's visibility timeout during long jobs. If a worker dies, the message becomes visible again and another worker takes over. A reaper requeues any job marked `processing` whose heartbeat has gone stale.
- **Idempotent work:** deterministic output keys (`job_id/chunk_n`) and conditional status updates (`UPDATE … WHERE status='processing' AND attempt=:n`) mean a duplicate delivery can never corrupt or double write a result.
- **Checkpoint per chunk:** each chunk's result is persisted as it completes, so a retry processes **only the missing chunks**. A 3 hour file that fails at chunk 17 doesn't start again from zero. Silence aware chunking is deterministic, so chunk boundaries are identical on every retry.
- **Fallbacks:** if the large model runs out of memory, retry with a smaller model or on CPU. If an external STT API is down, fall back to the local engine (the engine interface makes this a config change). The result records which model actually produced it.
- **Quality failures:** an empty transcript or one that's mostly low confidence isn't treated as a quiet success. It's flagged for review or rerun with a larger model.
- **Visibility to users:** jobs show `failed` with an error code (the prototype already does this), a `POST /v1/transcriptions/{id}/retry` endpoint would allow manual retries, and a `job.failed` webhook would fire. Metrics would track failure rate, attempts and DLQ depth.

### 4. Exposing it as an API

A versioned REST API built around an **asynchronous job resource**, documented with OpenAPI (FastAPI generates the docs at `/docs`). This is the **target production API**. The last column shows what exists in this repo today:

| Method | Path | Purpose | Built in this repo? |
|---|---|---|---|
| `POST` | `/v1/uploads` | Get a presigned URL for large files | ❌ Not built |
| `POST` | `/v1/transcriptions` | Create a job from `upload_id`, `audio_url` or a small multipart file. Options: `language`, `word_timestamps`, `model`, `webhook_url`. Header: `Idempotency-Key`. Returns **202** + `Location` (short clips can use `?mode=sync` → 200) | ⚠️ Partly: multipart upload, `language`, `word_timestamps`, `mode=auto\|sync\|async` (202 + `Location` or 200) and `format` are built. `upload_id`, `audio_url`, `model`, `webhook_url` and `Idempotency-Key` are not |
| `GET` | `/v1/transcriptions/{id}` | Status, progress, error, result links | ✅ Built as `GET /v1/jobs/{id}` |
| `GET` | `/v1/transcriptions/{id}/transcript?format=json\|srt\|vtt\|txt` | The transcript | ✅ Built as `GET /v1/jobs/{id}/transcript?format=…` |
| `GET` | `/v1/transcriptions?status=&cursor=` | List jobs (cursor pagination) | ❌ Not built |
| `POST` | `/v1/transcriptions/{id}/retry` | Retry a failed job | ❌ Not built |
| `DELETE` | `/v1/transcriptions/{id}` | Delete audio and transcript (privacy) | ❌ Not built (the prototype already deletes uploaded audio after processing) |
| `GET` | `/health` | Liveness check | ✅ Built |

- **Notifications:** polling, plus **webhooks** (`job.completed` / `job.failed`, HMAC signed, retried) so clients don't need to poll. Server sent events are optional for live progress. Live/streaming transcription would be a separate WebSocket API.
- **Security:** API keys or OAuth2/JWT validated at the gateway, tenant isolation on every query, TLS, file size and type limits, short lived presigned URLs.
- **Consistent errors:** `application/problem+json` with stable error codes. `413`/`415`/`422` for bad input (as already implemented), `429` with `Retry-After`, `404`, `409` (transcript not ready).
- **Operations:** the API and GPU workers are deployed and scaled separately. Health and readiness probes, Prometheus metrics (queue depth, real time factor, p95 latency, failure rate), OpenTelemetry tracing from upload to transcript, and structured logs without any audio content.

Notifications, auth, rate limits, problem+json errors, metrics and tracing above are **design only**. The prototype already implements the core of this contract: `POST /v1/transcriptions` with automatic sync/async routing, job polling (`/v1/jobs/{id}`), multiformat transcript download, a health check, and the 404/409/413/415/422 error mapping (with FastAPI's standard `{"detail": ...}` error body).

### Further extensions

- Speaker diarization (pyannote / WhisperX) adding a `speaker` field to each segment; custom vocabulary via `initial_prompt` for domain terms
- LLM postprocessing on the JSON: summaries, action items, PII redaction, with segment timestamps used as citations
- Accuracy tracking: word error rate (WER) on a labelled evaluation set for each model version before rollout
