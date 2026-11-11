import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.engines.mock_engine import MockEngine
from tests.conftest import tone_with_gaps, write_wav


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, MockEngine())) as c:
        yield c


def _upload(client, path: Path, data: dict | None = None, **params):
    with path.open("rb") as fh:
        return client.post("/v1/transcriptions", files={"file": (path.name, fh)}, data=data, params=params)


def _wait(client, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_health(client):
    assert client.get("/health").json()["engine"] == "mock"


def test_short_file_is_transcribed_synchronously(client, speechlike_wav, settings):
    r = _upload(client, speechlike_wav)
    assert r.status_code == 200
    body = r.json()
    assert body["segments"] and body["segments"][0]["start"] == 0.0
    assert body["audio"]["filename"] == "speech.wav"
    assert not any(settings.upload_dir.iterdir()), "uploaded audio must be deleted after processing"


def test_sync_response_can_be_srt(client, speechlike_wav):
    r = _upload(client, speechlike_wav, format="srt")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-subrip")
    assert "-->" in r.text


def test_long_file_becomes_async_job(client, tmp_path):
    long_wav = write_wav(tmp_path / "long.wav", tone_with_gaps(16_000, [("tone", 40)]), 16_000)
    r = _upload(client, long_wav)
    assert r.status_code == 202
    job_id = r.json()["id"]
    assert r.headers["location"] == f"/v1/jobs/{job_id}"

    job = _wait(client, job_id)
    assert job["status"] == "completed" and job["progress"] == 1.0
    assert job["result"]["chunks"] >= 4  # 40 s in ~10 s chunks
    assert job["result"]["duration_seconds"] == pytest.approx(40.0, abs=0.01)

    vtt = client.get(f"/v1/jobs/{job_id}/transcript", params={"format": "vtt"})
    assert vtt.status_code == 200 and vtt.text.startswith("WEBVTT")


def test_unsupported_extension_is_415(client, tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4")
    assert _upload(client, f).status_code == 415


def test_garbage_with_audio_extension_is_422(client, tmp_path):
    f = tmp_path / "fake.mp3"
    f.write_bytes(b"\x00\x01 definitely not mp3" * 200)
    r = _upload(client, f)
    assert r.status_code == 422
    assert "uploads" not in r.json()["detail"], "server paths must not leak into error messages"


def test_oversized_upload_is_413(settings, tmp_path, speechlike_wav):
    settings.max_upload_mb = 1  # speech.wav is ~4.4 MB
    with TestClient(create_app(settings, MockEngine())) as c:
        assert _upload(c, speechlike_wav).status_code == 413


def test_invalid_language_is_rejected_before_any_job_is_queued(client, speechlike_wav, settings):
    # Swagger UI pre-fills optional string fields with the literal "string"; that once produced
    # a 202 followed by a failed job. It must be an immediate, explanatory 422 instead.
    r = _upload(client, speechlike_wav, data={"language": "string"}, mode="async")
    assert r.status_code == 422
    assert "Unsupported language 'string'" in r.json()["detail"]
    assert not settings.upload_dir.exists() or not any(settings.upload_dir.iterdir())


@pytest.mark.parametrize(("sent", "expected"), [("auto", "en"), ("", "en"), ("EN", "en"), ("fr", "fr")])
def test_language_auto_empty_and_case_insensitive(client, speechlike_wav, sent, expected):
    r = _upload(client, speechlike_wav, data={"language": sent})
    assert r.status_code == 200
    assert r.json()["language"] == expected  # mock engine: auto-detect -> 'en', explicit code echoed


def test_unknown_job_is_404(client):
    assert client.get("/v1/jobs/nope").status_code == 404
