"""End-to-end API demo: upload a file, follow the async job, download the transcript.

    uvicorn app.main:app --port 8000          # in another terminal
    python scripts/demo_api.py samples/long_talk.mp3 --url http://localhost:8000
"""

import argparse
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("file", type=Path)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--mode", default="auto", choices=["auto", "sync", "async"])
    p.add_argument("--out", type=Path, help="save the SRT here")
    args = p.parse_args()

    with httpx.Client(base_url=args.url, timeout=600) as c:
        print("health:", c.get("/health").json())
        with args.file.open("rb") as fh:
            r = c.post("/v1/transcriptions", params={"mode": args.mode}, files={"file": (args.file.name, fh)})
        print("POST /v1/transcriptions ->", r.status_code)
        if r.status_code == 200:
            result = r.json()
        elif r.status_code == 202:
            location = r.headers["location"]
            print("job:", location)
            while True:
                job = c.get(location).json()
                print(f"  status={job['status']:<10} progress={job['progress']:.0%}")
                if job["status"] in ("completed", "failed"):
                    break
                time.sleep(2)
            if job["status"] == "failed":
                print("job failed:", job["error"])
                return 1
            result = job["result"]
            if args.out:
                srt = c.get(f"{location}/transcript", params={"format": "srt"}).text
                args.out.write_text(srt, encoding="utf-8")
                print("saved", args.out)
        else:
            print(r.text)
            return 1

    print(f"\nlanguage={result['language']} duration={result['duration_seconds']}s chunks={result['chunks']} "
          f"processing={result['processing_seconds']}s RTF={result['real_time_factor']}")
    for s in result["segments"]:
        print(f"[{s['start']:7.2f} - {s['end']:7.2f}] {s['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
