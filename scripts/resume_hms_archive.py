"""Resume the authorized Kaggle HMS competition ZIP with HTTP byte ranges.

Run after accepting competition rules and configuring ~/.kaggle/kaggle.json.
The archive is not trusted until `unzip -t` succeeds after this command.
"""
import argparse
from pathlib import Path
import re
import urllib.error
import urllib.request

from download_hms import AuthSafeRedirect, COMPETITION, auth_header


URL = "https://www.kaggle.com/api/v1/competitions/data/download-all/" + COMPETITION
RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def resume(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    offset = path.stat().st_size if path.exists() else 0
    headers = {"Authorization": auth_header(), "User-Agent": "ODEBrain-HMS-research/0.1"}
    if offset:
        headers["Range"] = "bytes=%d-" % offset
    request = urllib.request.Request(URL, headers=headers)
    try:
        response = urllib.request.build_opener(AuthSafeRedirect()).open(request, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise SystemExit("Kaggle HTTP 429; wait before retrying") from exc
        raise
    with response:
        if offset:
            match = RANGE_RE.fullmatch(response.headers.get("Content-Range", ""))
            if response.status != 206 or match is None or int(match.group(1)) != offset:
                raise ValueError("Server did not honor the requested byte range; archive left untouched")
            total = int(match.group(3))
            expected_bytes = total - offset
        else:
            if response.status != 200:
                raise ValueError("Expected HTTP 200 for a new archive")
            total = int(response.headers["Content-Length"])
            expected_bytes = total
        if response.headers.get("Content-Type", "").split(";")[0] != "application/zip":
            raise ValueError("Kaggle did not return a ZIP archive")
        received = 0
        with path.open("ab" if offset else "wb") as target:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                received += len(chunk)
                if received // (1024**3) != (received - len(chunk)) // (1024**3):
                    print("received %.2f / %.2f GiB" % ((offset + received) / 1024**3, total / 1024**3), flush=True)
            target.flush()
    if received != expected_bytes or path.stat().st_size != total:
        raise IOError("Incomplete archive: %d of %d bytes; rerun to resume" % (path.stat().st_size, total))
    print("Archive bytes complete: %d; run unzip -t before extraction" % total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/hms-harmful-brain-activity-classification.zip")
    args = parser.parse_args()
    resume(args.output)
