"""Verify published GitHub links and write the formal slide resource evidence.

This command performs anonymous HTTP GET requests. It writes the result only
after the repository responds 200 and the downloaded release ZIP exactly
matches the locally frozen archive and passes its per-file manifest checks.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.package_public_resources import sha256_file, verify  # noqa: E402


def checked_urls(repository_url: str, weights_url: str, archive_name: str) -> None:
    repository = urlparse(repository_url)
    weights = urlparse(weights_url)
    parts = repository.path.strip("/").split("/")
    if (repository.scheme != "https" or repository.hostname != "github.com" or
            len(parts) != 2 or not all(parts) or repository.query or repository.fragment):
        raise ValueError("Repository URL must be a public GitHub repository HTTPS URL")
    expected_prefix = "/" + "/".join(parts) + "/releases/download/"
    if (weights.scheme != "https" or weights.hostname != "github.com" or
            not weights.path.startswith(expected_prefix) or
            not weights.path.endswith("/" + archive_name) or
            weights.query or weights.fragment):
        raise ValueError("Weights URL must be a release ZIP under the same GitHub repository")


def get_repository(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "ODEBrain-HMS-resource-verifier/1.0",
                                    "Accept": "text/html"}, method="GET")
    with urlopen(request, timeout=45) as response:
        status = response.status
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
        response.read(512 * 1024)
    parsed_final = urlparse(final_url)
    if (status != 200 or parsed_final.hostname != "github.com" or
            not content_type.lower().startswith("text/html")):
        raise ValueError("Repository anonymous GET did not return a public GitHub HTML page")
    return {"url": url, "verified_http_status": status, "resolved_url": final_url}


def get_weights(url: str, destination: Path) -> dict:
    request = Request(url, headers={"User-Agent": "ODEBrain-HMS-resource-verifier/1.0",
                                    "Accept": "application/octet-stream"}, method="GET")
    with urlopen(request, timeout=120) as response, destination.open("wb") as output:
        status = response.status
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
        size = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > 256 * 1024 * 1024:
                raise ValueError("Release ZIP exceeded the 256 MiB resource limit")
            output.write(block)
    if status != 200 or size == 0:
        raise ValueError("Release ZIP anonymous GET did not return HTTP 200 content")
    return {"url": url, "verified_http_status": status, "resolved_url": final_url,
            "bytes": size, "content_type": content_type}


def create_evidence(repository_url: str, weights_url: str, archive: Path, output: Path) -> dict:
    archive = archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    checked_urls(repository_url, weights_url, archive.name)
    local = verify(archive)
    repository = get_repository(repository_url)
    with tempfile.TemporaryDirectory(prefix="odebrain-release-verify-") as temporary:
        downloaded = Path(temporary) / archive.name
        weights = get_weights(weights_url, downloaded)
        downloaded_sha256 = sha256_file(downloaded)
        if downloaded_sha256 != local["sha256"] or downloaded.stat().st_size != archive.stat().st_size:
            raise ValueError("Published ZIP does not match the locally frozen archive")
        verified_download = verify(downloaded)
        if verified_download["entry_count"] != local["entry_count"]:
            raise ValueError("Published ZIP manifest differs from the local archive")
    verified_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    repository["verified_at"] = verified_at
    weights.update({"verified_at": verified_at, "sha256": local["sha256"],
                    "entry_count": local["entry_count"]})
    evidence = {"experiment_status": "formal", "repository": repository, "weights": weights,
                "verification_method": "anonymous_http_get_and_full_zip_sha256",
                "local_frozen_archive": str(archive)}
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(output.suffix + ".partial")
    pending.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(output)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-url", required=True)
    parser.add_argument("--weights-url", required=True)
    parser.add_argument("--archive", type=Path,
                        default=ROOT / "artifacts/release/odebrain_hms_resources_v1.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "results/formal_resources.json")
    args = parser.parse_args()
    evidence = create_evidence(args.repository_url, args.weights_url, args.archive, args.output)
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
