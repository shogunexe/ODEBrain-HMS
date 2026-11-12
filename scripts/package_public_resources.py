"""Freeze and verify the small public HMS figure and final-weight bundle.

The bundle contains six selected case figures derived from the Kaggle HMS
data, so its notice retains the upstream CC BY-NC 4.0 terms. EEG arrays stay
local and must be regenerated after downloading the data from Kaggle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = PurePosixPath("results/dynamics_refit")
RUN_ROOT = PurePosixPath("artifacts/refit")
RUNS = (
    "tcn", "spectrogram_cnn", "fusion", "odebrain_inspired_hms",
    "graph_gru", "ode_pretrain", "graph_gru_pretrain",
)
RUN_FILES = (
    "final.pt", "config.json", "graph_normalizer.npz", "completed.json",
    "normalizer_record.json", "validation_record.json", "window_snapshot.json",
)
ROOT_NOTICES = (PurePosixPath("LICENSE.md"), PurePosixPath("THIRD_PARTY_NOTICES.md"))
MANIFEST_NAME = "PUBLIC_RESOURCE_MANIFEST.json"
NOTICE_NAME = "PUBLIC_RESOURCE_NOTICE.txt"
NOTICE = """ODEBrain-inspired HMS project resource bundle

Contains visualizations for six selected EEG cases derived from the
HMS - Harmful Brain Activity Classification dataset, plus seven project-trained
final checkpoints (five classifiers and two forecast pretraining checkpoints).
It does not include raw EEG or case NPZ arrays. The interactive dashboard
requires those arrays, which users must regenerate locally from Kaggle data.

Upstream dataset: HMS - Harmful Brain Activity Classification, Kaggle (2024),
Jin Jing, Zhen Lin, Chaoqi Yang, Ashley Chow, Sohier Dane, Jimeng Sun,
and M. Brandon Westover.
https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data
Kaggle labels the dataset Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0): https://creativecommons.org/licenses/by-nc/4.0/
The case figures are transformed, selected views of that data.
Use them only under the upstream terms, including attribution and noncommercial
use. The original data creators and Kaggle do not endorse this project.
The project-trained checkpoints and original project code are provided for
inspection and reproducibility. No general license to copy, modify,
redistribute, or use them in another project has been granted. See the bundled
LICENSE.md and THIRD_PARTY_NOTICES.md. The checkpoints are not clinical tools.

Original data and competition rules remain at the Kaggle URL above. Download
the full dataset there, after accepting its rules, to reproduce training.
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("Unsafe bundle path: " + value)
    return path


def collect_case_paths(root: Path) -> list[PurePosixPath]:
    index_path = root / CASE_ROOT / "result_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("provenance") != "real_hms_checkpoint" or len(index.get("cases", [])) != 6:
        raise ValueError("Expected six formal HMS checkpoint cases")
    files = {CASE_ROOT / "result_index.json"}
    for case in index["cases"]:
        if case.get("split") not in ("test", "val"):
            raise ValueError("Only validation/test demonstration cases are permitted")
        case_id = safe_relative(case["id"])
        if len(case_id.parts) != 1:
            raise ValueError("Unexpected case ID")
        metadata = CASE_ROOT / safe_relative(case["metadata"])
        if metadata.parts[:2] != CASE_ROOT.parts:
            raise ValueError("Case metadata path escaped result root")
        files.add(metadata)
        meta = json.loads((root / CASE_ROOT / case["metadata"]).read_text(encoding="utf-8"))
        if meta.get("provenance") != "real_hms_checkpoint" or meta.get("split_hash") != index.get("split_hash"):
            raise ValueError("Case and index provenance mismatch")
        if meta.get("checkpoint_sha256") != index.get("checkpoint_sha256"):
            raise ValueError("Case and index checkpoint mismatch")
        for variants in meta.get("figures", {}).values():
            for ext in ("png", "svg"):
                name = safe_relative(variants[ext])
                if len(name.parts) != 1 or name.suffix != "." + ext:
                    raise ValueError("Unexpected figure path")
                files.add(CASE_ROOT / "cases" / case_id / name)
        if len(meta.get("figures", {})) != 6:
            raise ValueError("Expected six figure types for every case")
    if len(files) != 1 + 6 * (1 + 12):
        raise ValueError("Unexpected case resource count")
    return sorted(files)


def collect_paths(root: Path) -> tuple[list[PurePosixPath], dict]:
    paths = collect_case_paths(root)
    paths.extend(ROOT_NOTICES)
    index = json.loads((root / CASE_ROOT / "result_index.json").read_text(encoding="utf-8"))
    weights = {}
    for run in RUNS:
        base = RUN_ROOT / run
        completed = json.loads((root / base / "completed.json").read_text(encoding="utf-8"))
        checkpoint = root / base / "final.pt"
        digest = sha256_file(checkpoint)
        if completed.get("final_checkpoint_sha256") != digest:
            raise ValueError("Completion marker/checkpoint mismatch: " + run)
        weights[run] = digest
        paths.extend(base / name for name in RUN_FILES)
    if weights["odebrain_inspired_hms"] != index.get("checkpoint_sha256"):
        raise ValueError("Released ODE checkpoint differs from dynamics cases")
    for path in paths:
        source = root.joinpath(*path.parts)
        if not source.is_file() or source.is_symlink():
            raise ValueError("Missing or linked release input: " + str(path))
    return sorted(paths), {"split_hash": index["split_hash"], "weights_sha256": weights,
                           "case_count": len(index["cases"])}


def build(root: Path, archive: Path) -> dict:
    paths, provenance = collect_paths(root)
    archive.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for relative in paths:
        source = root.joinpath(*relative.parts)
        entries.append({"path": str(relative), "sha256": sha256_file(source), "bytes": source.stat().st_size})
    manifest = {"schema_version": "odebrain_public_resources_v1", "provenance": provenance,
                "entries": entries, "notice_sha256": sha256_bytes(NOTICE.encode("utf-8"))}
    pending = archive.with_suffix(archive.suffix + ".partial")
    with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as target:
        for relative in paths:
            source = root.joinpath(*relative.parts)
            info = zipfile.ZipInfo(str(relative), date_time=(2024, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            target.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED,
                            compresslevel=6)
        for name, content in ((NOTICE_NAME, NOTICE.encode("utf-8")),
                              (MANIFEST_NAME, (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))):
            info = zipfile.ZipInfo(name, date_time=(2024, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            target.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED,
                            compresslevel=6)
    pending.replace(archive)
    return {"archive": str(archive), "sha256": sha256_file(archive), "bytes": archive.stat().st_size,
            "entry_count": len(entries), "provenance": provenance}


def verify(archive: Path, destination: Optional[Path] = None) -> dict:
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate paths in archive")
        if MANIFEST_NAME not in names or NOTICE_NAME not in names:
            raise ValueError("Archive lacks manifest or attribution notice")
        manifest = json.loads(source.read(MANIFEST_NAME))
        if manifest.get("schema_version") != "odebrain_public_resources_v1":
            raise ValueError("Unknown resource manifest schema")
        if sha256_bytes(source.read(NOTICE_NAME)) != manifest.get("notice_sha256"):
            raise ValueError("Attribution notice checksum mismatch")
        recorded = {row["path"]: row for row in manifest["entries"]}
        if set(names) != set(recorded) | {MANIFEST_NAME, NOTICE_NAME}:
            raise ValueError("Archive and manifest paths differ")
        for name, row in recorded.items():
            safe_relative(name)
            if name.endswith("/case.npz") or name == str(CASE_ROOT / "train_pca.npz"):
                raise ValueError("EEG-derived case arrays cannot be installed from the public bundle")
            if not (name in {str(path) for path in ROOT_NOTICES} or
                    name.startswith(str(CASE_ROOT) + "/") or
                    name.startswith(str(RUN_ROOT) + "/")):
                raise ValueError("Archive contains an unexpected root")
            payload = source.read(name)
            if len(payload) != row["bytes"] or sha256_bytes(payload) != row["sha256"]:
                raise ValueError("Resource checksum mismatch: " + name)
            if destination is not None:
                target = destination.joinpath(*PurePosixPath(name).parts)
                if target.exists():
                    if not target.is_file() or sha256_file(target) != row["sha256"]:
                        raise ValueError("Existing file differs from release: " + str(target))
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
    return {"archive": str(archive), "sha256": sha256_file(archive),
            "entry_count": len(recorded), "destination": str(destination) if destination else None,
            "provenance": manifest["provenance"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify", "install"))
    parser.add_argument("--archive", type=Path,
                        default=ROOT / "artifacts/release/odebrain_hms_resources_v1.zip")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root for build/install")
    args = parser.parse_args()
    if args.action == "build":
        report = build(args.root.resolve(), args.archive.resolve())
    else:
        report = verify(args.archive.resolve(), args.root.resolve() if args.action == "install" else None)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
