"""Build and verify the public ODEBrain v2 checkpoint bundle.

The bundle contains no HMS EEG, prediction rows, or cached patient windows.
"""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "artifacts/release/odebrain_hms_v2_resources.zip"
FILES = [
    "artifacts/refit/odebrain_github_v2_hms/final.pt",
    "artifacts/refit/odebrain_github_v2_hms/config.json",
    "artifacts/refit/odebrain_github_v2_hms/graph_normalizer.npz",
    "artifacts/refit/odebrain_github_v2_hms/completed.json",
    "artifacts/refit/odebrain_github_v2_hms/validation_record.json",
    "artifacts/refit/odebrain_github_v2_pretrain/final.pt",
    "artifacts/refit/odebrain_github_v2_pretrain/config.json",
    "artifacts/refit/odebrain_github_v2_pretrain/graph_normalizer.npz",
    "artifacts/refit/odebrain_github_v2_pretrain/completed.json",
    "artifacts/refit/odebrain_github_v2_pretrain/validation_record.json",
    "configs/models/odebrain_github_hms.yaml",
    "configs/models/odebrain_github_pretrain.yaml",
    "results/odebrain_github_v2/final_comparison.json",
    "results/odebrain_github_v2/classification_weighted_metrics.json",
    "results/odebrain_github_v2/paired_kl_intervals.json",
    "results/odebrain_github_v2/forecast_comparison.json",
    "results/odebrain_github_v2/six_class_field/six_class_train_pca.npz",
]
NOTICE = """ODEBrain-inspired HMS v2 research resources

This is an independent HMS adaptation, not the official ODEBrain/TUSZ model.
Code, methods, and limitations: https://github.com/shogunexe/ODEBrain-HMS
Original ODEBrain source: https://github.com/HHJIAnmo/ODEBRAIN
HMS source: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification
No raw EEG or Kaggle metadata is included. Do not use for clinical diagnosis.
Project license and third-party notices are in the repository root.
"""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def build(path):
    summary = json.loads((ROOT / "results/odebrain_github_v2/final_comparison.json").read_text())
    ode = next(row for row in summary["rows"] if row["name"] == "odebrain_inspired_hms")
    main_bytes = (ROOT / FILES[0]).read_bytes()
    if sha256(main_bytes) != ode["checkpoint_sha256"]:
        raise ValueError("v2 classifier checksum disagrees with formal results")
    if summary["experiment_status"] != "formal" or summary["split"] != "test":
        raise ValueError("release requires frozen formal test results")

    entries = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
        for name in FILES:
            payload = (ROOT / name).read_bytes()
            z.writestr(name, payload)
            entries.append({"path": name, "bytes": len(payload), "sha256": sha256(payload)})
        z.writestr("PUBLIC_RESOURCE_NOTICE.txt", NOTICE.encode("utf-8"))
        manifest = {
            "schema_version": "odebrain_hms_v2_public_resources_v1",
            "model_architecture": "github_trajectory_v2",
            "classifier_checkpoint_sha256": ode["checkpoint_sha256"],
            "split_hash": summary["split_hash"],
            "files": entries,
        }
        z.writestr("PUBLIC_RESOURCE_MANIFEST.json",
                   json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    verify(path)
    print(json.dumps({"archive": str(path), "bytes": path.stat().st_size,
                      "sha256": sha256(path.read_bytes()), "files": len(entries)},
                     ensure_ascii=False))


def verify(path):
    with ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError("corrupt ZIP")
        manifest = json.loads(z.read("PUBLIC_RESOURCE_MANIFEST.json"))
        expected = {entry["path"] for entry in manifest["files"]}
        if expected != set(FILES):
            raise ValueError("bundle file list differs from the release specification")
        for entry in manifest["files"]:
            payload = z.read(entry["path"])
            if len(payload) != entry["bytes"] or sha256(payload) != entry["sha256"]:
                raise ValueError("bundle content mismatch: " + entry["path"])
        classifier = next(e for e in manifest["files"] if e["path"] == FILES[0])
        if classifier["sha256"] != manifest["classifier_checkpoint_sha256"]:
            raise ValueError("classifier checkpoint does not match manifest")
    print("verified", path, "files", len(expected))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    (build if args.action == "build" else verify)(args.archive)
