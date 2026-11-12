#!/usr/bin/env python3
"""Recompute EEG-record-equal test KL from frozen Spark prediction Parquet.

This script reads saved predictions and audited labels. It does not fit models,
choose checkpoints, or change the original Spark metrics file.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.hms import CLASSES, manifest_hash
from src.evaluation.metrics import sample_kl


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calculate(predictions, manifest_test, reported):
    """Verify exact test identity and return three descriptive KL aggregations."""
    labels = ["target_" + name for name in CLASSES]
    required = {"window_id", "eeg_id", "patient_id", "split", "probabilities", *labels}
    if not required.issubset(predictions.columns):
        raise ValueError("Spark predictions lack identities, probabilities or soft targets")
    if not {"window_id", "eeg_id", "patient_id", "split", *labels}.issubset(manifest_test.columns):
        raise ValueError("Audited manifest lacks test identity or soft targets")
    if (predictions.window_id.isna().any() or predictions.window_id.astype(str).duplicated().any()
            or predictions.eeg_id.isna().any() or predictions.patient_id.isna().any()
            or not predictions.split.eq("test").all()):
        raise ValueError("Spark prediction test identities are missing, repeated or mislabeled")
    if manifest_test.window_id.astype(str).duplicated().any() or not manifest_test.split.eq("test").all():
        raise ValueError("Audited test manifest has duplicate or non-test windows")
    expected = manifest_test.set_index(manifest_test.window_id.astype(str), drop=False)
    ids = predictions.window_id.astype(str).tolist()
    if len(ids) != len(expected) or set(ids) != set(expected.index):
        raise ValueError("Spark predictions do not cover exactly the audited test windows")
    truth = expected.loc[ids]
    for key in ("eeg_id", "patient_id"):
        if not np.array_equal(predictions[key].astype(str).to_numpy(), truth[key].astype(str).to_numpy()):
            raise ValueError("Spark prediction %s differs from audited manifest" % key)
    target = predictions[labels].to_numpy(dtype=np.float64)
    if not np.allclose(target, truth[labels].to_numpy(dtype=np.float64), atol=1e-6, rtol=0):
        raise ValueError("Spark prediction soft targets differ from audited manifest")
    probs = np.asarray(predictions.probabilities.tolist(), dtype=np.float64)
    if (probs.shape != target.shape or not np.isfinite(probs).all() or (probs < 0).any()
            or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("Spark prediction probabilities are invalid")
    kl = sample_kl(target, probs)
    frame = pd.DataFrame({"eeg_id": predictions.eeg_id.astype(str).to_numpy(),
                          "patient_id": predictions.patient_id.astype(str).to_numpy(),
                          "kl": kl})
    result = {"n_windows": len(frame),
              "n_patients": int(frame.patient_id.nunique()),
              "n_eeg_records": int(frame.eeg_id.nunique()),
              "kl_window_mean": float(kl.mean()),
              "kl_patient_equal_mean": float(frame.groupby("patient_id").kl.mean().mean()),
              "kl_eeg_record_equal_mean": float(frame.groupby("eeg_id").kl.mean().mean())}
    for key, previous in (("n_windows", reported.get("n")),
                          ("kl_window_mean", reported.get("kl_window_mean")),
                          ("kl_patient_equal_mean", reported.get("kl_patient_equal_mean"))):
        if previous is None or not math.isclose(float(result[key]), float(previous),
                                                 rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("Recomputed %s differs from original Spark metric" % key)
    return result


def supplement(manifest_path, metrics_path, output_path):
    manifest_path, metrics_path, output_path = map(lambda p: Path(p).resolve(),
                                                   (manifest_path, metrics_path, output_path))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("experiment_status") != "formal" or metrics.get("target_source") != "expert_votes":
        raise ValueError("Supplement requires completed formal Spark soft-label predictions")
    manifest = pd.read_csv(manifest_path)
    selected = manifest.loc[(manifest.split == "test") & (manifest.audit_reason == "ok")].copy()
    if selected.empty or selected.patient_id.isna().any() or selected.eeg_id.isna().any():
        raise ValueError("Audited test manifest lacks valid patient or EEG records")
    result = {"schema_version": "hms_spark_eeg_record_equal_kl_v1",
              "experiment_status": "formal", "split": "test",
              "aggregation_unit": "eeg_id",
              "definition": "mean window KL within each EEG recording, then unweighted mean across EEG recordings",
              "test_used_for_selection": False,
              "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
              "split_hash": manifest_hash(manifest),
              "original_metrics": str(metrics_path),
              "original_metrics_sha256": sha256(metrics_path),
              "models": {}}
    for name in ("lr", "rf"):
        reported = metrics.get("models", {}).get(name, {}).get("splits", {}).get("test")
        if not isinstance(reported, dict):
            raise ValueError("Original Spark metrics lack formal %s test result" % name)
        directory = metrics_path.parent / (name + "_test_predictions")
        parts = sorted(directory.glob("*.parquet"))
        if not parts or not (directory / "_SUCCESS").is_file():
            raise ValueError("Spark %s test predictions are incomplete" % name)
        predictions = pd.read_parquet(directory)
        calculated = calculate(predictions, selected, reported)
        result["models"][name] = {
            **calculated, "predictions_path": str(directory),
            "prediction_parts_sha256": {part.name: sha256(part) for part in parts}}
    identities = {(row["n_windows"], row["n_patients"], row["n_eeg_records"])
                  for row in result["models"].values()}
    if len(identities) != 1:
        raise ValueError("Spark LR and RF differ in test coverage")
    result["n_windows"], result["n_patients"], result["n_eeg_records"] = identities.pop()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, output_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/curated/split_manifest.csv")
    parser.add_argument("--metrics", default="results/spark_baselines/metrics.json")
    parser.add_argument("--output", default="results/spark_baselines/eeg_record_equal_kl.json")
    args = parser.parse_args()
    result = supplement(args.manifest, args.metrics, args.output)
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "n_windows": result["n_windows"],
                      "n_eeg_records": result["n_eeg_records"],
                      "kl_eeg_record_equal_mean": {
                          name: row["kl_eeg_record_equal_mean"]
                          for name, row in result["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
