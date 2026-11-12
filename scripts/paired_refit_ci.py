#!/usr/bin/env python3
"""Descriptive paired KL intervals for frozen final refit test predictions.

The five configurations and their train+val refit protocol were fixed before
the held-out test. This script only audits existing predictions and resamples
whole patients; its intervals must never select a model or configuration.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.hms import CLASSES, manifest_hash
from src.evaluation.metrics import sample_kl


MODELS = ("tcn", "spectrogram_cnn", "fusion", "odebrain_inspired_hms", "graph_gru")
ODE = "odebrain_inspired_hms"
TARGET = ["target_" + name for name in CLASSES]
PRED = ["pred_" + name for name in CLASSES]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_intervals(delta, patient_ids, eeg_ids, n_boot=5000, seed=42):
    """Resample patients jointly for patient, EEG-record and window estimands."""
    delta = np.asarray(delta, dtype=np.float64)
    patient_ids = np.asarray(patient_ids, dtype=str)
    eeg_ids = np.asarray(eeg_ids, dtype=str)
    require(type(n_boot) is int and n_boot >= 100, "n_boot must be at least 100")
    require(type(seed) is int and seed >= 0, "seed must be nonnegative")
    require(delta.ndim == 1 and len(delta) > 0 and len(delta) == len(patient_ids)
            and len(delta) == len(eeg_ids) and np.isfinite(delta).all(),
            "Paired KL differences or identities are invalid")
    require((patient_ids != "").all() and (eeg_ids != "").all(),
            "Empty patient or EEG-record identity")
    patients, patient_index = np.unique(patient_ids, return_inverse=True)
    records, record_index = np.unique(eeg_ids, return_inverse=True)
    require(len(patients) >= 2, "At least two test patients are required")
    record_patients = np.full(len(records), -1, dtype=np.int64)
    for p, r in zip(patient_index, record_index):
        if record_patients[r] == -1:
            record_patients[r] = p
        else:
            require(record_patients[r] == p, "An EEG record spans patients")

    window_counts = np.bincount(patient_index, minlength=len(patients))
    window_sums = np.bincount(patient_index, weights=delta, minlength=len(patients))
    patient_means = window_sums / window_counts
    record_counts = np.bincount(record_index, minlength=len(records))
    record_means = np.bincount(record_index, weights=delta,
                               minlength=len(records)) / record_counts
    patient_record_counts = np.bincount(record_patients, minlength=len(patients))
    patient_record_sums = np.bincount(record_patients, weights=record_means,
                                      minlength=len(patients))

    rng = np.random.RandomState(seed)
    draw = rng.randint(len(patients), size=(n_boot, len(patients)))
    patient_samples = patient_means[draw].mean(axis=1)
    record_samples = patient_record_sums[draw].sum(axis=1) / patient_record_counts[draw].sum(axis=1)
    window_samples = window_sums[draw].sum(axis=1) / window_counts[draw].sum(axis=1)

    def estimate(point, samples):
        return {"difference": float(point),
                "ci95": [float(x) for x in np.percentile(samples, (2.5, 97.5))]}

    return {"n_windows": int(len(delta)), "n_patients": int(len(patients)),
            "n_eeg_records": int(len(records)),
            "patient_equal": estimate(patient_means.mean(), patient_samples),
            "eeg_record_equal": estimate(record_means.mean(), record_samples),
            "window_equal": estimate(delta.mean(), window_samples)}


def analyze(manifest_path, comparison_path, results_dir, n_boot=5000, seed=42):
    manifest_path = Path(manifest_path).resolve()
    comparison_path = Path(comparison_path).resolve()
    results_dir = Path(results_dir).resolve()
    manifest = pd.read_csv(manifest_path, dtype={"window_id": str, "patient_id": str,
                                                  "eeg_id": str})
    required = {"window_id", "patient_id", "eeg_id", "split", "audit_reason",
                "class_id", *TARGET}
    require(required.issubset(manifest), "Manifest lacks prediction audit columns")
    require(manifest.window_id.notna().all() and manifest.window_id.is_unique,
            "Manifest window IDs are not unique and complete")
    test = manifest.loc[(manifest.split == "test") & (manifest.audit_reason == "ok")]
    require(len(test) > 0 and (test.patient_id.notna() & test.eeg_id.notna()).all(),
            "Audited test set or identities are empty")
    test = test.set_index("window_id", verify_integrity=True).sort_index()
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    require(comparison.get("schema_version") == "hms_final_refit_comparison_v1"
            and comparison.get("experiment_status") == "formal"
            and comparison.get("training_protocol") == "train_val_fixed_epoch_refit"
            and comparison.get("split") == "test"
            and comparison.get("test_used_for_selection") is False,
            "Comparison is not the frozen final refit test table")
    split_hash = manifest_hash(manifest)
    manifest_digest = sha256(manifest_path)
    require(comparison.get("split_hash") == split_hash and
            comparison.get("manifest_sha256") == manifest_digest and
            comparison.get("n_windows") == len(test) and
            comparison.get("n_patients") == test.patient_id.nunique() and
            comparison.get("n_eeg_records") == test.eeg_id.nunique(),
            "Comparison differs from current audited test manifest")
    rows = comparison.get("rows")
    require(isinstance(rows, list) and len(rows) == len(MODELS) and
            {row.get("name") for row in rows if isinstance(row, dict)} == set(MODELS),
            "Final comparison must contain the five fixed models")
    by_name = {row["name"]: row for row in rows}
    actual_target = test[TARGET].to_numpy(dtype=np.float64)
    actual_class = pd.to_numeric(test.class_id, errors="raise").to_numpy(dtype=np.int64)
    ids = test.index.to_numpy(dtype=str)
    patients = test.patient_id.to_numpy(dtype=str)
    eegs = test.eeg_id.to_numpy(dtype=str)
    kl_by_model = {}
    evidence = {}
    for name in MODELS:
        path = results_dir / name / "test_predictions.csv"
        row = by_name[name]
        digest = sha256(path)
        require(row.get("test_predictions_sha256") == digest,
                "Prediction SHA differs from final comparison: " + name)
        frame = pd.read_csv(path, dtype={"window_id": str, "patient_id": str,
                                         "eeg_id": str})
        require({"window_id", "patient_id", "class_id", *TARGET, *PRED}.issubset(frame),
                "Prediction CSV lacks required columns: " + name)
        require(len(frame) == len(test) and frame.window_id.notna().all()
                and frame.window_id.is_unique and set(frame.window_id) == set(ids),
                "Prediction CSV does not cover exact held-out test: " + name)
        frame = frame.set_index("window_id", verify_integrity=True).loc[ids]
        require(frame.patient_id.to_numpy(dtype=str).tolist() == patients.tolist(),
                "Prediction patient mismatch: " + name)
        if "eeg_id" in frame:
            require(frame.eeg_id.notna().all() and
                    frame.eeg_id.to_numpy(dtype=str).tolist() == eegs.tolist(),
                    "Prediction EEG record mismatch: " + name)
        hard = pd.to_numeric(frame.class_id, errors="raise").to_numpy(dtype=np.int64)
        require(np.array_equal(hard, actual_class), "Prediction class mismatch: " + name)
        target = frame[TARGET].to_numpy(dtype=np.float64)
        pred = frame[PRED].to_numpy(dtype=np.float64)
        require(np.isfinite(target).all() and np.allclose(target, actual_target,
                                                         atol=1e-6, rtol=0),
                "Prediction target differs from manifest: " + name)
        require(np.isfinite(pred).all() and (pred >= 0).all() and (pred <= 1).all()
                and np.allclose(pred.sum(axis=1), 1, atol=1e-5, rtol=0),
                "Invalid predicted probabilities: " + name)
        kl = sample_kl(actual_target, pred)
        require(np.isclose(kl.mean(), float(row["kl_window_mean"]), atol=1e-7,
                           rtol=1e-6), "Prediction KL differs from final table: " + name)
        patient_means = pd.Series(kl).groupby(patients, sort=False).mean()
        record_means = pd.Series(kl).groupby(eegs, sort=False).mean()
        require(np.isclose(patient_means.mean(), float(row["kl_patient_equal_mean"]),
                           atol=1e-7, rtol=1e-6) and
                np.isclose(record_means.mean(), float(row["kl_eeg_record_equal_mean"]),
                           atol=1e-7, rtol=1e-6),
                "Prediction group KL differs from final table: " + name)
        kl_by_model[name] = kl
        evidence[name] = {"predictions": str(path), "predictions_sha256": digest,
                          "checkpoint_sha256": row["checkpoint_sha256"]}
    comparisons = {"ode_minus_" + name:
                   paired_intervals(kl_by_model[ODE] - kl_by_model[name],
                                    patients, eegs, n_boot, seed)
                   for name in MODELS if name != ODE}
    return {"schema_version": "hms_paired_refit_kl_v1", "status": "formal",
            "training_protocol": "train_val_fixed_epoch_refit", "split": "test",
            "purpose": "descriptive_post_selection_evaluation_only",
            "test_used_for_selection": False,
            "test_used_for_model_or_configuration_selection": False,
            "difference_direction": "ODE minus baseline; positive means ODE higher KL",
            "method": {"resampling_unit": "patient", "paired": True,
                       "n_boot": n_boot, "seed": seed,
                       "interval": "percentile_95",
                       "patient_equal": "mean of within-patient window-mean KL differences",
                       "eeg_record_equal": "mean of within-eeg-record window-mean KL differences; patient cluster bootstrap",
                       "window_equal": "mean of window KL differences; patient cluster bootstrap"},
            "manifest": str(manifest_path), "manifest_sha256": manifest_digest,
            "split_hash": split_hash, "final_comparison": str(comparison_path),
            "final_comparison_sha256": sha256(comparison_path),
            "n_windows": int(len(test)), "n_patients": int(test.patient_id.nunique()),
            "n_eeg_records": int(test.eeg_id.nunique()),
            "runs": evidence, "comparisons": comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/curated/split_manifest.csv")
    parser.add_argument("--comparison", default="results/refit/final_comparison.json")
    parser.add_argument("--results-dir", default="results/refit")
    parser.add_argument("--output", default="results/refit/paired_kl_intervals.json")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    relative_to_root = lambda value: Path(value) if Path(value).is_absolute() else ROOT / value
    try:
        result = analyze(relative_to_root(args.manifest), relative_to_root(args.comparison),
                         relative_to_root(args.results_dir), args.n_boot, args.seed)
        output = relative_to_root(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(output.suffix + ".tmp")
        temp.write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                   allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temp, output)
        print(output)
    except (OSError, KeyError, TypeError, ValueError, pd.errors.ParserError) as exc:
        parser.exit(2, "Paired refit analysis rejected: {}\n".format(exc))


if __name__ == "__main__":
    main()
