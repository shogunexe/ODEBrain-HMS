#!/usr/bin/env python3
"""Audit weighted test classification metrics from frozen prediction files.

All three views score the same annotated test windows. Patient and EEG-record
views give each window 1 / group_size weight; they do not vote or predict a
single class for the group. This script never fits or selects a model.
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
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.data.hms import CLASSES, manifest_hash
from src.evaluation.metrics import normalize_probabilities, sample_kl

DEEP_MODELS = ("tcn", "spectrogram_cnn", "fusion", "odebrain_inspired_hms", "graph_gru")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_hash(path):
    path = Path(path)
    if path.is_file():
        return {path.name: sha256(path)}
    parts = sorted(path.glob("*.parquet"))
    if not parts or not (path / "_SUCCESS").is_file():
        raise ValueError("Incomplete prediction directory: %s" % path)
    return {part.name: sha256(part) for part in parts}


def checked_arrays(predictions, manifest_test, kind):
    """Join on window_id and validate test coverage, labels and probabilities."""
    labels = ["target_" + name for name in CLASSES]
    needed = {"window_id", "patient_id", "class_id", *labels}
    if not needed.issubset(predictions.columns):
        raise ValueError("Predictions lack identity, hard labels or soft targets")
    expected = manifest_test.set_index(manifest_test.window_id.astype(str), drop=False)
    ids = predictions.window_id.astype(str)
    if ids.isna().any() or ids.duplicated().any() or len(ids) != len(expected) or set(ids) != set(expected.index):
        raise ValueError("Predictions do not cover exactly the audited test windows")
    truth = expected.loc[ids.to_list()]
    for field in ("patient_id", "class_id"):
        if predictions[field].isna().any() or not np.array_equal(
                predictions[field].astype(str).to_numpy(), truth[field].astype(str).to_numpy()):
            raise ValueError("Prediction %s differs from audited manifest" % field)
    if kind == "spark":
        if not {"eeg_id", "split", "probabilities"}.issubset(predictions.columns):
            raise ValueError("Spark predictions lack EEG identity, split or probabilities")
        if not predictions.split.eq("test").all() or not np.array_equal(
                predictions.eeg_id.astype(str).to_numpy(), truth.eeg_id.astype(str).to_numpy()):
            raise ValueError("Spark EEG identity or split differs from audited manifest")
        scores = np.asarray(predictions.probabilities.tolist(), dtype=np.float64)
    else:
        score_names = ["pred_" + name for name in CLASSES]
        if not set(score_names).issubset(predictions.columns):
            raise ValueError("Deep predictions lack six probabilities")
        scores = predictions[score_names].to_numpy(dtype=np.float64)
    targets = predictions[labels].to_numpy(dtype=np.float64)
    if not np.allclose(targets, truth[labels].to_numpy(dtype=np.float64), atol=1e-6, rtol=0):
        raise ValueError("Prediction soft targets differ from audited manifest")
    if scores.shape != targets.shape or not np.isfinite(scores).all() or (scores < 0).any() or not np.allclose(
            scores.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Prediction probabilities are invalid")
    if kind == "spark" and not np.array_equal(predictions.prediction.astype(int), scores.argmax(axis=1)):
        raise ValueError("Spark prediction column differs from probability argmax")
    return (targets, normalize_probabilities(scores), truth.class_id.to_numpy(dtype=int),
            truth.patient_id.astype(str).to_numpy(), truth.eeg_id.astype(str).to_numpy())


def group_weights(groups):
    groups = np.asarray(groups).astype(str)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def weighted_view(targets, scores, hard, weights):
    """Weighted *window* classification metrics; matrix sum equals total weight."""
    hard = np.asarray(hard, dtype=int)
    scores = normalize_probabilities(scores)
    weights = np.asarray(weights, dtype=np.float64)
    if (len(hard) == 0 or len(weights) != len(hard) or len(targets) != len(hard)
            or not np.isfinite(weights).all() or (weights <= 0).any()
            or (hard < 0).any() or (hard >= len(CLASSES)).any()):
        raise ValueError("Invalid weighted classification inputs")
    pred = scores.argmax(axis=1)
    matrix = confusion_matrix(hard, pred, labels=np.arange(len(CLASSES)), sample_weight=weights)
    total = float(weights.sum())
    rows = {}
    for i, name in enumerate(CLASSES):
        tp = float(matrix[i, i])
        support = float(matrix[i, :].sum())
        predicted_support = float(matrix[:, i].sum())
        fp = predicted_support - tp
        negatives = total - support
        precision = tp / predicted_support if predicted_support else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        binary = (hard == i).astype(int)
        auc = (float(roc_auc_score(binary, scores[:, i], sample_weight=weights))
               if support > 0 and negatives > 0 else None)
        rows[name] = {"support_weight": support, "predicted_support_weight": predicted_support,
                      "precision": precision, "recall": recall, "f1": f1, "ovr_auroc": auc,
                      "false_positive_weight": fp,
                      "false_positive_rate": fp / negatives if negatives else None}
    kl = sample_kl(targets, scores)
    return {"total_window_weight": total, "kl_mean": float(np.average(kl, weights=weights)),
            "macro_f1": float(np.mean([rows[name]["f1"] for name in CLASSES])),
            "confusion_matrix": matrix.tolist(), "per_class": rows}


def validate_kl(views, reported):
    names = {"window_equal": "kl_window_mean", "patient_equal": "kl_patient_equal_mean",
             "eeg_record_equal": "kl_eeg_record_equal_mean"}
    for view, key in names.items():
        if key not in reported or not math.isclose(views[view]["kl_mean"], float(reported[key]),
                                                 rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("Recomputed %s differs from formal comparison" % key)


def supplement(manifest_path, deep_path, spark_path, spark_record_path, output_path,
               deep_results_dir=None):
    manifest_path, deep_path, spark_path, spark_record_path, output_path = map(
        lambda path: Path(path).resolve(),
        (manifest_path, deep_path, spark_path, spark_record_path, output_path))
    manifest = pd.read_csv(manifest_path)
    test = manifest.loc[(manifest.split == "test") & (manifest.audit_reason == "ok")].copy()
    if test.empty or test.window_id.isna().any() or test.window_id.astype(str).duplicated().any():
        raise ValueError("Audited test manifest is empty or has duplicate windows")
    deep = json.loads(deep_path.read_text(encoding="utf-8"))
    spark = json.loads(spark_path.read_text(encoding="utf-8"))
    spark_record = json.loads(spark_record_path.read_text(encoding="utf-8"))
    manifest_sha = sha256(manifest_path)
    split_hash = manifest_hash(manifest)
    if (deep.get("experiment_status") != "formal" or deep.get("split") != "test"
            or deep.get("training_protocol") != "train_val_fixed_epoch_refit"
            or deep.get("manifest_sha256") != manifest_sha or deep.get("split_hash") != split_hash
            or spark.get("experiment_status") != "formal"
            or spark_record.get("experiment_status") != "formal"
            or spark_record.get("manifest_sha256") != manifest_sha
            or spark_record.get("split_hash") != split_hash):
        raise ValueError("Formal source metadata disagree")
    result = {"schema_version": "hms_classification_weighted_metrics_v1",
              "experiment_status": "formal", "split": "test", "test_used_for_selection": False,
              "interpretation": "Each result scores the same test windows. Group-equal views weight each window by inverse windows in its patient or EEG recording; no group-level voting or classification is performed.",
              "hard_label_source": "expert_consensus_class_id",
              "kl_target_source": "expert_vote_distribution",
              "class_order": list(CLASSES), "manifest": str(manifest_path),
              "manifest_sha256": manifest_sha, "split_hash": split_hash,
              "source_sha256": {"deep_comparison": sha256(deep_path), "spark_metrics": sha256(spark_path),
                                "spark_record_metrics": sha256(spark_record_path)},
              "n_windows": len(test), "n_patients": int(test.patient_id.nunique()),
              "n_eeg_records": int(test.eeg_id.nunique()), "models": {}}
    deep_rows = {row["name"]: row for row in deep.get("rows", [])}
    if set(deep_rows) != set(DEEP_MODELS):
        raise ValueError("Final deep comparison lacks exactly five models")
    deep_results_dir = (Path(deep_results_dir).resolve() if deep_results_dir is not None
                        else deep_path.parent)
    for kind, names in (("deep", DEEP_MODELS), ("spark", ("lr", "rf"))):
        for name in names:
            path = (deep_results_dir / Path(deep_rows[name]["run_dir"]).name / "test_predictions.csv" if kind == "deep"
                    else spark_path.parent / (name + "_test_predictions"))
            predictions = pd.read_csv(path) if kind == "deep" else pd.read_parquet(path)
            targets, scores, hard, patients, records = checked_arrays(predictions, test, kind)
            if len(hard) != result["n_windows"] or len(np.unique(patients)) != result["n_patients"] or len(
                    np.unique(records)) != result["n_eeg_records"]:
                raise ValueError("Prediction group coverage differs from audited manifest")
            views = {"window_equal": weighted_view(targets, scores, hard, np.ones(len(hard))),
                     "patient_equal": weighted_view(targets, scores, hard, group_weights(patients)),
                     "eeg_record_equal": weighted_view(targets, scores, hard, group_weights(records))}
            if kind == "deep":
                reported = deep_rows[name]
                if sha256(path) != reported.get("test_predictions_sha256"):
                    raise ValueError("Deep prediction file differs from final comparison")
            else:
                reported = dict(spark["models"][name]["splits"]["test"])
                reported.update(spark_record["models"][name])
                if prediction_hash(path) != spark_record["models"][name]["prediction_parts_sha256"]:
                    raise ValueError("Spark prediction files differ from record KL audit")
            validate_kl(views, reported)
            result["models"][name] = {"model_family": kind,
                                      "training_protocol": ("train_val_fixed_epoch_refit" if kind == "deep"
                                                            else "spark_train_only"),
                                      "predictions_path": str(path), "prediction_sha256": prediction_hash(path),
                                      "aggregations": views}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, output_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/curated/split_manifest.csv")
    parser.add_argument("--deep", default="results/refit/final_comparison.json")
    parser.add_argument("--spark", default="results/spark_baselines/metrics.json")
    parser.add_argument("--spark-record", default="results/spark_baselines/eeg_record_equal_kl.json")
    parser.add_argument("--output", default="results/classification_weighted_metrics.json")
    parser.add_argument("--deep-results-dir", help="Directory of refit test prediction folders")
    args = parser.parse_args()
    result = supplement(args.manifest, args.deep, args.spark, args.spark_record,
                        args.output, args.deep_results_dir)
    print(json.dumps({"output": str(Path(args.output).resolve()), "models": list(result["models"]),
                      "n_windows": result["n_windows"], "n_patients": result["n_patients"],
                      "n_eeg_records": result["n_eeg_records"]}, indent=2))


if __name__ == "__main__":
    main()
