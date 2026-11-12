"""Audit and cluster-resampling checks for frozen final-refit comparisons."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.paired_refit_ci import (MODELS, PRED, TARGET, analyze,
                                     paired_intervals, sha256)
from src.data.hms import manifest_hash
from src.evaluation.metrics import sample_kl


def test_three_estimands_keep_patient_and_record_weights_distinct():
    difference = np.array([0., 0., 2., 4.])
    result = paired_intervals(difference, ["p1", "p1", "p1", "p2"],
                              ["r1", "r1", "r2", "r3"], 1000, 17)
    assert result["patient_equal"]["difference"] == pytest.approx(7 / 3)
    assert result["eeg_record_equal"]["difference"] == pytest.approx(2)
    assert result["window_equal"]["difference"] == pytest.approx(1.5)
    assert result == paired_intervals(difference, ["p1", "p1", "p1", "p2"],
                                      ["r1", "r1", "r2", "r3"], 1000, 17)
    assert result["n_patients"] == 2
    assert result["n_eeg_records"] == 3


def test_record_cannot_span_patients():
    with pytest.raises(ValueError, match="spans patients"):
        paired_intervals([1., 2.], ["p1", "p2"], ["same", "same"], 100)


def _fixture(tmp_path):
    manifest = pd.DataFrame({
        "window_id": ["a", "b", "c", "d"],
        "patient_id": ["p1", "p1", "p1", "p2"],
        "eeg_id": ["r1", "r1", "r2", "r3"],
        "eeg_label_offset_seconds": [0, 10, 20, 0],
        "class_id": [0] * 4, "split": ["test"] * 4,
        "audit_reason": ["ok"] * 4,
        **{col: [int(i == 0)] * 4 for i, col in enumerate(TARGET)}})
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    results_dir = tmp_path / "results"
    rows = []
    for i, name in enumerate(MODELS):
        pred = np.full((4, 6), (0.4 + i * 0.1) / 5)
        pred[:, 0] = 0.6 - i * 0.1
        frame = manifest[["window_id", "patient_id", "class_id", *TARGET]].copy()
        for j, col in enumerate(PRED):
            frame[col] = pred[:, j]
        path = results_dir / name / "test_predictions.csv"
        path.parent.mkdir(parents=True)
        frame.to_csv(path, index=False)
        kl = sample_kl(manifest[TARGET].to_numpy(), pred)
        rows.append({"name": name, "kl_window_mean": float(kl.mean()),
                     "kl_patient_equal_mean": float(kl.mean()),
                     "kl_eeg_record_equal_mean": float(kl.mean()),
                     "test_predictions_sha256": sha256(path),
                     "checkpoint_sha256": str(i) * 64})
    comparison = {"schema_version": "hms_final_refit_comparison_v1",
                  "experiment_status": "formal",
                  "training_protocol": "train_val_fixed_epoch_refit",
                  "split": "test", "test_used_for_selection": False,
                  "split_hash": manifest_hash(manifest),
                  "manifest_sha256": sha256(manifest_path),
                  "n_windows": 4, "n_patients": 2, "n_eeg_records": 3,
                  "rows": rows}
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(comparison))
    return manifest_path, comparison_path, results_dir


def test_analysis_rechecks_csv_and_frozen_comparison(tmp_path):
    args = _fixture(tmp_path)
    result = analyze(*args, 100, 42)
    assert set(result["comparisons"]) == {"ode_minus_" + name for name in MODELS
                                          if name != "odebrain_inspired_hms"}
    assert result["n_windows"] == 4
    assert result["test_used_for_selection"] is False
    assert result["test_used_for_model_or_configuration_selection"] is False
    prediction = args[2] / "graph_gru" / "test_predictions.csv"
    frame = pd.read_csv(prediction)
    frame.loc[0, "patient_id"] = "wrong"
    frame.to_csv(prediction, index=False)
    with pytest.raises(ValueError, match="Prediction SHA differs"):
        analyze(*args, 100, 42)
    comparison = json.loads(args[1].read_text())
    for row in comparison["rows"]:
        if row["name"] == "graph_gru":
            row["test_predictions_sha256"] = sha256(prediction)
    args[1].write_text(json.dumps(comparison))
    with pytest.raises(ValueError, match="patient mismatch"):
        analyze(*args, 100, 42)
