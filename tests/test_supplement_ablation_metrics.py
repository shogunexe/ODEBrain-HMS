"""Frozen ablation CSVs must retain source identity during offline rescoring."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.run_ablations import load_plan, sha256
from scripts.supplement_ablation_metrics import supplement
from scripts.supplement_classification_metrics import group_weights, weighted_view
from src.data.hms import CLASSES, manifest_hash


def test_three_arm_rescoring_and_prediction_tamper_rejection(tmp_path):
    plan_path, plan = load_plan("configs/ablations.yaml")
    hard = np.array([0, 0, 1])
    targets = np.eye(6)[hard]
    scores = np.array([[.8, .2, 0, 0, 0, 0],
                       [.6, .4, 0, 0, 0, 0],
                       [.7, .3, 0, 0, 0, 0]])
    manifest = pd.DataFrame({
        "window_id": ["w1", "w2", "w3"], "patient_id": [1, 1, 2],
        "eeg_id": [10, 11, 11], "class_id": hard,
        "eeg_label_offset_seconds": [0, 50, 100], "split": ["test"] * 3,
        "audit_reason": ["ok"] * 3,
        **{"target_" + name: targets[:, i] for i, name in enumerate(CLASSES)},
    })
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    predictions = manifest.drop(columns=["eeg_id", "eeg_label_offset_seconds", "split", "audit_reason"])
    for i, name in enumerate(CLASSES):
        predictions["pred_" + name] = scores[:, i]
    window = weighted_view(targets, scores, hard, np.ones(3))["kl_mean"]
    patient = weighted_view(targets, scores, hard, group_weights([1, 1, 2]))["kl_mean"]
    rows = []
    for arm in plan["arms"]:
        pred_path = tmp_path / (arm["name"] + ".csv")
        metric_path = tmp_path / (arm["name"] + ".json")
        predictions.to_csv(pred_path, index=False)
        metric_path.write_text(json.dumps({"split": "test", "split_hash": manifest_hash(manifest),
                                           "n_windows": 3, "kl_window_mean": window,
                                           "kl_patient_equal_mean": patient}))
        rows.append({"name": arm["name"], "initialization": arm["initialization"],
                     "rk4_step_seconds": arm["rk4_step_seconds"], "seed": plan["seed"],
                     "kl_window_mean": window, "kl_patient_equal_mean": patient,
                     "predictions_file": str(pred_path), "predictions_sha256": sha256(pred_path),
                     "metrics_file": str(metric_path), "metrics_sha256": sha256(metric_path)})
    summary_path = tmp_path / "ablation.json"
    summary_path.write_text(json.dumps({"schema_version": "ode_ablation_formal_v1",
                                        "experiment_status": "formal", "split": "test",
                                        "synthetic": False, "smoke_subset": False,
                                        "plan": str(plan_path), "plan_sha256": sha256(plan_path),
                                        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
                                        "split_hash": manifest_hash(manifest),
                                        "n_windows": 3, "n_patients": 2, "rows": rows}))
    output = tmp_path / "weighted.json"
    result = supplement(summary_path, output)
    assert result["n_eeg_records"] == 2
    assert len(result["rows"]) == 3
    assert result["rows"][0]["aggregations"]["eeg_record_equal"]["kl_mean"] != pytest.approx(patient)
    assert output.is_file()
    predictions.loc[0, "pred_seizure"] = .1
    predictions.to_csv(rows[0]["predictions_file"], index=False)
    with pytest.raises(ValueError, match="Frozen prediction or metric file changed"):
        supplement(summary_path, output)
