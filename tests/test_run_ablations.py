"""Fail-closed checks for formal ODE ablation aggregation."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from scripts import run_ablations as ab
from src.evaluation.metrics import sample_kl


def test_plan_has_matched_initialization_and_step_contrast():
    path, plan = ab.load_plan("configs/ablations.yaml")
    configs = ab.expected_configs(path, plan)
    assert configs["pretrained_rk4_025"]["model"]["kwargs"]["rk4_step_seconds"] == .25
    assert configs["scratch_rk4_025"]["model"]["kwargs"] == configs["pretrained_rk4_025"]["model"]["kwargs"]
    assert configs["pretrained_rk4_050"]["model"]["kwargs"]["rk4_step_seconds"] == .5
    assert configs["pretrain"]["ablation"]["code_sha256"]["src/models/odebrain_hms.py"]


def test_metric_rejects_partial_or_different_split():
    test = pd.DataFrame({"window_id": ["a", "b"], "patient_id": ["1", "2"]})
    metric = {"smoke_subset": False, "evaluation_scope": "full_split", "split": "test",
              "model": "odebrain_inspired_hms", "split_hash": "a" * 64,
              "n_windows": 2, "n_patients": 2, "metadata_eeg_coverage": 1.0,
              "kl_window_mean": .5, "kl_patient_equal_mean": .5, "macro_f1": .4}
    ab.validate_metric(metric, test, "a" * 64, "example")
    bad = dict(metric, smoke_subset=True)
    with pytest.raises(ab.AblationError, match="smoke or partial"):
        ab.validate_metric(bad, test, "a" * 64, "example")
    bad = dict(metric, n_windows=1)
    with pytest.raises(ab.AblationError, match="counts differ"):
        ab.validate_metric(bad, test, "a" * 64, "example")
    with pytest.raises(ab.AblationError, match="Split hash differs"):
        ab.validate_metric(metric, test, "b" * 64, "example")


def test_predictions_recompute_kl_and_identity(tmp_path):
    test = pd.DataFrame({"window_id": ["a", "b"], "patient_id": ["1", "2"]})
    target = np.eye(6)[[0, 1]]
    probabilities = np.array([[.8, .04, .04, .04, .04, .04],
                              [.04, .8, .04, .04, .04, .04]])
    rows = {"window_id": test.window_id, "patient_id": test.patient_id, "class_id": [0, 1]}
    for i, name in enumerate(ab.CLASSES):
        rows["target_" + name] = target[:, i]
        rows["pred_" + name] = probabilities[:, i]
    path = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    kl = float(sample_kl(target, probabilities).mean())
    metric = {"kl_window_mean": kl, "kl_patient_equal_mean": kl}
    ab.validate_predictions(path, test, metric)
    truth = pd.DataFrame(rows).drop(columns=["pred_" + name for name in ab.CLASSES])
    ab.validate_predictions(path, truth, metric)
    with pytest.raises(ab.AblationError, match="soft labels differ"):
        ab.validate_predictions(path, truth.assign(target_seizure=[.5, 0.]), metric)
    with pytest.raises(ab.AblationError, match="patient identity"):
        ab.validate_predictions(path, test.assign(patient_id=["2", "1"]), metric)
    with pytest.raises(ab.AblationError, match="Metric KL differs"):
        ab.validate_predictions(path, test, dict(metric, kl_window_mean=kl + .1))


def test_checkpoint_rejects_init_lineage_mismatch(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = tmp_path / "arm.yaml"
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("window_id\na\n", encoding="utf-8")
    expected = {"model": {"name": "odebrain_inspired_hms"}, "training": {"epochs": 2},
                "ablation": {"seed": 42}}
    config_path.write_text(yaml.safe_dump(expected), encoding="utf-8")
    snapshot = run_dir / "window_snapshot.json"
    snapshot.write_text('{"train":["a"]}', encoding="utf-8")
    saved = dict(expected, smoke_subset=False, source_config_sha256=ab.sha256(config_path),
                 manifest_sha256=ab.sha256(manifest_path),
                 window_snapshot_sha256=ab.sha256(snapshot))
    (run_dir / "config.json").write_text(json.dumps(saved), encoding="utf-8")
    (run_dir / "history.jsonl").write_text('{"epoch":0}\n', encoding="utf-8")
    (run_dir / "graph_normalizer.npz").write_bytes(b"test")
    torch.save({"config": saved, "epoch": 0, "monitor": "kl"}, run_dir / "best.pt")
    completion = {"source_config_sha256": ab.sha256(config_path),
                  "manifest_sha256": ab.sha256(manifest_path),
                  "window_snapshot_sha256": ab.sha256(snapshot),
                  "best_checkpoint_sha256": ab.sha256(run_dir / "best.pt"),
                  "max_epochs": 2}
    (run_dir / "completed.json").write_text(json.dumps(completion), encoding="utf-8")
    monkeypatch.setattr(ab, "paths", lambda plan, name: (config_path, run_dir))
    ab.validate_checkpoint("scratch", {}, expected, manifest_path)
    with pytest.raises(ab.AblationError, match="Initialization checkpoint differs"):
        ab.validate_checkpoint("scratch", {}, expected, manifest_path, "f" * 64)


def test_completed_run_is_validated_and_skipped(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "completed.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "generated.yaml"
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("window_id\na\n", encoding="utf-8")
    config = {"data": {"manifest": str(manifest_path)}, "training": {"epochs": 2}}
    monkeypatch.setattr(ab, "paths", lambda plan, name: (config_path, run_dir))
    calls = []
    monkeypatch.setattr(ab, "validate_checkpoint", lambda *args: calls.append(args))
    monkeypatch.setattr(ab, "command", lambda argv: pytest.fail("completed run must not retrain"))
    args = SimpleNamespace(python="python", workers=0, device="cpu")
    ab._run_train("pretrain", {"seed": 42}, config, None, args)
    assert len(calls) == 1
    assert calls[0][0] == "pretrain"
