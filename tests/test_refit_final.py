"""Fixed-budget final refit checks with tiny synthetic metadata only."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import torch

from scripts.refit_final import (_check_final_run, _forecast_eligible_ids,
                                 _make_dev_dataset, _plan,
                                 _refit_snapshot, _source_run, summarize_refits,
                                 train_refit)
from src.cli import file_sha256, make_model
from src.training.engine import load_checkpoint, save_checkpoint


def make_source(directory):
    root = Path(directory)
    manifest = root / "split_manifest.csv"
    rows = []
    for name, split, patient in (("w1", "train", "p1"),
                                 ("w2", "val", "p2"), ("w3", "test", "p3")):
        rows.append({"window_id": name, "patient_id": patient, "split": split,
                     "audit_reason": "ok", "eeg_path": str(root / (name + ".parquet")),
                     "eeg_id": name, "eeg_label_offset_seconds": 0.0,
                     "class_id": 0})
    pd.DataFrame(rows).to_csv(manifest, index=False)
    source = root / "source"
    source.mkdir()
    snapshot = {"train": ["w1"], "val": ["w2"]}
    snapshot_bytes = json.dumps(snapshot, sort_keys=True).encode("utf-8")
    (source / "window_snapshot.json").write_bytes(snapshot_bytes)
    config = {"model": {"name": "tcn", "kwargs": {"eeg_channels": 1,
                                               "width": 8, "embedding_dim": 8}},
              "training": {"epochs": 3, "batch_size": 1, "lr": .001,
                           "weight_decay": .01, "classification_weight": 1.0,
                           "forecast_weight": 0.0, "grad_clip": 1.0},
              "data": {"manifest": str(manifest)},
              "smoke_subset": False,
              "manifest_sha256": file_sha256(manifest),
              "window_snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest()}
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    model = make_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    save_checkpoint(source / "best.pt", model, optimizer, epoch=1,
                    best_val_score=.2, config=config, monitor="kl")
    history = [{"epoch": 0, "val": {"kl": .5}},
               {"epoch": 1, "val": {"kl": .2}}]
    (source / "history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history), encoding="utf-8")
    (source / "completed.json").write_text(json.dumps({
        "best_checkpoint_sha256": file_sha256(source / "best.pt"),
        "manifest_sha256": config["manifest_sha256"]}), encoding="utf-8")
    return source


class FinalRefitTests(unittest.TestCase):
    def test_forecast_refit_uses_only_cache_eligible_development_ids(self):
        table = pd.DataFrame([
            {"window_id": "t_ok", "split": "train", "audit_reason": "ok", "patient_id": "p1"},
            {"window_id": "t_bad", "split": "train", "audit_reason": "ok", "patient_id": "p1"},
            {"window_id": "v_ok", "split": "val", "audit_reason": "ok", "patient_id": "p2"},
            {"window_id": "v_bad", "split": "val", "audit_reason": "ok", "patient_id": "p2"},
            {"window_id": "test_ok", "split": "test", "audit_reason": "ok", "patient_id": "p3"},
        ])
        ids = {"train": ["t_ok", "t_bad"], "val": ["v_ok", "v_bad"],
               "test": ["test_ok"]}
        cache = mock.Mock(eligibility={"t_ok": "ok", "t_bad": "history_constant_channel",
                                      "v_ok": "ok", "v_bad": "future_constant_channel",
                                      "test_ok": "ok"})
        config = {"data": {"graph_cache_dir": "forecast_cache", "task": "forecast"},
                  "manifest_sha256": "x"}
        with mock.patch("scripts.refit_final.GraphCacheReader", return_value=cache):
            eligible = _forecast_eligible_ids(config, Path("manifest.csv"), table, ids)
        self.assertEqual(eligible, {"train": ["t_ok"], "val": ["v_ok"],
                                    "test": ["test_ok"]})
        source = {"manifest": Path("manifest.csv"), "name": "odebrain_inspired_hms",
                  "task": "forecast", "config": config, "table": table, "ids": eligible}
        fake_dataset = mock.Mock(rows=table.loc[table.window_id == "t_ok"].copy())
        with mock.patch("scripts.refit_final.HMSWindowDataset", return_value=fake_dataset):
            result = _make_dev_dataset(source, None)
        self.assertEqual(result.rows.window_id.tolist(), ["t_ok", "v_ok"])
        self.assertEqual(_refit_snapshot(dict(source, split_hash="x"))['train_dev'],
                         ["t_ok", "v_ok"])

    def test_source_best_epoch_freezes_budget_and_dev_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            source_path = make_source(temporary)
            source = _source_run(source_path, require_full=False)
            plan = _plan(source)
            self.assertEqual(plan["selected_epochs"], 2)
            self.assertEqual(plan["train_dev_windows"], 2)
            self.assertEqual(plan["estimated_optimizer_steps"], 4)
            self.assertEqual(_refit_snapshot(source)["train_dev"], ["w1", "w2"])
            dataset = _make_dev_dataset(source, None)
            self.assertEqual(dataset.rows.window_id.tolist(), ["w1", "w2"])
            self.assertNotIn("w3", dataset.rows.window_id.tolist())

    def test_refit_runs_fixed_epochs_without_validation_or_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            source_path = make_source(temporary)
            run_dir = Path(temporary) / "refit"
            metrics = {"loss": .1, "kl": .1, "forecast_mse": None,
                       "examples": 2, "forecast_elements": 0}
            with mock.patch("scripts.refit_final.require_full_data"), \
                    mock.patch("scripts.refit_final.train_epoch", return_value=metrics) as train:
                result = train_refit(source_path, run_dir, device="cpu")
                self.assertEqual(train.call_count, 2)
                checked = _check_final_run(run_dir)
            self.assertEqual(result["fixed_epochs_completed"], 2)
            self.assertEqual(checked["config"]["refit"]["training_split"], "train+val")
            self.assertFalse(checked["config"]["refit"]["test_used_for_selection"])
            model = make_model(checked["config"])
            checkpoint = load_checkpoint(run_dir / "final.pt", model, restore_rng=False)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["monitor"], "source_validation_fixed_budget")
            history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
            self.assertEqual([row["epoch"] for row in history], [0, 1])
            self.assertTrue(all(row["validation_performed"] is False and
                                row["test_performed"] is False for row in history))

    def test_source_validation_mismatch_and_tampered_normalizer_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            source_path = make_source(temporary)
            history = source_path / "history.jsonl"
            text = history.read_text().replace('"kl": 0.2', '"kl": 0.3')
            history.write_text(text)
            with self.assertRaises(ValueError):
                _source_run(source_path, require_full=False)
            history.write_text(text.replace('"kl": 0.3', '"kl": 0.2'))
            run_dir = Path(temporary) / "refit"
            metrics = {"loss": .1, "kl": .1, "forecast_mse": None,
                       "examples": 2, "forecast_elements": 0}
            with mock.patch("scripts.refit_final.require_full_data"), \
                    mock.patch("scripts.refit_final.train_epoch", return_value=metrics):
                train_refit(source_path, run_dir)
                record = run_dir / "normalizer_record.json"
                data = json.loads(record.read_text())
                data["sha256"] = "0" * 64
                record.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    _check_final_run(run_dir)

    def test_summary_requires_five_distinct_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                summarize_refits({"tcn": "missing"}, Path(temporary) / "summary.json")
            self.assertFalse((Path(temporary) / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
