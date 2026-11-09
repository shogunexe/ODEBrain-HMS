"""Causality and aggregation checks for the formal forecast evaluator."""

import tempfile
import unittest
from pathlib import Path
import json
import hashlib

import numpy as np
import torch

from scripts.evaluate_forecast import (ErrorAccumulator, _comparison_rows, _history_only,
                                       evaluate_forecast, persistence_forecast)
from scripts.prepare_formal_slides import EvidenceError, validate_comparison


class ForecastEvaluationTests(unittest.TestCase):
    def test_history_whitelist_excludes_future_and_checks_cut(self):
        batch = {
            "graph_nodes": torch.arange(10, dtype=torch.float32).reshape(1, 5, 2, 1),
            "graph_adj": torch.zeros(1, 5, 2, 2),
            "graph_mask": torch.ones(1, 5, 2, dtype=torch.bool),
            "observed_eeg": torch.ones(1, 2, 1000),
            "sample_rate": torch.tensor([200]),
            "future_times_s": torch.tensor([1., 3., 5.]),
            "future_graph_nodes": torch.full((1, 3, 2, 1), 999.),
            "future_graph_mask": torch.ones(1, 3, 2, dtype=torch.bool),
            "patient_id": ["patient-1"],
        }
        observed = _history_only(batch, torch.device("cpu"), 5, (1, 3, 5))
        self.assertEqual(set(observed), {"graph_nodes", "graph_adj", "graph_mask",
                                         "eeg", "dynamics_times_s"})
        batch["future_graph_nodes"].fill_(-999)
        unchanged = _history_only(batch, torch.device("cpu"), 5, (1, 3, 5))
        self.assertTrue(torch.equal(observed["graph_nodes"], unchanged["graph_nodes"]))
        batch["observed_eeg"] = torch.ones(1, 2, 1001)
        with self.assertRaises(ValueError):
            _history_only(batch, torch.device("cpu"), 5, (1, 3, 5))

    def test_persistence_uses_last_valid_history_only(self):
        nodes = torch.tensor([[[[1.], [4.]], [[2.], [9.]], [[3.], [10.]]]])
        mask = torch.tensor([[[True, True], [True, False], [False, False]]])
        prediction = persistence_forecast(nodes, mask, (1, 3, 5))
        self.assertEqual(tuple(prediction.shape), (1, 3, 2, 1))
        self.assertTrue(torch.equal(prediction[0, 0, :, 0], torch.tensor([2., 4.])))
        self.assertTrue(torch.equal(prediction[:, 0], prediction[:, 2]))

    def test_mae_uses_same_valid_targets_and_inverse_scale(self):
        acc = ErrorAccumulator((1, 3, 5))
        target = torch.zeros(2, 3, 2, 1)
        future_mask = torch.ones(2, 3, 2, dtype=torch.bool)
        observed_mask = torch.tensor([[[True, False]], [[True, True]]])
        predictions = {"odebrain_inspired_hms": torch.ones_like(target),
                       "graph_gru": torch.ones_like(target) * 2,
                       "persistence": torch.zeros_like(target)}
        acc.add(predictions, target, future_mask, observed_mask,
                ["p1", "p2"], np.array([0.]), np.array([2.]))
        result = acc.result()
        for row in result["odebrain_inspired_hms"]:
            self.assertEqual(row["valid_node_feature_values"], 3)
            self.assertEqual(row["mae"], 2.0)
            self.assertEqual(row["patient_macro_mae"], 2.0)
        self.assertEqual(result["graph_gru"][0]["mae"], 4.0)
        self.assertEqual(result["persistence"][0]["mae"], 0.0)
        rows = _comparison_rows(result)
        self.assertEqual({row["name"] for row in rows},
                         {"odebrain_inspired_hms", "graph_gru", "persistence"})
        self.assertEqual(rows[0]["mae_by_horizon_s"],
                         {"1": 2.0, "3": 2.0, "5": 2.0})

    def test_smoke_cannot_write_formal_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "forecast_comparison.json"
            with self.assertRaises(ValueError):
                evaluate_forecast("missing-ode", "missing-gru", "missing-manifest",
                                  output, smoke=True)
            self.assertFalse(output.exists())

    def test_slide_gate_accepts_formal_rows_and_rejects_smoke(self):
        rows = [{"name": name, "mae_by_horizon_s": {"1": 1.0, "3": 2.0, "5": 3.0}}
                for name in ("odebrain_inspired_hms", "graph_gru", "persistence")]
        payload = {"experiment_status": "formal", "smoke_subset": False,
                   "synthetic": False, "schema_version": "hms_forecast_comparison_v2",
                   "split": "test", "split_hash": "a" * 64,
                   "n_windows": 2, "n_patients": 2, "rows": rows}
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            manifest.write_text("window_id\n1\n2\n", encoding="utf-8")
            cache_dir = Path(directory) / "forecast_cache"
            cache_dir.mkdir()
            eligible_hash = hashlib.sha256(b"1\n2").hexdigest()
            summary = {"eligible": 2, "excluded": 0, "reason_counts": {"ok": 2},
                       "eligible_window_ids_sha256": eligible_hash}
            cache_index = cache_dir / "index.json"
            cache_index.write_text(json.dumps({
                "schema_version": "hms_graph_forecast_cache_v2",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "forecast_eligibility": {"by_split": {"test": summary}},
            }), encoding="utf-8")
            payload["provenance"] = {"graph_cache_dir": str(cache_dir)}
            payload["forecast_eligibility"] = {
                "original_test_windows": 2, "eligible_test_windows": 2,
                "excluded_test_windows": 0, "reason_counts": {"ok": 2},
                "eligible_window_ids_sha256": eligible_hash,
                "cache_index_sha256": hashlib.sha256(cache_index.read_bytes()).hexdigest(),
            }
            path = Path(directory) / "forecast_comparison.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            bundle = {"forecast_comparison": str(path), "manifest": str(manifest)}
            validate_comparison(bundle, "forecast_comparison",
                                "a" * 64, {"test": (2, 2)}, {})
            payload["experiment_status"] = "real_data_smoke"
            payload["smoke_subset"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(EvidenceError):
                validate_comparison(bundle, "forecast_comparison",
                                    "a" * 64, {"test": (2, 2)}, {})


if __name__ == "__main__":
    unittest.main()
