"""Refit forecast comparison preserves fixed checkpoints and eligible scope."""

import hashlib
import json
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from scripts.evaluate_forecast import (_eligible_development_ids, _read_refit_run,
                                       _training_window_ids, evaluate_forecast)


def _hash_ids(values):
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def test_refit_loader_uses_verified_final_checkpoint_and_forecast_task(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("window_id\nw1\n", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "final.pt"
    checkpoint.write_bytes(b"frozen final checkpoint")
    (run / "window_snapshot.json").write_text(json.dumps({"train_dev": ["w1"]}), encoding="utf-8")
    config = {"model": {"name": "graph_gru", "kwargs": {"forecast_horizons": [1, 3, 5]}},
              "data": {"task": "forecast", "forecast_horizons": [1, 3, 5],
                       "forecast_cut_seconds": 5},
              "training": {"forecast_weight": 1.0}, "smoke_subset": False,
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "refit": {"task": "forecast", "fixed_epochs": 7}}
    checked = {"config": config, "dir": run, "model": object(),
               "checkpoint_path": checkpoint, "mean": [0.] * 5,
               "std": [1.] * 5, "source": {"checkpoint_path": tmp_path / "best.pt"}}
    with mock.patch("scripts.refit_final._check_final_run", return_value=checked) as verify:
        result = _read_refit_run(run, "graph_gru", manifest)
        assert result["checkpoint_path"] == checkpoint
        assert result["checkpoint"]["epoch"] == 6
        assert _training_window_ids(result, "refit") == ["w1"]
        verify.assert_called_once_with(run)
        config["refit"]["task"] = "classification"
        with pytest.raises(ValueError, match="forecast pretraining"):
            _read_refit_run(run, "graph_gru", manifest)


def test_refit_eligibility_requires_all_and_only_eligible_development_windows(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("window_id,split,audit_reason\nt1,train,ok\nt2,train,ok\n"
                        "v1,val,ok\nx1,test,ok\n", encoding="utf-8")
    table = pd.read_csv(manifest)
    cache = tmp_path / "cache"
    cache.mkdir()
    index = {"schema_version": "hms_graph_forecast_cache_v2",
             "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
             "records": [{"window_ids": ["t1", "t2", "v1", "x1"],
                          "eligibility_reason": ["ok", "constant_history", "ok", "ok"]}],
             "forecast_eligibility": {"by_split": {
                 "train": {"eligible": 1, "excluded": 1,
                           "eligible_window_ids_sha256": _hash_ids(["t1"])},
                 "val": {"eligible": 1, "excluded": 0,
                         "eligible_window_ids_sha256": _hash_ids(["v1"])}}}}
    path = cache / "index.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    assert _eligible_development_ids(cache, manifest, table) == {"t1", "v1"}
    index["forecast_eligibility"]["by_split"]["train"]["eligible"] = 2
    path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="eligibility summary"):
        _eligible_development_ids(cache, manifest, table)


def test_refit_forecast_cannot_use_smoke_scope(tmp_path):
    with pytest.raises(ValueError, match="formal test scope"):
        evaluate_forecast("ode", "gru", "manifest", tmp_path / "forecast_comparison_smoke.json",
                          protocol="refit", smoke=True)
