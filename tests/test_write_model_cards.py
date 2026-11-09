"""Model-card packaging must reject provisional provenance."""

import json
from pathlib import Path

import pytest

from scripts.write_model_cards import (CardError, check_metric, parse_runs,
                                       reject_smoke, same_sha)


@pytest.mark.parametrize("record", [
    {"smoke_subset": True},
    {"training": {"experiment_status": "real_data_smoke"}},
    {"source": {"synthetic": True}},
])
def test_smoke_rejected_recursively(record):
    with pytest.raises(CardError):
        reject_smoke(record, "fixture")


def test_metric_rejects_smoke_before_missing_predictions(tmp_path):
    metric = tmp_path / "val_metrics.json"
    metric.write_text(json.dumps({"smoke_subset": True, "split": "val"}), encoding="utf-8")
    with pytest.raises(CardError, match="smoke_subset"):
        check_metric(metric, "tcn", tmp_path, "val", {})


def test_sha_and_run_mapping_fail_closed():
    with pytest.raises(CardError, match="SHA256 mismatch"):
        same_sha("a" * 64, "b" * 64, "checkpoint")
    with pytest.raises(CardError, match="repeated"):
        parse_runs(["tcn=artifacts/runs/a", "tcn=artifacts/runs/b"])
