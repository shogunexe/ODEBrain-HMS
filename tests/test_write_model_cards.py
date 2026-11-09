"""Model-card packaging must reject provisional provenance."""

import json
from pathlib import Path

import pytest

from scripts.write_model_cards import (CardError, check_metric, parse_runs,
                                       refit_card_markdown, reject_smoke,
                                       same_sha, verified_refit_comparison)


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


def test_refit_comparison_rejected_if_recomputation_differs(tmp_path, monkeypatch):
    import scripts.refit_final as refit
    from scripts.write_model_cards import MODELS

    comparison = tmp_path / "final_comparison.json"
    comparison.write_text(json.dumps({"split_hash": "a" * 64,
                                      "manifest_sha256": "b" * 64,
                                      "n_windows": 2, "n_patients": 1,
                                      "rows": []}), encoding="utf-8")

    def recompute(_runs, output, _results_root):
        Path(output).write_text(json.dumps({"split_hash": "a" * 64,
                                            "manifest_sha256": "b" * 64,
                                            "n_windows": 2, "n_patients": 1,
                                            "rows": [{"name": "tcn"}]}),
                                encoding="utf-8")

    monkeypatch.setattr(refit, "summarize_refits", recompute)
    runs = {name: str(tmp_path / name) for name in MODELS}
    context = {"split_hash": "a" * 64, "sha256": "b" * 64,
               "counts": {"test": (2, 1)}}
    with pytest.raises(CardError, match="independently recomputed"):
        verified_refit_comparison(runs, str(comparison), str(tmp_path), context)


def test_refit_card_reports_only_final_test_metric():
    card = {"model": "tcn", "weights_sha256": "a" * 64,
            "split_hash": "b" * 64, "manifest_file_sha256": "c" * 64,
            "split_counts": {"train": {"windows": 10, "patients": 2},
                             "val": {"windows": 4, "patients": 1},
                             "test": {"windows": 3, "patients": 1}},
            "sample_rate_hz": 200, "input_channels": ["C3", "C4"],
            "input_modalities": {"raw_eeg": True, "spectrogram": False, "graph": False},
            "graph_normalizer_sha256": "d" * 64,
            "fixed_refit_epochs": 8,
            "selection": {"selection_monitor": "kl"},
            "source_checkpoint_sha256": "e" * 64, "refit_seed": 42,
            "config": "artifacts/refit/tcn/config.json", "config_sha256": "f" * 64,
            "training": {"epochs": 8}, "pretraining": None,
            "comparison": "results/refit/final_comparison.json",
            "comparison_sha256": "0" * 64,
            "source_run": "artifacts/runs/tcn",
            "checkpoint": "artifacts/refit/tcn/final.pt",
            "evaluation": {"test": {"n_windows": 3, "n_patients": 1,
                                    "kl_window_mean": 1.0,
                                    "kl_patient_equal_mean": 1.1,
                                    "macro_f1": .2}}}
    rendered = refit_card_markdown(card)
    assert "| test | 3 | 1 | 1.000000 | 1.100000 | 0.200000 |" in rendered
    assert "| val |" not in rendered
    assert "训练+验证" in rendered
