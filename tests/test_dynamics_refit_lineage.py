"""Final ODE dynamics must inherit a verified development-set forecast head."""

import copy
from unittest import mock

import pytest
import torch

from src.dynamics.analysis import sha256_file
from src.dynamics.export import _verified_forecast_source, load_trained_model
from src.models.odebrain_hms import ODEBrainInspiredHMS


def _fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "split_manifest.csv"
    manifest.write_text("window_id,split\nw1,train\n", encoding="utf-8")
    digest = sha256_file(manifest)
    model_config = {"name": "odebrain_inspired_hms", "kwargs": {"hidden_dim": 8,
                                                               "raw_dim": 4}}
    pretrain_config = {"schema_version": "hms_final_refit_v1", "smoke_subset": False,
                       "manifest_sha256": digest, "model": model_config,
                       "training": {"forecast_weight": 1.0,
                                    "classification_weight": 0.0},
                       "refit": {"task": "forecast"},
                       "data": {"graph_cache_dir": "forecast_cache"}}
    pretrain_model = ODEBrainInspiredHMS(node_features=5, num_nodes=19,
                                         hidden_dim=8, raw_dim=4)
    init_dir = tmp_path / "pretrain"
    init_dir.mkdir()
    source_path = init_dir / "final.pt"
    torch.save({"model": pretrain_model.state_dict(), "config": pretrain_config},
               source_path)
    final_model = copy.deepcopy(pretrain_model)
    with torch.no_grad():
        final_model.classifier[1].weight.add_(1)
    final_dir = tmp_path / "classifier"
    final_dir.mkdir()
    final_path = final_dir / "final.pt"
    config = {"schema_version": "hms_final_refit_v1", "smoke_subset": False,
              "manifest_sha256": digest, "model": model_config,
              "training": {"classification_weight": 1.0,
                           "forecast_weight": 0.0},
              "refit": {"task": "classification", "init_refit_run": str(init_dir),
                        "init_refit_checkpoint_sha256": sha256_file(source_path)}}
    torch.save({"model": final_model.state_dict(), "config": config}, final_path)
    checked = {"checkpoint_path": final_path, "config": config, "source": object()}
    init = {"checkpoint": source_path, "checkpoint_sha256": sha256_file(source_path),
            "path": init_dir, "config": pretrain_config}
    return final_path, manifest, checked, init


def _verify(final_path, manifest, checked, init):
    model, checkpoint = load_trained_model(final_path)
    with mock.patch("scripts.refit_final._check_final_run", return_value=checked) as check, \
            mock.patch("scripts.refit_final._refit_pretrain", return_value=init) as pretrain:
        result = _verified_forecast_source(final_path, checkpoint["config"],
                                           model, manifest, "cpu", None)
    check.assert_called_once_with(final_path.parent)
    pretrain.assert_called_once_with(str(init["path"]), checked["source"])
    return result


def test_refit_dynamics_accepts_verified_development_forecast_head(tmp_path):
    final_path, manifest, checked, init = _fixture(tmp_path)
    source, lineage, smoke = _verify(final_path, manifest, checked, init)
    assert source["config"] == init["config"]
    assert lineage["forecast_head_source"] == str(init["checkpoint"])
    assert lineage["forecast_head_source_sha256"] == sha256_file(init["checkpoint"])
    assert lineage["training_protocol"] == "train_val_fixed_epoch_refit"
    assert smoke is False


def test_refit_dynamics_rejects_changed_source_hash_and_head(tmp_path):
    final_path, manifest, checked, init = _fixture(tmp_path)
    checked["config"] = copy.deepcopy(checked["config"])
    checked["config"]["refit"]["init_refit_checkpoint_sha256"] = "0" * 64
    torch.save({"model": load_trained_model(final_path)[0].state_dict(),
                "config": checked["config"]}, final_path)
    with pytest.raises(ValueError, match="lineage changed"):
        _verify(final_path, manifest, checked, init)

    final_path, manifest, checked, init = _fixture(tmp_path / "second")
    model, checkpoint = load_trained_model(final_path)
    with torch.no_grad():
        model.forecast_head.bias.add_(0.1)
    torch.save({"model": model.state_dict(), "config": checkpoint["config"]}, final_path)
    with pytest.raises(ValueError, match="Forecast head differs"):
        _verify(final_path, manifest, checked, init)


def test_refit_dynamics_rejects_unverified_source_and_wrong_task(tmp_path):
    final_path, manifest, checked, init = _fixture(tmp_path)
    checked["checkpoint_path"] = final_path.parent / "best.pt"
    with pytest.raises(ValueError, match="verified final ODE classifier"):
        _verify(final_path, manifest, checked, init)

    final_path, manifest, checked, init = _fixture(tmp_path / "third")
    init["config"] = copy.deepcopy(init["config"])
    init["config"]["refit"]["task"] = "classification"
    with pytest.raises(ValueError, match="verified final pretraining refit"):
        _verify(final_path, manifest, checked, init)
