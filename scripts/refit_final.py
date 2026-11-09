"""Fixed-budget train+val refit and one held-out test evaluation.

The source run's validation-selected best epoch fixes the number of refit
epochs. Refit never evaluates a validation or test loader while training.
Graph classifiers that originally used forecast pretraining must initialize
from a separately refitted train+val forecast model.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.cli import file_sha256, input_modalities, make_model, require_full_data
from src.data.dataset import (HMSWindowDataset, RecordingBatchSampler,
                              fit_graph_normalizer, hms_collate)
from src.data.hms import CLASSES, manifest_hash
from src.evaluation.metrics import classification_metrics, patient_bootstrap_kl
from src.training.engine import load_checkpoint, save_checkpoint, train_epoch


GRAPH_MODELS = ("odebrain_inspired_hms", "graph_gru")
CLASSIFIERS = ("tcn", "spectrogram_cnn", "fusion") + GRAPH_MODELS
SCHEMA = "hms_final_refit_v1"


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _hash_json(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _manifest_path(config):
    path = Path(config.get("data", {}).get("manifest", "data/curated/split_manifest.csv"))
    return (path if path.is_absolute() else ROOT / path).resolve()


def _manifest_view(path):
    table = pd.read_csv(path)
    required = {"window_id", "patient_id", "eeg_path", "audit_reason", "split"}
    if not required.issubset(table.columns):
        raise ValueError("Refit requires an audited patient split manifest")
    if table.window_id.isna().any() or table.window_id.astype(str).duplicated().any():
        raise ValueError("Manifest window IDs must be unique and nonempty")
    valid = table.audit_reason == "ok"
    if not table.loc[valid, "split"].isin(("train", "val", "test")).all():
        raise ValueError("Valid windows have an unknown split")
    patients = {split: set(table.loc[table.split == split, "patient_id"].astype(str))
                for split in ("train", "val", "test")}
    if (patients["train"] & patients["val"] or patients["train"] & patients["test"] or
            patients["val"] & patients["test"]):
        raise ValueError("Patient leakage across manifest splits")
    ids = {split: table.loc[(table.split == split) & valid, "window_id"].astype(str).tolist()
           for split in ("train", "val", "test")}
    if any(not values for values in ids.values()):
        raise ValueError("Every formal split needs audited windows")
    return table, ids, patients


def _source_run(source_dir, require_full=True):
    source_dir = Path(source_dir).resolve()
    config_path = source_dir / "config.json"
    checkpoint_path = source_dir / "best.pt"
    snapshot_path = source_dir / "window_snapshot.json"
    history_path = source_dir / "history.jsonl"
    completed_path = source_dir / "completed.json"
    for path in (config_path, checkpoint_path, snapshot_path, history_path, completed_path):
        if not path.is_file():
            raise FileNotFoundError("Source run is incomplete: %s" % path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("smoke_subset") is not False:
        raise ValueError("Final refit requires a completed, nonsmoke source run")
    name = config.get("model", {}).get("name")
    if name not in CLASSIFIERS:
        raise ValueError("Unsupported deep model for final refit: %s" % name)
    manifest = _manifest_path(config)
    if require_full:
        require_full_data(manifest)
    if file_sha256(manifest) != config.get("manifest_sha256"):
        raise ValueError("Source manifest hash differs from current manifest")
    table, ids, patients = _manifest_view(manifest)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if file_sha256(snapshot_path) != config.get("window_snapshot_sha256"):
        raise ValueError("Source training snapshot hash mismatch")
    if (set(map(str, snapshot.get("train", []))) != set(ids["train"]) or
            set(map(str, snapshot.get("val", []))) != set(ids["val"]) or
            len(snapshot.get("train", [])) != len(ids["train"]) or
            len(snapshot.get("val", [])) != len(ids["val"])):
        raise ValueError("Source run did not cover the complete development split")
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    if (completed.get("best_checkpoint_sha256") != file_sha256(checkpoint_path) or
            completed.get("manifest_sha256") != config["manifest_sha256"]):
        raise ValueError("Source run completion record is stale")
    model = make_model(config)
    checkpoint = load_checkpoint(checkpoint_path, model, restore_rng=False)
    if checkpoint.get("config") != config:
        raise ValueError("Source checkpoint config differs from run config")
    best_epoch = checkpoint.get("epoch")
    if type(best_epoch) is not int or best_epoch < 0:
        raise ValueError("Source best epoch is invalid")
    monitor = checkpoint.get("monitor")
    expected_monitor = ("forecast_mse" if
                        float(config.get("training", {}).get("classification_weight", 1)) == 0
                        else "kl")
    if monitor != expected_monitor:
        raise ValueError("Source best checkpoint monitor does not match its objective")
    best_row = None
    for line in history_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("epoch") == best_epoch:
            best_row = row
    if best_row is None or best_row.get("val", {}).get(monitor) is None:
        raise ValueError("Source validation record for selected epoch is missing")
    if not math.isclose(float(best_row["val"][monitor]),
                        float(checkpoint.get("best_val_score", float("nan"))),
                        rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("Source validation record differs from selected checkpoint")
    task = config.get("data", {}).get("task", "classification")
    if task == "forecast":
        if name not in GRAPH_MODELS or float(config["training"].get("forecast_weight", 0)) <= 0:
            raise ValueError("Only graph models may have forecast refit sources")
    elif task != "classification" or float(config["training"].get("classification_weight", 1)) <= 0:
        raise ValueError("Classification refit requires a classification source run")
    return {"dir": source_dir, "config": config, "config_path": config_path,
            "checkpoint_path": checkpoint_path, "checkpoint": checkpoint,
            "snapshot_path": snapshot_path, "best_row": best_row,
            "selected_epochs": best_epoch + 1, "monitor": monitor,
            "name": name, "task": task, "manifest": manifest,
            "table": table, "ids": ids, "patients": patients,
            "split_hash": manifest_hash(table)}


def _same_model(a, b):
    return a["model"] == b["model"]


def _refit_pretrain(init_dir, source):
    if source["task"] != "classification" or source["name"] not in GRAPH_MODELS:
        raise ValueError("Refit pretraining applies only to graph classifiers")
    checked = _check_final_run(init_dir)
    path = checked["dir"]
    config = checked["config"]
    checkpoint_path = checked["checkpoint_path"]
    if (config.get("schema_version") != SCHEMA or
            config.get("refit", {}).get("task") != "forecast" or
            config.get("smoke_subset") is not False or
            config.get("manifest_sha256") != source["config"]["manifest_sha256"] or
            not _same_model(config, source["config"])):
        raise ValueError("Refit pretraining model or manifest differs from classifier")
    original_init = source["config"].get("init_checkpoint")
    if not original_init:
        raise ValueError("Source classifier was not pretrained; refit must start randomly")
    if Path(original_init).resolve().parent != Path(config["refit"]["source_run"]).resolve():
        raise ValueError("Refit pretraining does not reproduce the source pretraining lineage")
    if (source["config"].get("init_checkpoint_sha256") !=
            config["refit"]["source_best_checkpoint_sha256"]):
        raise ValueError("Refit pretraining selected a different source checkpoint")
    return {"path": path, "config": config, "checkpoint": checkpoint_path,
            "normalizer": (checked["mean"], checked["std"]),
            "checkpoint_sha256": file_sha256(checkpoint_path)}


def _selection_record(source):
    return {"source_run": str(source["dir"]),
            "source_best_checkpoint": str(source["checkpoint_path"]),
            "source_best_checkpoint_sha256": file_sha256(source["checkpoint_path"]),
            "source_config_sha256": file_sha256(source["config_path"]),
            "source_snapshot_sha256": file_sha256(source["snapshot_path"]),
            "selection_split": "val", "selection_monitor": source["monitor"],
            "source_best_epoch_zero_indexed": source["selected_epochs"] - 1,
            "fixed_refit_epochs": source["selected_epochs"],
            "source_best_validation_score": float(source["checkpoint"]["best_val_score"]),
            "source_validation_row": source["best_row"],
            "test_used_for_selection": False}


def _plan(source, init=None, measured_seconds_per_epoch=None):
    if source["task"] == "classification" and source["name"] in GRAPH_MODELS:
        original_pretrained = bool(source["config"].get("init_checkpoint"))
        if original_pretrained != bool(init):
            raise ValueError("Graph classifier refit must match its source initialization policy")
    elif init is not None:
        raise ValueError("This source run must refit from random initialization")
    n_dev = len(source["ids"]["train"]) + len(source["ids"]["val"])
    batch_size = int(source["config"]["training"].get("batch_size", 8))
    if batch_size < 1:
        raise ValueError("Source batch size is invalid")
    n_batches = math.ceil(n_dev / batch_size)
    model = make_model(source["config"])
    summary = {"schema_version": SCHEMA, "model": source["name"],
               "task": source["task"], "source_run": str(source["dir"]),
               "split_hash": source["split_hash"],
               "manifest_sha256": source["config"]["manifest_sha256"],
               "train_windows": len(source["ids"]["train"]),
               "val_windows": len(source["ids"]["val"]),
               "train_dev_windows": n_dev,
               "train_dev_patients": len(source["patients"]["train"] | source["patients"]["val"]),
               "batch_size": batch_size,
               "selected_epochs": source["selected_epochs"],
               "optimizer_steps_per_epoch": n_batches,
               "estimated_optimizer_steps": n_batches * source["selected_epochs"],
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "initialization": "refit_forecast_pretraining" if init else "random",
               "refit_pretrain_run": str(init["path"]) if init else None,
               "test_used_for_selection": False}
    if measured_seconds_per_epoch is not None:
        if measured_seconds_per_epoch <= 0:
            raise ValueError("Measured seconds per epoch must be positive")
        summary["projected_hours_from_user_measured_epoch"] = (
            source["selected_epochs"] * measured_seconds_per_epoch / 3600.0)
    return summary


def _refit_snapshot(source):
    ids = sorted(source["ids"]["train"] + source["ids"]["val"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate development windows")
    return {"schema_version": SCHEMA,
            "manifest_sha256": source["config"]["manifest_sha256"],
            "split_hash": source["split_hash"],
            "train_dev": ids,
            "source_train": sorted(source["ids"]["train"]),
            "source_val": sorted(source["ids"]["val"])}


def _build_refit_config(source, init, seed, snapshot, graph_cache_dir=None):
    config = copy.deepcopy(source["config"])
    for key in ("source_config_sha256", "window_snapshot_sha256", "init_checkpoint",
                "init_checkpoint_sha256"):
        config.pop(key, None)
    config["schema_version"] = SCHEMA
    config["smoke_subset"] = False
    config["window_snapshot_sha256"] = _hash_json(snapshot)
    config["training"]["epochs"] = source["selected_epochs"]
    if graph_cache_dir is not None:
        config.setdefault("data", {})["graph_cache_dir"] = str(Path(graph_cache_dir).resolve())
    config["refit"] = {
        "task": source["task"], "training_split": "train+val",
        "source_run": str(source["dir"]),
        "source_best_checkpoint_sha256": file_sha256(source["checkpoint_path"]),
        "source_best_epoch_zero_indexed": source["selected_epochs"] - 1,
        "fixed_epochs": source["selected_epochs"],
        "selection_monitor": source["monitor"],
        "source_best_validation_score": float(source["checkpoint"]["best_val_score"]),
        "initialization": "refit_forecast_pretraining" if init else "random",
        "init_refit_run": str(init["path"]) if init else None,
        "init_refit_checkpoint_sha256": init["checkpoint_sha256"] if init else None,
        "seed": int(seed), "test_used_for_selection": False,
    }
    return config


def _make_dev_dataset(source, normalizer, graph_cache_dir=None):
    config = source["config"]
    include_spec, include_graph = input_modalities(source["name"])
    data_config = config.get("data", {})
    cache = graph_cache_dir if graph_cache_dir is not None else data_config.get("graph_cache_dir")
    dataset = HMSWindowDataset(source["manifest"], "train", task=source["task"],
                               forecast_cut_seconds=int(data_config.get("forecast_cut_seconds", 5)),
                               forecast_horizons=tuple(data_config.get("forecast_horizons", (1, 3, 5))),
                               include_spectrogram=include_spec, include_graph=include_graph,
                               graph_cache_dir=cache,
                               graph_mean=None if normalizer is None else normalizer[0],
                               graph_std=None if normalizer is None else normalizer[1])
    rows = source["table"].loc[(source["table"].audit_reason == "ok") &
                                source["table"].split.isin(("train", "val"))].copy()
    dataset.rows = rows.sort_values("window_id").reset_index(drop=True)
    if set(dataset.rows.window_id.astype(str)) != set(source["ids"]["train"] + source["ids"]["val"]):
        raise ValueError("Refit dataset differs from frozen development windows")
    if (dataset.rows.split == "test").any():
        raise ValueError("Test rows entered refit dataset")
    return dataset


def _normalizer(source, init, run_dir, resume, graph_cache_dir):
    target = run_dir / "graph_normalizer.npz"
    record_path = run_dir / "normalizer_record.json"
    include_graph = source["name"] in GRAPH_MODELS
    if resume:
        if not target.is_file() or not record_path.is_file():
            raise FileNotFoundError("Resume requires saved refit graph normalizer and record")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (record.get("sha256") != file_sha256(target) or
                record.get("source") != ("refit_pretraining" if init else
                                         "train_val_fit" if include_graph else "identity")):
            raise ValueError("Resume graph normalizer has changed")
        with np.load(target, allow_pickle=False) as values:
            mean, std = values["mean"].copy(), values["std"].copy()
        return mean, std
    if not include_graph:
        mean, std = np.zeros(5, np.float32), np.ones(5, np.float32)
    elif init is not None:
        mean, std = init["normalizer"]
    else:
        data = _make_dev_dataset(source, None, graph_cache_dir)
        mean, std = fit_graph_normalizer(data)
    if mean.shape != (5,) or std.shape != (5,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid train+val graph normalizer")
    np.savez(target, mean=mean.astype(np.float32), std=std.astype(np.float32))
    _atomic_json(record_path, {"sha256": file_sha256(target),
                               "source": "refit_pretraining" if init else
                               "train_val_fit" if include_graph else "identity",
                               "training_split": "train+val", "test_used": False})
    return mean, std


def _check_existing_run(run_dir, config, snapshot, selection, resume):
    config_path = run_dir / "config.json"
    snapshot_path = run_dir / "window_snapshot.json"
    selection_path = run_dir / "validation_record.json"
    if resume:
        for path in (config_path, snapshot_path, selection_path, run_dir / "latest.pt",
                     run_dir / "normalizer_record.json"):
            if not path.is_file():
                raise FileNotFoundError("Resume artifact missing: %s" % path)
        if (json.loads(config_path.read_text(encoding="utf-8")) != config or
                json.loads(snapshot_path.read_text(encoding="utf-8")) != snapshot or
                json.loads(selection_path.read_text(encoding="utf-8")) != selection):
            raise ValueError("Resume config, snapshot or validation record has changed")
        if (run_dir / "final.pt").exists():
            raise FileExistsError("Refit is already complete")
    else:
        if any(path.exists() for path in (config_path, snapshot_path, selection_path,
                                          run_dir / "latest.pt", run_dir / "final.pt")):
            raise FileExistsError("Refit run directory is already in use; choose another or --resume")
        _atomic_json(config_path, config)
        _atomic_json(snapshot_path, snapshot)
        _atomic_json(selection_path, selection)


def train_refit(source_dir, run_dir, init_refit_dir=None, seed=42,
                device="cpu", workers=0, resume=False, graph_cache_dir=None):
    """Train exactly source-best-epoch+1 epochs on train+val, without evaluation."""
    if workers < 0:
        raise ValueError("workers must be nonnegative")
    source = _source_run(source_dir)
    init = _refit_pretrain(init_refit_dir, source) if init_refit_dir else None
    plan = _plan(source, init)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = _refit_snapshot(source)
    config = _build_refit_config(source, init, seed, snapshot, graph_cache_dir)
    selection = _selection_record(source)
    _check_existing_run(run_dir, config, snapshot, selection, resume)
    if not resume:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    mean, std = _normalizer(source, init, run_dir, resume, graph_cache_dir)
    dataset = _make_dev_dataset(source, (mean, std), graph_cache_dir)
    batch_size = plan["batch_size"]
    loader = DataLoader(dataset, batch_sampler=RecordingBatchSampler(dataset, batch_size),
                        num_workers=workers, collate_fn=hms_collate)
    model = make_model(config)
    if init is not None and not resume:
        init_checkpoint = load_checkpoint(init["checkpoint"], model,
                                          optimizer=None, restore_rng=False)
        if init_checkpoint.get("config") != init["config"]:
            raise ValueError("Refit initialization checkpoint config has changed")
    model.to(device)
    settings = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings.get("lr", .001)),
                                  weight_decay=float(settings.get("weight_decay", .01)))
    start_epoch = 0
    if resume:
        prior = load_checkpoint(run_dir / "latest.pt", model, optimizer=optimizer,
                                map_location=device, restore_rng=True)
        if prior.get("config") != config or prior.get("monitor") != "source_validation_fixed_budget":
            raise ValueError("Resume checkpoint differs from frozen refit protocol")
        start_epoch = int(prior["epoch"]) + 1
    if start_epoch > source["selected_epochs"]:
        raise ValueError("Latest checkpoint exceeded the fixed refit budget")
    begin = time.perf_counter()
    for epoch in range(start_epoch, source["selected_epochs"]):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        tick = time.perf_counter()
        metrics = train_epoch(model, loader, optimizer, device=device,
                              forecast_weight=float(settings.get("forecast_weight", 0)),
                              classification_weight=float(settings.get("classification_weight", 1)),
                              grad_clip=float(settings.get("grad_clip", 1)))
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        row = {"epoch": epoch, "train": metrics,
               "epoch_seconds": time.perf_counter() - tick,
               "fixed_budget_source_best_epoch": source["selected_epochs"] - 1,
               "validation_performed": False, "test_performed": False}
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, allow_nan=False) + "\n")
        save_checkpoint(run_dir / "latest.pt", model, optimizer, epoch,
                        float(source["checkpoint"]["best_val_score"]), config,
                        monitor="source_validation_fixed_budget")
    final_epoch = source["selected_epochs"] - 1
    save_checkpoint(run_dir / "final.pt", model, optimizer, final_epoch,
                    float(source["checkpoint"]["best_val_score"]), config,
                    monitor="source_validation_fixed_budget")
    completed = {"schema_version": SCHEMA, "experiment_status": "formal_refit",
                 "model": source["name"], "task": source["task"],
                 "source_run": str(source["dir"]),
                 "final_checkpoint_sha256": file_sha256(run_dir / "final.pt"),
                 "manifest_sha256": config["manifest_sha256"],
                 "window_snapshot_sha256": file_sha256(run_dir / "window_snapshot.json"),
                 "validation_record_sha256": file_sha256(run_dir / "validation_record.json"),
                 "normalizer_sha256": file_sha256(run_dir / "graph_normalizer.npz"),
                 "normalizer_record_sha256": file_sha256(run_dir / "normalizer_record.json"),
                 "fixed_epochs_completed": source["selected_epochs"],
                 "elapsed_seconds_this_invocation": time.perf_counter() - begin,
                 "device": str(device), "test_used_for_selection": False}
    _atomic_json(run_dir / "completed.json", completed)
    return {"run_dir": str(run_dir), "final_checkpoint": str(run_dir / "final.pt"),
            "fixed_epochs_completed": source["selected_epochs"],
            "train_dev_windows": len(dataset), "test_used_for_selection": False}


def _check_final_run(run_dir):
    run_dir = Path(run_dir).resolve()
    files = {name: run_dir / name for name in
             ("config.json", "window_snapshot.json", "validation_record.json",
              "graph_normalizer.npz", "normalizer_record.json", "final.pt",
              "completed.json")}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError("Final refit artifact missing: %s" % path)
    config = json.loads(files["config.json"].read_text(encoding="utf-8"))
    completed = json.loads(files["completed.json"].read_text(encoding="utf-8"))
    snapshot = json.loads(files["window_snapshot.json"].read_text(encoding="utf-8"))
    selection = json.loads(files["validation_record.json"].read_text(encoding="utf-8"))
    if (config.get("schema_version") != SCHEMA or config.get("smoke_subset") is not False or
            config.get("refit", {}).get("training_split") != "train+val" or
            config.get("refit", {}).get("test_used_for_selection") is not False):
        raise ValueError("Run is not a formal fixed-budget development refit")
    if (completed.get("schema_version") != SCHEMA or
            completed.get("final_checkpoint_sha256") != file_sha256(files["final.pt"]) or
            completed.get("window_snapshot_sha256") != file_sha256(files["window_snapshot.json"]) or
            completed.get("validation_record_sha256") != file_sha256(files["validation_record.json"]) or
            completed.get("normalizer_sha256") != file_sha256(files["graph_normalizer.npz"]) or
            completed.get("normalizer_record_sha256") != file_sha256(files["normalizer_record.json"])):
        raise ValueError("Refit completion hashes do not match final artifacts")
    normalizer_record = json.loads(files["normalizer_record.json"].read_text(encoding="utf-8"))
    expected_normalizer_source = ("refit_pretraining" if config["refit"].get("init_refit_run")
                                  else "train_val_fit" if config["model"]["name"] in GRAPH_MODELS
                                  else "identity")
    if (normalizer_record.get("sha256") != file_sha256(files["graph_normalizer.npz"]) or
            normalizer_record.get("source") != expected_normalizer_source or
            normalizer_record.get("test_used") is not False):
        raise ValueError("Refit normalizer changed after fitting")
    if config.get("window_snapshot_sha256") != _hash_json(snapshot):
        raise ValueError("Refit window snapshot differs from frozen config")
    source = _source_run(config["refit"]["source_run"])
    if (source["config"]["manifest_sha256"] != config.get("manifest_sha256") or
            not _same_model(source["config"], config) or
            source["selected_epochs"] != config["refit"]["fixed_epochs"] or
            file_sha256(source["checkpoint_path"]) != config["refit"]["source_best_checkpoint_sha256"] or
            selection != _selection_record(source) or
            completed.get("fixed_epochs_completed") != source["selected_epochs"]):
        raise ValueError("Refit no longer matches its validation-selected source")
    expected_dev = set(source["ids"]["train"] + source["ids"]["val"])
    if set(map(str, snapshot.get("train_dev", []))) != expected_dev or len(snapshot["train_dev"]) != len(expected_dev):
        raise ValueError("Refit snapshot does not cover all development windows")
    with np.load(files["graph_normalizer.npz"], allow_pickle=False) as values:
        mean, std = values["mean"].copy(), values["std"].copy()
    if mean.shape != (5,) or std.shape != (5,) or (std <= 0).any() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Refit graph normalization is invalid")
    model = make_model(config)
    checkpoint = load_checkpoint(files["final.pt"], model, restore_rng=False)
    if (checkpoint.get("config") != config or checkpoint.get("epoch") != source["selected_epochs"] - 1 or
            checkpoint.get("monitor") != "source_validation_fixed_budget"):
        raise ValueError("Final checkpoint differs from the fixed refit budget")
    if config["model"]["name"] in GRAPH_MODELS:
        init_dir = config["refit"].get("init_refit_run")
        original_pretrained = bool(source["config"].get("init_checkpoint"))
        if original_pretrained != bool(init_dir):
            raise ValueError("Graph classifier pretraining lineage changed")
        if init_dir:
            init = _refit_pretrain(init_dir, source)
            if (init["checkpoint_sha256"] != config["refit"].get("init_refit_checkpoint_sha256") or
                    not np.array_equal(mean, init["normalizer"][0]) or
                    not np.array_equal(std, init["normalizer"][1])):
                raise ValueError("Refit classifier preprocessing differs from pretraining")
    return {"dir": run_dir, "config": config, "completed": completed,
            "source": source, "model": model, "checkpoint_path": files["final.pt"],
            "mean": mean, "std": std}


def evaluate_refit(run_dir, output_dir=None, batch_size=8, workers=0, device="cpu"):
    """Evaluate the frozen final checkpoint on held-out test exactly once per call."""
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and workers nonnegative")
    run = _check_final_run(run_dir)
    source = run["source"]
    config = run["config"]
    if source["task"] != "classification":
        raise ValueError("Forecast pretraining refit has no classification test evaluation")
    include_spec, include_graph = input_modalities(source["name"])
    data_config = config.get("data", {})
    dataset = HMSWindowDataset(source["manifest"], "test", task="classification",
                               include_spectrogram=include_spec, include_graph=include_graph,
                               graph_cache_dir=data_config.get("graph_cache_dir") if include_graph else None,
                               graph_mean=run["mean"], graph_std=run["std"])
    expected = set(source["ids"]["test"])
    if set(dataset.rows.window_id.astype(str)) != expected or len(dataset) != len(expected):
        raise ValueError("Held-out test dataset differs from the frozen split")
    if set(dataset.rows.patient_id.astype(str)) & (
            source["patients"]["train"] | source["patients"]["val"]):
        raise ValueError("Test patients overlap refit development patients")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, collate_fn=hms_collate)
    model = run["model"].to(device).eval()
    targets, probabilities, hard_labels, patients, windows = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            model_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                           for key, value in batch.items()}
            output = model(model_batch)
            logits = output[0] if isinstance(output, tuple) else output
            targets.append(batch["target"].numpy())
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
            hard_labels.extend(batch["class_id"].tolist())
            patients.extend(str(x) for x in batch["patient_id"])
            windows.extend(str(x) for x in batch["window_id"])
    if not probabilities or len(windows) != len(expected) or set(windows) != expected:
        raise ValueError("Final test evaluation did not cover every held-out window")
    target = np.concatenate(targets)
    probs = np.concatenate(probabilities)
    metrics = classification_metrics(target, probs, hard_labels, patients)
    metrics["kl_patient_bootstrap"] = patient_bootstrap_kl(target, probs, patients)
    metrics.update({"experiment_status": "formal", "smoke_subset": False,
                    "synthetic": False, "split": "test",
                    "training_protocol": "train_val_fixed_epoch_refit",
                    "split_hash": source["split_hash"],
                    "manifest_sha256": config["manifest_sha256"],
                    "model": source["name"], "run_dir": str(run["dir"]),
                    "checkpoint_sha256": file_sha256(run["checkpoint_path"]),
                    "source_validation_run": str(source["dir"]),
                    "fixed_refit_epochs": source["selected_epochs"],
                    "test_used_for_selection": False})
    output_dir = (Path(output_dir) if output_dir is not None else
                  ROOT / "results" / "refit" / run["dir"].name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({"window_id": windows, "patient_id": patients,
                                "class_id": hard_labels})
    for index, name in enumerate(CLASSES):
        predictions["target_" + name] = target[:, index]
        predictions["pred_" + name] = probs[:, index]
    temp_csv = output_dir / "test_predictions.csv.tmp"
    predictions.to_csv(temp_csv, index=False)
    os.replace(temp_csv, output_dir / "test_predictions.csv")
    _atomic_json(output_dir / "test_metrics.json", metrics)
    return {"metrics": str(output_dir / "test_metrics.json"),
            "predictions": str(output_dir / "test_predictions.csv"),
            "n_windows": metrics["n_windows"], "n_patients": metrics["n_patients"]}


def summarize_refits(run_dirs, output_path, results_root=None):
    """Verify five paired final-test runs and write one refit-only comparison."""
    if set(run_dirs) != set(CLASSIFIERS) or len(set(map(str, run_dirs.values()))) != len(CLASSIFIERS):
        raise ValueError("Summarize requires exactly five distinct classification refit runs")
    output_path = Path(output_path).resolve()
    results_root = (Path(results_root) if results_root is not None else
                    ROOT / "results" / "refit").resolve()
    rows = []
    shared = None
    for name in CLASSIFIERS:
        run = _check_final_run(run_dirs[name])
        source = run["source"]
        if source["name"] != name or source["task"] != "classification":
            raise ValueError("Refit run name or task differs from summary entry: %s" % name)
        metrics_path = results_root / run["dir"].name / "test_metrics.json"
        predictions_path = results_root / run["dir"].name / "test_predictions.csv"
        if not metrics_path.is_file() or not predictions_path.is_file():
            raise FileNotFoundError("Run lacks final test evidence: %s" % name)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        expected_ids = set(source["ids"]["test"])
        expected_patients = source["patients"]["test"]
        if (metrics.get("experiment_status") != "formal" or
                metrics.get("smoke_subset") is not False or
                metrics.get("synthetic") is not False or
                metrics.get("training_protocol") != "train_val_fixed_epoch_refit" or
                metrics.get("split") != "test" or metrics.get("model") != name or
                Path(metrics.get("run_dir", "")).resolve() != run["dir"] or
                metrics.get("manifest_sha256") != run["config"]["manifest_sha256"] or
                metrics.get("split_hash") != source["split_hash"] or
                metrics.get("checkpoint_sha256") != file_sha256(run["checkpoint_path"]) or
                metrics.get("fixed_refit_epochs") != source["selected_epochs"] or
                metrics.get("n_windows") != len(expected_ids) or
                metrics.get("n_patients") != len(expected_patients)):
            raise ValueError("Final test metrics provenance differs from refit: %s" % name)
        frame = pd.read_csv(predictions_path)
        needed = {"window_id", "patient_id", "class_id"}
        needed.update("target_" + label for label in CLASSES)
        needed.update("pred_" + label for label in CLASSES)
        if not needed.issubset(frame.columns):
            raise ValueError("Test predictions lack required columns: %s" % name)
        ids = frame.window_id.astype(str)
        if ids.duplicated().any() or set(ids) != expected_ids or len(frame) != len(expected_ids):
            raise ValueError("Test predictions do not cover frozen windows: %s" % name)
        manifest_rows = source["table"].set_index(source["table"].window_id.astype(str)).loc[ids]
        if (frame.patient_id.astype(str).tolist() != manifest_rows.patient_id.astype(str).tolist() or
                frame.class_id.astype(int).tolist() != manifest_rows.class_id.astype(int).tolist()):
            raise ValueError("Prediction identities or labels differ from manifest: %s" % name)
        target_columns = ["target_" + label for label in CLASSES]
        pred_columns = ["pred_" + label for label in CLASSES]
        target = frame[target_columns].to_numpy(dtype=np.float64)
        manifest_target = manifest_rows[target_columns].to_numpy(dtype=np.float64)
        probs = frame[pred_columns].to_numpy(dtype=np.float64)
        if (not np.isfinite(target).all() or not np.isfinite(probs).all() or
                not np.allclose(target, manifest_target, atol=1e-6) or
                (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("Test targets or probabilities are invalid: %s" % name)
        recomputed = classification_metrics(target, probs, frame.class_id.to_numpy(),
                                            frame.patient_id.astype(str).to_numpy())
        for key in ("kl_window_mean", "kl_patient_equal_mean", "macro_f1"):
            if not math.isclose(float(metrics.get(key, float("nan"))), recomputed[key],
                                rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError("Test metric differs from predictions (%s): %s" % (name, key))
        if metrics.get("confusion_matrix") != recomputed["confusion_matrix"]:
            raise ValueError("Test confusion matrix differs from predictions: %s" % name)
        identity = (source["split_hash"], run["config"]["manifest_sha256"],
                    frozenset(expected_ids), frozenset(expected_patients))
        if shared is None:
            shared = identity
        elif identity != shared:
            raise ValueError("Five refit models do not share one held-out test split")
        rows.append({"name": name,
                     "kl_window_mean": recomputed["kl_window_mean"],
                     "kl_patient_equal_mean": recomputed["kl_patient_equal_mean"],
                     "macro_f1": recomputed["macro_f1"],
                     "n_windows": len(expected_ids),
                     "n_patients": len(expected_patients),
                     "fixed_refit_epochs": source["selected_epochs"],
                     "source_best_epoch_zero_indexed": source["selected_epochs"] - 1,
                     "checkpoint_sha256": file_sha256(run["checkpoint_path"]),
                     "test_metrics_sha256": file_sha256(metrics_path),
                     "test_predictions_sha256": file_sha256(predictions_path),
                     "run_dir": str(run["dir"])})
    result = {"schema_version": "hms_final_refit_comparison_v1",
              "experiment_status": "formal", "smoke_subset": False,
              "synthetic": False, "training_protocol": "train_val_fixed_epoch_refit",
              "split": "test", "split_hash": shared[0],
              "manifest_sha256": shared[1],
              "n_windows": len(shared[2]), "n_patients": len(shared[3]),
              "rows": rows, "test_used_for_selection": False}
    _atomic_json(output_path, result)
    return {"output": str(output_path), "n_models": len(rows),
            "n_windows": result["n_windows"], "n_patients": result["n_patients"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Inspect a completed initial run without training")
    plan.add_argument("--source-run", required=True)
    plan.add_argument("--init-refit-run")
    plan.add_argument("--measured-seconds-per-epoch", type=float)
    train = commands.add_parser("train", help="Run fixed-budget train+val refit")
    train.add_argument("--source-run", required=True)
    train.add_argument("--run-dir", required=True)
    train.add_argument("--init-refit-run")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cpu")
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--graph-cache-dir")
    train.add_argument("--resume", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Evaluate final refit on held-out test")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--output-dir")
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--workers", type=int, default=0)
    evaluate.add_argument("--device", default="cpu")
    summarize = commands.add_parser("summarize", help="Verify five refit test results together")
    summarize.add_argument("--run", action="append", required=True,
                           help="Model and run directory as name=path; exactly five required")
    summarize.add_argument("--results-root", default="results/refit")
    summarize.add_argument("--output", default="results/refit/final_comparison.json")
    args = parser.parse_args(argv)
    if args.command == "plan":
        source = _source_run(args.source_run)
        init = _refit_pretrain(args.init_refit_run, source) if args.init_refit_run else None
        result = _plan(source, init, args.measured_seconds_per_epoch)
    elif args.command == "train":
        result = train_refit(args.source_run, args.run_dir, args.init_refit_run,
                             args.seed, args.device, args.workers, args.resume,
                             args.graph_cache_dir)
    elif args.command == "evaluate":
        result = evaluate_refit(args.run_dir, args.output_dir, args.batch_size,
                                args.workers, args.device)
    else:
        runs = {}
        for item in args.run:
            name, separator, path = item.partition("=")
            if not separator or not name or not path or name in runs:
                parser.error("Every --run must be a unique name=path")
            runs[name] = path
        result = summarize_refits(runs, args.output, args.results_root)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
