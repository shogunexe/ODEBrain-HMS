"""Compare ODE, GraphGRU and persistence on identical causal HMS graph targets.

Formal invocation writes ``results/forecast_comparison.json`` only when both
checkpoints, the audited full manifest, and the held-out test split pass all
provenance checks. ``--smoke`` evaluates frozen validation windows and writes
``forecast_comparison_smoke.json``; it never produces a formal result.
``--protocol refit`` evaluates the two fixed-epoch train+val forecast refits
from their ``final.pt`` checkpoints and writes a separate comparison.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.cli import file_sha256, make_model, require_full_data
from src.data.dataset import HMSWindowDataset, hms_collate
from src.data.hms import manifest_hash
from src.training.engine import load_checkpoint


HORIZONS = (1, 3, 5)
METHODS = ("odebrain_inspired_hms", "graph_gru", "persistence")
PROTOCOLS = ("initial", "refit")


def _canonical_hash(values):
    return hashlib.sha256(json.dumps(sorted(str(x) for x in values),
                                     ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _read_run(run_dir, expected_name, manifest_path, smoke):
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best.pt"
    normalizer_path = run_dir / "graph_normalizer.npz"
    for path in (config_path, checkpoint_path, normalizer_path):
        if not path.is_file():
            raise FileNotFoundError("Required run artifact missing: %s" % path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model", {}).get("name") != expected_name:
        raise ValueError("Expected %s model in %s" % (expected_name, run_dir))
    if bool(config.get("smoke_subset")) != smoke:
        raise ValueError("Both checkpoints must match the requested formal/smoke scope")
    if config.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Checkpoint manifest hash differs from current split manifest")
    horizons = tuple(config.get("data", {}).get("forecast_horizons", ()))
    cut = int(config.get("data", {}).get("forecast_cut_seconds", -1))
    if horizons != HORIZONS or cut < 1:
        raise ValueError("Both models must use 1/3/5-second horizons and a positive cut")
    if expected_name == "graph_gru" and tuple(config["model"]["kwargs"].get("forecast_horizons", ())) != HORIZONS:
        raise ValueError("GraphGRU output horizons differ from the evaluation protocol")
    model = make_model(config)
    checkpoint = load_checkpoint(checkpoint_path, model, optimizer=None, restore_rng=False)
    if checkpoint.get("config") != config:
        raise ValueError("Checkpoint config differs from saved run config")
    if expected_name == "odebrain_inspired_hms":
        directly_trained = (config.get("data", {}).get("task") == "forecast" and
                            float(config.get("training", {}).get("forecast_weight", 0)) > 0)
        if not directly_trained:
            _verify_pretrain_source(config, manifest_path, smoke)
    elif config.get("data", {}).get("task") != "forecast" or float(config.get("training", {}).get("forecast_weight", 0)) <= 0:
        raise ValueError("GraphGRU checkpoint must be trained for causal graph forecasting")
    snapshot_path = run_dir / "window_snapshot.json"
    snapshot_hash = config.get("window_snapshot_sha256")
    if not snapshot_hash or not snapshot_path.is_file() or file_sha256(snapshot_path) != snapshot_hash:
        raise ValueError("Training window snapshot missing or changed")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    with np.load(normalizer_path, allow_pickle=False) as values:
        mean = np.asarray(values["mean"], dtype=np.float32).copy()
        std = np.asarray(values["std"], dtype=np.float32).copy()
    if mean.shape != (5,) or std.shape != (5,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid train-fitted graph normalizer")
    return {"run_dir": run_dir, "model": model, "config": config,
            "checkpoint": checkpoint, "checkpoint_path": checkpoint_path,
            "config_path": config_path, "normalizer_path": normalizer_path,
            "mean": mean, "std": std, "snapshot": snapshot}


def _read_refit_run(run_dir, expected_name, manifest_path):
    """Load a finished forecast refit without choosing a checkpoint or epoch."""
    from scripts.refit_final import _check_final_run

    checked = _check_final_run(run_dir)
    config = checked["config"]
    refit = config.get("refit", {})
    if (config.get("model", {}).get("name") != expected_name or
            refit.get("task") != "forecast" or
            config.get("data", {}).get("task") != "forecast" or
            float(config.get("training", {}).get("forecast_weight", 0)) <= 0 or
            config.get("smoke_subset") is not False):
        raise ValueError("Refit comparison requires finished formal graph forecast pretraining")
    if config.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Refit forecast manifest differs from evaluation manifest")
    horizons = tuple(config.get("data", {}).get("forecast_horizons", ()))
    cut = int(config.get("data", {}).get("forecast_cut_seconds", -1))
    if horizons != HORIZONS or cut < 1:
        raise ValueError("Refit forecast horizons or observation cut differ from protocol")
    if expected_name == "graph_gru" and tuple(config["model"]["kwargs"].get("forecast_horizons", ())) != HORIZONS:
        raise ValueError("GraphGRU output horizons differ from evaluation protocol")
    run_dir = checked["dir"]
    snapshot_path = run_dir / "window_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    normalizer_path = run_dir / "graph_normalizer.npz"
    return {"run_dir": run_dir, "model": checked["model"], "config": config,
            "checkpoint": {"epoch": refit["fixed_epochs"] - 1},
            "checkpoint_path": checked["checkpoint_path"],
            "config_path": run_dir / "config.json", "normalizer_path": normalizer_path,
            "mean": checked["mean"], "std": checked["std"], "snapshot": snapshot,
            "completed_path": run_dir / "completed.json",
            "selection_path": run_dir / "validation_record.json",
            "snapshot_path": snapshot_path,
            "source_checkpoint_path": checked["source"]["checkpoint_path"]}


def _training_window_ids(run, protocol):
    key = "train_dev" if protocol == "refit" else "train"
    ids = [str(x) for x in run["snapshot"].get(key, [])]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Checkpoint training snapshot is missing or has duplicate windows")
    return ids


def _eligible_development_ids(cache, manifest_path, table):
    index = json.loads((Path(cache) / "index.json").read_text(encoding="utf-8"))
    if (index.get("schema_version") != "hms_graph_forecast_cache_v2" or
            index.get("manifest_sha256") != file_sha256(manifest_path)):
        raise ValueError("Forecast eligibility cache has the wrong manifest or schema")
    eligible = set()
    for record in index.get("records", []):
        ids = record.get("window_ids", [])
        reasons = record.get("eligibility_reason", [])
        if len(ids) != len(reasons):
            raise ValueError("Forecast cache eligibility record is malformed")
        eligible.update(str(window_id) for window_id, reason in zip(ids, reasons)
                        if reason == "ok")
    dev = set()
    for split in ("train", "val"):
        split_ids = set(table.loc[(table.split == split) &
                                  (table.audit_reason == "ok"), "window_id"].astype(str))
        selected = eligible & split_ids
        summary = index.get("forecast_eligibility", {}).get("by_split", {}).get(split, {})
        digest = hashlib.sha256("\n".join(sorted(selected)).encode("utf-8")).hexdigest()
        if (len(selected) != summary.get("eligible") or
                len(split_ids) != summary.get("eligible", -1) + summary.get("excluded", -1) or
                digest != summary.get("eligible_window_ids_sha256")):
            raise ValueError("Forecast cache development eligibility summary differs from manifest")
        dev.update(selected)
    return dev


def _verify_pretrain_source(config, manifest_path, smoke):
    source_path = config.get("init_checkpoint")
    expected_sha = config.get("init_checkpoint_sha256")
    if not source_path or not expected_sha:
        raise ValueError("ODE finetune requires a verified graph-forecast pretrain source")
    source_path = Path(source_path)
    if not source_path.is_file() or file_sha256(source_path) != expected_sha:
        raise ValueError("ODE pretrain checkpoint missing or changed")
    source_config_path = source_path.parent / "config.json"
    if not source_config_path.is_file():
        raise ValueError("ODE pretrain config missing")
    source = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_model = make_model(source)
    source_checkpoint = load_checkpoint(source_path, source_model, optimizer=None,
                                        restore_rng=False)
    if source_checkpoint.get("config") != source:
        raise ValueError("ODE pretrain checkpoint config differs from its run config")
    if (source.get("model", {}).get("name") != "odebrain_inspired_hms" or
            source.get("manifest_sha256") != file_sha256(manifest_path) or
            bool(source.get("smoke_subset")) != smoke or
            source.get("data", {}).get("task") != "forecast" or
            float(source.get("training", {}).get("forecast_weight", 0)) <= 0):
        raise ValueError("ODE source checkpoint is not matching forecast pretraining")


def _check_manifest(manifest_path, split):
    table = pd.read_csv(manifest_path)
    required = {"window_id", "patient_id", "split", "audit_reason", "eeg_id"}
    if not required.issubset(table.columns):
        raise ValueError("Audited patient split manifest is incomplete")
    if table.window_id.astype(str).duplicated().any():
        raise ValueError("Manifest contains duplicate window IDs")
    selected = table.loc[(table.split == split) & (table.audit_reason == "ok")]
    if selected.empty:
        raise ValueError("Evaluation split has no audited windows")
    if selected.eeg_id.isna().any() or selected.eeg_id.astype(str).eq("").any():
        raise ValueError("Evaluation windows lack EEG recording IDs")
    train_patients = set(table.loc[table.split == "train", "patient_id"].astype(str))
    val_patients = set(table.loc[table.split == "val", "patient_id"].astype(str))
    test_patients = set(table.loc[table.split == "test", "patient_id"].astype(str))
    if train_patients & val_patients or train_patients & test_patients or val_patients & test_patients:
        raise ValueError("Patient overlap across train, validation and test")
    return table, selected


def _frozen_smoke_windows(ode_run, gru_run, selected):
    ode_ids = [str(x) for x in ode_run["snapshot"].get("val", [])]
    gru_ids = [str(x) for x in gru_run["snapshot"].get("val", [])]
    if not ode_ids or ode_ids != gru_ids:
        raise ValueError("Smoke runs must share identical frozen validation windows")
    available = set(selected.window_id.astype(str))
    if any(x not in available for x in ode_ids):
        raise ValueError("Frozen smoke window is absent from validation manifest")
    return ode_ids


def _history_only(batch, device, cut, horizons):
    """Construct a strict whitelist; future targets never enter model.forward."""
    nodes = batch["graph_nodes"]
    eeg = batch["observed_eeg"]
    if nodes.ndim != 4 or nodes.shape[1] != cut or eeg.ndim != 3:
        raise ValueError("Forecast input does not match the declared observation cut")
    sample_rate = batch["sample_rate"]
    if isinstance(sample_rate, torch.Tensor):
        if not bool((sample_rate == sample_rate[0]).all()):
            raise ValueError("Batch has inconsistent sample rates")
        sample_rate = int(sample_rate[0])
    if eeg.shape[-1] != cut * int(sample_rate):
        raise ValueError("Observed EEG includes samples outside the forecast cut")
    times = batch["future_times_s"]
    if tuple(float(x) for x in times.tolist()) != tuple(float(x) for x in horizons):
        raise ValueError("Dataset forecast times differ from the protocol")
    allowed = {key: batch[key].to(device) for key in
               ("graph_nodes", "graph_adj", "graph_mask")}
    allowed["eeg"] = eeg.to(device)
    allowed["dynamics_times_s"] = torch.tensor((0,) + horizons,
                                                dtype=torch.float32, device=device)
    return allowed


def persistence_forecast(nodes, mask, horizons):
    """Repeat each node's last valid observed attributes at every horizon."""
    if nodes.ndim != 4 or mask.shape != nodes.shape[:3]:
        raise ValueError("Invalid graph history or node mask")
    b, frames, n, features = nodes.shape
    positions = torch.arange(frames, device=nodes.device).view(1, frames, 1)
    last = torch.where(mask.bool(), positions, torch.full_like(positions, -1)).amax(dim=1)
    gather = last.clamp_min(0).view(b, 1, n, 1).expand(b, 1, n, features)
    last_nodes = nodes.gather(1, gather).squeeze(1)
    last_nodes = last_nodes * (last >= 0).unsqueeze(-1)
    return last_nodes.unsqueeze(1).expand(b, len(horizons), n, features)


class ErrorAccumulator:
    def __init__(self, horizons):
        self.horizons = tuple(horizons)
        self.sums = {name: np.zeros(len(horizons), dtype=np.float64) for name in METHODS}
        self.counts = np.zeros(len(horizons), dtype=np.int64)
        self.patient_sums = {name: defaultdict(lambda: np.zeros(len(horizons), dtype=np.float64))
                             for name in METHODS}
        self.patient_counts = defaultdict(lambda: np.zeros(len(horizons), dtype=np.int64))
        self.eeg_sums = {name: defaultdict(lambda: np.zeros(len(horizons), dtype=np.float64))
                         for name in METHODS}
        self.eeg_counts = defaultdict(lambda: np.zeros(len(horizons), dtype=np.int64))

    def add(self, predictions, target, future_mask, observed_mask,
            patient_ids, eeg_ids, mean, std):
        if target.ndim != 4 or future_mask.shape != target.shape[:3] or observed_mask.ndim != 3:
            raise ValueError("Forecast target and mask shape mismatch")
        if (target.shape[1] != len(self.horizons) or
                target.shape[0] != len(patient_ids) or target.shape[0] != len(eeg_ids)):
            raise ValueError("Forecast horizon, patient or EEG record count mismatch")
        if any(not str(eeg_id) for eeg_id in eeg_ids):
            raise ValueError("Forecast EEG record ID is empty")
        valid_nodes = observed_mask.any(dim=1)
        valid = future_mask.bool() & valid_nodes.unsqueeze(1)
        valid = valid.unsqueeze(-1).expand_as(target)
        if not bool(torch.isfinite(target[valid]).all()):
            raise ValueError("Valid future graph attributes are nonfinite")
        for name, prediction in predictions.items():
            if name not in METHODS or prediction.shape != target.shape:
                raise ValueError("Model prediction shape mismatch")
            if not bool(torch.isfinite(prediction[valid]).all()):
                raise ValueError("Model forecast is nonfinite")
        # Inverse the shared train-fitted standardization. Units are log1p band power.
        scale = torch.as_tensor(std, dtype=target.dtype, device=target.device).view(1, 1, 1, -1)
        errors = {name: (prediction - target).abs() * scale for name, prediction in predictions.items()}
        for horizon_index in range(len(self.horizons)):
            mask = valid[:, horizon_index]
            for i, (patient, eeg_id) in enumerate(zip(patient_ids, eeg_ids)):
                count = int(mask[i].sum().item())
                if count == 0:
                    continue
                self.counts[horizon_index] += count
                self.patient_counts[str(patient)][horizon_index] += count
                self.eeg_counts[str(eeg_id)][horizon_index] += count
                for name, error in errors.items():
                    value = float(error[i, horizon_index].masked_select(mask[i]).double().sum().item())
                    self.sums[name][horizon_index] += value
                    self.patient_sums[name][str(patient)][horizon_index] += value
                    self.eeg_sums[name][str(eeg_id)][horizon_index] += value

    def result(self):
        if (self.counts == 0).any():
            raise ValueError("At least one horizon has zero valid target attributes")
        output = {}
        for name in METHODS:
            rows = []
            for index, seconds in enumerate(self.horizons):
                per_patient = [sums[index] / self.patient_counts[patient][index]
                               for patient, sums in self.patient_sums[name].items()
                               if self.patient_counts[patient][index] > 0]
                per_eeg = [sums[index] / self.eeg_counts[eeg_id][index]
                           for eeg_id, sums in self.eeg_sums[name].items()
                           if self.eeg_counts[eeg_id][index] > 0]
                rows.append({"horizon_seconds": seconds,
                             "mae": float(self.sums[name][index] / self.counts[index]),
                             "patient_macro_mae": float(np.mean(per_patient)),
                             "eeg_record_equal_mae": float(np.mean(per_eeg)),
                             "valid_node_feature_values": int(self.counts[index]),
                             "patients_with_valid_target": len(per_patient),
                             "eeg_records_with_valid_target": len(per_eeg)})
            output[name] = rows
        return output


def _comparison_rows(methods):
    """Compact rows consumed by the formal slide evidence gate."""
    return [{"name": name,
             "mae_by_horizon_s": {str(row["horizon_seconds"]): row["mae"]
                                  for row in methods[name]},
             "eeg_record_equal_mae_by_horizon_s": {
                 str(row["horizon_seconds"]): row["eeg_record_equal_mae"]
                 for row in methods[name]}}
            for name in METHODS]


def evaluate_forecast(ode_run_dir, gru_run_dir, manifest_path, output_path,
                      smoke=False, batch_size=8, workers=0, device="cpu",
                      graph_cache_dir=None, protocol="initial"):
    if protocol not in PROTOCOLS or (protocol == "refit" and smoke):
        raise ValueError("Refit forecast comparison requires formal test scope")
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and workers nonnegative")
    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path).resolve()
    if smoke and "smoke" not in output_path.name:
        raise ValueError("Smoke evaluation output filename must include 'smoke'")
    if not smoke and output_path.name != "forecast_comparison.json":
        raise ValueError("Formal evaluation output must be named forecast_comparison.json")
    if not smoke:
        require_full_data(manifest_path)
    split = "val" if smoke else "test"
    table, selected = _check_manifest(manifest_path, split)
    if protocol == "refit":
        ode = _read_refit_run(ode_run_dir, "odebrain_inspired_hms", manifest_path)
        gru = _read_refit_run(gru_run_dir, "graph_gru", manifest_path)
    else:
        ode = _read_run(ode_run_dir, "odebrain_inspired_hms", manifest_path, smoke)
        gru = _read_run(gru_run_dir, "graph_gru", manifest_path, smoke)
    if ode["config"]["data"]["forecast_cut_seconds"] != gru["config"]["data"]["forecast_cut_seconds"]:
        raise ValueError("Models used different forecast observation cuts")
    if not np.array_equal(ode["mean"], gru["mean"]) or not np.array_equal(ode["std"], gru["std"]):
        raise ValueError("Models used different train-fitted graph normalization")
    if set(_training_window_ids(ode, protocol)) != set(_training_window_ids(gru, protocol)):
        raise ValueError("Models used different training window sets")
    if protocol == "refit":
        source_ids = set(table.loc[(table.audit_reason == "ok") &
                                   table.split.isin(("train", "val")), "window_id"].astype(str))
        if not set(_training_window_ids(ode, protocol)).issubset(source_ids):
            raise ValueError("Refit forecast snapshot contains windows outside the development split")
    cut = int(ode["config"]["data"]["forecast_cut_seconds"])
    if smoke:
        chosen = _frozen_smoke_windows(ode, gru, selected)
        selected = selected.set_index(selected.window_id.astype(str), drop=False).loc[chosen].reset_index(drop=True)
    cache = graph_cache_dir
    if cache is None and not smoke:
        cache = ode["config"].get("data", {}).get("graph_cache_dir")
    if not smoke and not cache:
        raise ValueError("Formal forecasting requires the audited eligibility cache")
    if protocol == "refit":
        expected_caches = [run["config"].get("data", {}).get("graph_cache_dir")
                           for run in (ode, gru)]
        if any(not path or Path(path).resolve() != Path(cache).resolve()
               for path in expected_caches):
            raise ValueError("Refit models and evaluation must use the same forecast eligibility cache")
        eligible_dev = _eligible_development_ids(cache, manifest_path, table)
        if set(_training_window_ids(ode, protocol)) != eligible_dev:
            raise ValueError("Refit forecast snapshot differs from all eligible development windows")
    dataset = HMSWindowDataset(manifest_path, split, task="forecast",
                               forecast_cut_seconds=cut, forecast_horizons=HORIZONS,
                               graph_mean=ode["mean"], graph_std=ode["std"],
                               include_spectrogram=False, include_graph=True,
                               graph_cache_dir=cache)
    eligibility = None
    if smoke:
        dataset.rows = selected.reset_index(drop=True)
    else:
        expected = dataset.graph_cache.eligibility_summary["by_split"][split]
        eligible_ids = set(dataset.rows.window_id.astype(str))
        original_ids = set(selected.window_id.astype(str))
        if (not eligible_ids.issubset(original_ids) or
                len(eligible_ids) != expected["eligible"] or
                len(original_ids) != expected["eligible"] + expected["excluded"] or
                hashlib.sha256("\n".join(sorted(eligible_ids)).encode("utf-8")).hexdigest()
                != expected["eligible_window_ids_sha256"]):
            raise ValueError("Forecast eligibility cache does not match audited test split")
        selected = dataset.rows.copy()
        eligibility = {
            "original_test_windows": len(original_ids),
            "eligible_test_windows": expected["eligible"],
            "excluded_test_windows": expected["excluded"],
            "reason_counts": expected["reason_counts"],
            "eligible_window_ids_sha256": expected["eligible_window_ids_sha256"],
            "cache_index_sha256": file_sha256(Path(cache) / "index.json"),
        }
    eeg_by_window = dict(zip(selected.window_id.astype(str), selected.eeg_id.astype(str)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, collate_fn=hms_collate)
    torch_device = torch.device(device)
    ode["model"].to(torch_device).eval()
    gru["model"].to(torch_device).eval()
    errors = ErrorAccumulator(HORIZONS)
    seen_windows = []
    seen_patients = []
    seen_eeg_ids = []
    with torch.no_grad():
        for batch in loader:
            history = _history_only(batch, torch_device, cut, HORIZONS)
            target = batch["future_graph_nodes"].to(torch_device)
            future_mask = batch["future_graph_mask"].to(torch_device)
            predictions = {"persistence": persistence_forecast(history["graph_nodes"],
                                                                  history["graph_mask"], HORIZONS)}
            for name, run in (("odebrain_inspired_hms", ode), ("graph_gru", gru)):
                _, extra = run["model"](history, return_dynamics=True)
                prediction = extra["predicted_graph_nodes"]
                if prediction.shape != target.shape:
                    raise ValueError("%s prediction does not match target horizon shape" % name)
                if name == "graph_gru" and tuple(float(x) for x in extra["future_times_s"].tolist()) != tuple(HORIZONS):
                    raise ValueError("GraphGRU forecast times differ from target times")
                if name == "odebrain_inspired_hms":
                    times = extra["times_s"]
                    expected = ((0.0,) + tuple(HORIZONS) if
                                run["model"].architecture == "legacy_v1" else None)
                    if expected is not None and tuple(float(x) for x in times.tolist()) != expected:
                        raise ValueError("ODE trajectory times differ from target times")
                    if expected is None:
                        class_grid = torch.linspace(0., 1., 12, device=times.device)
                        if not torch.allclose(times[:12], class_grid) or not all(
                                bool(torch.any(torch.isclose(times, torch.tensor(h, device=times.device,
                                                                                 dtype=times.dtype))))
                                for h in HORIZONS):
                            raise ValueError("ODE trajectory lacks class grid or forecast horizons")
                predictions[name] = prediction
            batch_windows = [str(x) for x in batch["window_id"]]
            if any(window_id not in eeg_by_window for window_id in batch_windows):
                raise ValueError("Forecast batch has a window absent from the evaluated manifest")
            batch_eeg_ids = [eeg_by_window[window_id] for window_id in batch_windows]
            errors.add(predictions, target, future_mask, history["graph_mask"],
                       batch["patient_id"], batch_eeg_ids, ode["mean"], ode["std"])
            seen_windows.extend(batch_windows)
            seen_patients.extend(str(x) for x in batch["patient_id"])
            seen_eeg_ids.extend(batch_eeg_ids)
    if len(seen_windows) != len(selected) or set(seen_windows) != set(selected.window_id.astype(str)):
        raise ValueError("Forecast evaluation did not cover the selected split exactly")
    if not smoke and (len(seen_windows) != eligibility["eligible_test_windows"] or
                      len(set(seen_patients)) > int(table.loc[table.split == "test", "patient_id"].nunique())):
        raise ValueError("Formal forecast counts differ from audited eligible test windows")
    for run in (ode, gru):
        training_patients = set(table.set_index(table.window_id.astype(str)).loc[
            _training_window_ids(run, protocol), "patient_id"].astype(str))
        if training_patients & set(seen_patients):
            raise ValueError("Checkpoint training patients overlap evaluation patients")
    methods = errors.result()
    result = {
        "schema_version": "hms_forecast_comparison_v1" if smoke else "hms_forecast_comparison_v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_status": "real_data_smoke" if smoke else "formal",
        "training_protocol": "train_val_fixed_epoch_refit" if protocol == "refit" else "validation_selected_train_only",
        "smoke_subset": bool(smoke),
        "synthetic": False,
        "evaluation_scope": "frozen_smoke_val" if smoke else "forecast_eligible_held_out_test",
        "split": split,
        "split_hash": manifest_hash(table),
        "manifest_sha256": file_sha256(manifest_path),
        "window_count": len(seen_windows),
        "patient_count": len(set(seen_patients)),
        "n_windows": len(seen_windows),
        "n_patients": len(set(seen_patients)),
        "n_eeg_records": len(set(seen_eeg_ids)),
        "window_ids_sha256": _canonical_hash(seen_windows),
        "patient_ids_sha256": _canonical_hash(set(seen_patients)),
        "eeg_record_ids_sha256": _canonical_hash(set(seen_eeg_ids)),
        "forecast_eligibility": eligibility,
        "forecast_cut_seconds": cut,
        "horizons_seconds": list(HORIZONS),
        "target": "future electrode-node log1p band-power attributes",
        "mae_units": "log1p_band_power",
        "eeg_record_equal_mae_definition": "mean valid node-feature absolute error within each eeg_id, then unweighted mean across eeg_id",
        "validity_rule": "future node valid AND node observed at least once before cut; each of five features counted",
        "methods": methods,
        "rows": _comparison_rows(methods),
        "provenance": {
            "manifest": str(manifest_path),
            "graph_cache_dir": str(Path(cache).resolve()) if cache else None,
            "ode": _run_provenance(ode, protocol), "graph_gru": _run_provenance(gru, protocol),
            "persistence": "last valid observed node attribute repeated at each horizon",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, output_path)
    return result


def _run_provenance(run, protocol="initial"):
    config = run["config"]
    result = {"run_dir": str(run["run_dir"]),
            "model_name": config["model"]["name"],
            "checkpoint": str(run["checkpoint_path"]),
            "checkpoint_sha256": file_sha256(run["checkpoint_path"]),
            "checkpoint_epoch": int(run["checkpoint"]["epoch"]),
            "config_sha256": file_sha256(run["config_path"]),
            "graph_normalizer_sha256": file_sha256(run["normalizer_path"]),
            "training_manifest_sha256": config["manifest_sha256"],
            "initial_pretrain_checkpoint_sha256": config.get("init_checkpoint_sha256"),
            "training_task": config.get("data", {}).get("task", "classification"),
            "smoke_subset": bool(config.get("smoke_subset"))}
    if protocol == "refit":
        refit = config["refit"]
        result.update({
            "training_protocol": "train_val_fixed_epoch_refit",
            "training_window_count": len(_training_window_ids(run, protocol)),
            "training_window_ids_sha256": _canonical_hash(_training_window_ids(run, protocol)),
            "fixed_epochs": int(refit["fixed_epochs"]),
            "source_best_epoch_zero_indexed": int(refit["source_best_epoch_zero_indexed"]),
            "source_best_checkpoint": str(run["source_checkpoint_path"]),
            "source_best_checkpoint_sha256": refit["source_best_checkpoint_sha256"],
            "completion_sha256": file_sha256(run["completed_path"]),
            "validation_record_sha256": file_sha256(run["selection_path"]),
            "window_snapshot_sha256": file_sha256(run["snapshot_path"]),
        })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ode-run-dir", required=True)
    parser.add_argument("--gru-run-dir", required=True)
    parser.add_argument("--manifest", default="data/curated/split_manifest.csv")
    parser.add_argument("--output")
    parser.add_argument("--graph-cache-dir", help="Optional validated forecast graph cache")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Evaluate frozen validation smoke windows only")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="initial",
                        help="Use initial best.pt runs or fixed-epoch train+val forecast final.pt refits")
    args = parser.parse_args(argv)
    output = args.output or ("results/forecast_comparison_smoke.json" if args.smoke else
                             "results/refit/forecast_comparison.json" if args.protocol == "refit" else
                             "results/forecast_comparison.json")
    result = evaluate_forecast(args.ode_run_dir, args.gru_run_dir, args.manifest,
                               output, smoke=args.smoke, batch_size=args.batch_size,
                               workers=args.workers, device=args.device,
                               graph_cache_dir=args.graph_cache_dir,
                               protocol=args.protocol)
    print(json.dumps({"output": str(Path(output).resolve()),
                      "evaluation_scope": result["evaluation_scope"],
                      "window_count": result["window_count"],
                      "patient_count": result["patient_count"]}, indent=2))


if __name__ == "__main__":
    main()
