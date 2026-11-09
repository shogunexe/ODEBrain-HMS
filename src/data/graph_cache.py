"""Resumable, recording-grouped graph feature cache for HMS windows.

Build with ``python -m src.data.graph_cache build --manifest ... --output ...``.
One compressed shard contains all audited windows of one EEG recording. The
cache contains only observed graph inputs and, for forecasting, future graph
*targets*. Raw EEG and labels remain in the original manifest/files.
"""

import argparse
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from .hms import (ELECTRODES, SAMPLE_RATE, WINDOW_SECONDS, clean_eeg,
                  graph_frames)


SCHEMA_VERSION = "hms_graph_cache_v1"
FORECAST_SCHEMA_VERSION = "hms_graph_forecast_cache_v2"
FORECAST_REASONS = frozenset((
    "ok", "history_constant_channel", "history_too_many_missing_samples",
    "history_no_valid_graph_input", "future_constant_channel",
    "future_too_many_missing_samples", "future_no_valid_target",
))


class ForecastIneligibleError(ValueError):
    """An audited classification window cannot support strict 5 s forecasting."""

    def __init__(self, reason):
        if reason not in FORECAST_REASONS or reason == "ok":
            raise ValueError("unknown forecast ineligibility reason: %s" % reason)
        self.reason = reason
        super().__init__(reason)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _settings(task, cut, horizons, max_missing_fraction):
    cut = int(cut)
    horizons = tuple(int(h) for h in horizons)
    if task not in ("classification", "forecast"):
        raise ValueError("task must be classification or forecast")
    if task == "forecast" and (cut < 1 or not horizons or
                               tuple(sorted(set(horizons))) != horizons or
                               cut + horizons[-1] > WINDOW_SECONDS):
        raise ValueError("invalid forecast cut or horizons")
    if not 0 <= max_missing_fraction < 1:
        raise ValueError("max_missing_fraction must be in [0,1)")
    return {"task": task, "forecast_cut_seconds": cut,
            "forecast_horizons": list(horizons),
            "max_missing_fraction": float(max_missing_fraction)}


def _manifest_rows(manifest_path):
    table = pd.read_csv(manifest_path)
    required = {"window_id", "eeg_path", "eeg_label_offset_seconds", "split", "audit_reason"}
    if not required.issubset(table.columns):
        raise ValueError("graph cache needs an audited patient split manifest")
    valid = table.loc[table.audit_reason == "ok"].copy()
    if valid.window_id.isna().any() or valid.window_id.astype(str).duplicated().any():
        raise ValueError("valid window IDs must be nonempty and unique")
    if valid.eeg_path.isna().any() or valid.eeg_label_offset_seconds.isna().any():
        raise ValueError("valid windows need EEG paths and offsets")
    return valid


def _record_specs(table):
    specs = []
    for path, group in table.groupby("eeg_path", sort=True):
        path = str(Path(path).resolve())
        group = group.sort_values("window_id")
        specs.append({"path": path,
                      "shard": hashlib.sha256(path.encode("utf-8")).hexdigest() + ".npz",
                      "window_ids": group.window_id.astype(str).tolist(),
                      "offset_seconds": group.eeg_label_offset_seconds.astype(float).tolist()})
    return specs


def _metadata(manifest_path, settings, records):
    version = FORECAST_SCHEMA_VERSION if settings["task"] == "forecast" else SCHEMA_VERSION
    return {"schema_version": version,
            "manifest_sha256": file_sha256(manifest_path),
            "settings": settings, "records": records,
            "n_windows": sum(len(record["window_ids"]) for record in records)}


def _base_index(index):
    """Remove forecast results while retaining immutable manifest provenance."""
    base = {key: value for key, value in index.items() if key != "forecast_eligibility"}
    if base.get("schema_version") == FORECAST_SCHEMA_VERSION:
        base["records"] = [{key: value for key, value in record.items()
                            if key != "eligibility_reason"} for record in base["records"]]
    return base


def _eligibility_summary(records, manifest_rows):
    reasons = {window_id: reason for record in records
               for window_id, reason in zip(record["window_ids"], record["eligibility_reason"])}
    if len(reasons) != sum(len(record["window_ids"]) for record in records):
        raise ValueError("forecast eligibility window IDs are incomplete or duplicated")
    eligible_ids = sorted(window_id for window_id, reason in reasons.items() if reason == "ok")
    result = {"total_windows": len(reasons), "eligible_windows": 0,
              "excluded_windows": 0, "reason_counts": {}, "by_split": {},
              "eligible_window_ids_sha256": hashlib.sha256(
                  "\n".join(eligible_ids).encode("utf-8")).hexdigest()}
    for split, group in manifest_rows.groupby("split", sort=True):
        counts = {}
        split_eligible_ids = []
        for window_id in group.window_id.astype(str):
            reason = reasons[window_id]
            counts[reason] = counts.get(reason, 0) + 1
            result["reason_counts"][reason] = result["reason_counts"].get(reason, 0) + 1
            if reason == "ok":
                split_eligible_ids.append(window_id)
        eligible = counts.get("ok", 0)
        result["by_split"][str(split)] = {"eligible": eligible,
                                           "excluded": len(group) - eligible,
                                           "reason_counts": counts,
                                           "eligible_window_ids_sha256": hashlib.sha256(
                                               "\n".join(sorted(split_eligible_ids)).encode("utf-8")).hexdigest()}
        result["eligible_windows"] += eligible
    result["excluded_windows"] = result["total_windows"] - result["eligible_windows"]
    return result


def forecast_graph_features(eeg, observed, cut, horizons, max_missing_fraction=0.05):
    """Compute strict past graph inputs and separately masked future targets."""
    cut_samples = int(cut) * SAMPLE_RATE
    try:
        past, past_mask = clean_eeg(eeg[:, :cut_samples], observed[:, :cut_samples],
                                    max_missing_fraction=max_missing_fraction)
    except ValueError as exc:
        if str(exc) in ("constant_channel", "too_many_missing_samples"):
            raise ForecastIneligibleError("history_" + str(exc)) from exc
        raise
    nodes, adj, mask = graph_frames(past, past_mask)
    if not mask.any():
        raise ForecastIneligibleError("history_no_valid_graph_input")
    target_end = cut_samples + max(horizons) * SAMPLE_RATE
    try:
        future, future_mask = clean_eeg(eeg[:, cut_samples:target_end],
                                        observed[:, cut_samples:target_end],
                                        max_missing_fraction=max_missing_fraction)
    except ValueError as exc:
        if str(exc) in ("constant_channel", "too_many_missing_samples"):
            raise ForecastIneligibleError("future_" + str(exc)) from exc
        raise
    future_frames, _, future_masks = graph_frames(future, future_mask)
    indices = np.asarray(horizons) - 1
    targets = future_frames[indices]
    target_masks = future_masks[indices]
    if not target_masks.any(axis=1).all():
        raise ForecastIneligibleError("future_no_valid_target")
    return {"observed_eeg": past, "observed_mask": past_mask,
            "graph_nodes": nodes, "graph_adj": adj, "graph_mask": mask,
            "future_graph_nodes": targets, "future_graph_mask": target_masks}


def _atomic_json(path, value):
    descriptor, temp_path = tempfile.mkstemp(prefix=".index_", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _atomic_npz(path, arrays):
    descriptor, temp_path = tempfile.mkstemp(prefix=".shard_", suffix=".npz", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as output:
            np.savez_compressed(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@contextmanager
def _build_lock(directory):
    with (directory / ".build.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _read_shard(path, record, settings):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    ids = arrays.get("window_ids")
    offsets = arrays.get("offset_seconds")
    n = len(record["window_ids"])
    frames = settings["forecast_cut_seconds"] if settings["task"] == "forecast" else WINDOW_SECONDS
    expected = {"window_ids", "offset_seconds", "graph_nodes", "graph_adj", "graph_mask"}
    if settings["task"] == "forecast":
        expected |= {"future_graph_nodes", "future_graph_mask", "eligibility_reason"}
    if set(arrays) != expected or ids is None or offsets is None:
        raise ValueError("graph cache shard fields differ from metadata")
    if ids.astype(str).tolist() != record["window_ids"] or not np.array_equal(
            offsets, np.asarray(record["offset_seconds"], dtype=np.float64)):
        raise ValueError("graph cache shard window IDs or offsets differ from metadata")
    graph_count = n
    if settings["task"] == "forecast":
        reasons = arrays["eligibility_reason"].astype(str).tolist()
        if len(reasons) != n or not set(reasons).issubset(FORECAST_REASONS):
            raise ValueError("forecast eligibility reasons invalid")
        if "eligibility_reason" in record and reasons != record["eligibility_reason"]:
            raise ValueError("forecast eligibility reasons differ from index")
        graph_count = reasons.count("ok")
    shapes = {"graph_nodes": (graph_count, frames, len(ELECTRODES), 5),
              "graph_adj": (graph_count, frames, len(ELECTRODES), len(ELECTRODES)),
              "graph_mask": (graph_count, frames, len(ELECTRODES))}
    if settings["task"] == "forecast":
        h = len(settings["forecast_horizons"])
        shapes.update({"future_graph_nodes": (graph_count, h, len(ELECTRODES), 5),
                       "future_graph_mask": (graph_count, h, len(ELECTRODES))})
    for name, shape in shapes.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError("invalid graph cache shard %s" % name)
    return arrays


def _build_record(record, settings):
    frame = pd.read_parquet(record["path"], columns=list(ELECTRODES))
    data = frame.to_numpy(dtype=np.float32).T
    arrays = {"window_ids": np.asarray(record["window_ids"], dtype=str),
              "offset_seconds": np.asarray(record["offset_seconds"], dtype=np.float64)}
    fields = {"graph_nodes": [], "graph_adj": [], "graph_mask": []}
    if settings["task"] == "forecast":
        fields.update({"future_graph_nodes": [], "future_graph_mask": []})
    reasons = []
    for window_id, offset in zip(record["window_ids"], record["offset_seconds"]):
        start = int(round(offset * SAMPLE_RATE))
        stop = start + WINDOW_SECONDS * SAMPLE_RATE
        if start < 0 or stop > data.shape[1]:
            raise ValueError("audited window is outside recording: %s at %s" % (record["path"], offset))
        eeg = data[:, start:stop]
        observed = np.isfinite(eeg)
        if settings["task"] == "forecast":
            try:
                computed = forecast_graph_features(
                    eeg, observed, settings["forecast_cut_seconds"],
                    settings["forecast_horizons"], settings["max_missing_fraction"])
            except ForecastIneligibleError as exc:
                reasons.append(exc.reason)
                continue
            reasons.append("ok")
            nodes, adj, mask = (computed["graph_nodes"], computed["graph_adj"],
                                computed["graph_mask"])
            fields["future_graph_nodes"].append(computed["future_graph_nodes"])
            fields["future_graph_mask"].append(computed["future_graph_mask"])
        else:
            clean, valid = clean_eeg(eeg, observed,
                                     max_missing_fraction=settings["max_missing_fraction"])
            nodes, adj, mask = graph_frames(clean, valid)
        fields["graph_nodes"].append(nodes)
        fields["graph_adj"].append(adj)
        fields["graph_mask"].append(mask)
    if settings["task"] == "forecast":
        arrays["eligibility_reason"] = np.asarray(reasons, dtype=str)
        cut = settings["forecast_cut_seconds"]
        count = len(settings["forecast_horizons"])
        empty_shapes = {"graph_nodes": (0, cut, len(ELECTRODES), 5),
                        "graph_adj": (0, cut, len(ELECTRODES), len(ELECTRODES)),
                        "graph_mask": (0, cut, len(ELECTRODES)),
                        "future_graph_nodes": (0, count, len(ELECTRODES), 5),
                        "future_graph_mask": (0, count, len(ELECTRODES))}
        arrays.update({name: np.stack(values) if values else np.empty(
            empty_shapes[name], dtype=bool if name.endswith("mask") else np.float32)
                       for name, values in fields.items()})
    else:
        arrays.update({name: np.stack(values) for name, values in fields.items()})
    return arrays


def _build_one_record(record, settings, output_dir):
    """Worker owns one recording and writes its shard; no large arrays cross IPC."""
    shard_path = Path(output_dir) / record["shard"]
    if shard_path.exists():
        try:
            arrays = _read_shard(shard_path, record, settings)
            reasons = arrays["eligibility_reason"].astype(str).tolist() if settings["task"] == "forecast" else None
            return "reused", reasons
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            pass
    try:
        arrays = _build_record(record, settings)
        _atomic_npz(shard_path, arrays)
    except Exception as exc:
        raise RuntimeError("graph cache failed for EEG %s: %s" % (record["path"], exc)) from exc
    reasons = arrays["eligibility_reason"].astype(str).tolist() if settings["task"] == "forecast" else None
    return "built", reasons


def _build_records(records, settings, output_dir, workers):
    total = len(records)
    made = reused = completed = 0
    eligibility_by_shard = {}
    interval = max(1, min(100, total // 20))
    last_log = time.monotonic()
    print("graph cache: %d recordings, %d workers" % (total, workers), flush=True)

    def progress(record, result):
        nonlocal made, reused, completed, last_log
        status, reasons = result
        completed += 1
        if status == "built":
            made += 1
        elif status == "reused":
            reused += 1
        else:
            raise ValueError("unknown graph cache worker result: %s" % status)
        if settings["task"] == "forecast":
            eligibility_by_shard[record["shard"]] = reasons
        now = time.monotonic()
        if completed == total or completed % interval == 0 or now - last_log >= 30:
            print("graph cache: %d/%d recordings; built=%d reused=%d" %
                  (completed, total, made, reused), flush=True)
            last_log = now

    if workers == 1:
        for record in records:
            progress(record, _build_one_record(record, settings, output_dir))
    else:
        # At most 2*workers pending futures, regardless of manifest size.
        record_iter = iter(records)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending = {}

            def submit_next():
                try:
                    record = next(record_iter)
                except StopIteration:
                    return False
                future = executor.submit(_build_one_record, record, settings, str(output_dir))
                pending[future] = record
                return True

            for _ in range(min(total, 2 * workers)):
                submit_next()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    record = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        for other in pending:
                            other.cancel()
                        raise RuntimeError("graph cache worker failed for EEG %s" % record["path"]) from exc
                    progress(record, result)
                    submit_next()
    return made, reused, eligibility_by_shard


def build_graph_cache(manifest_path, output_dir, task="classification", forecast_cut_seconds=5,
                      forecast_horizons=(1, 3, 5), max_missing_fraction=0.05,
                      workers=4):
    """Build missing/corrupt shards, reading each EEG parquet once per recording."""
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be at least one")
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    settings = _settings(task, forecast_cut_seconds, forecast_horizons, max_missing_fraction)
    manifest_rows = _manifest_rows(manifest_path)
    records = _record_specs(manifest_rows)
    expected = _metadata(manifest_path, settings, records)
    output_dir.mkdir(parents=True, exist_ok=True)
    with _build_lock(output_dir):
        index_path = output_dir / "index.json"
        if index_path.exists():
            current = json.loads(index_path.read_text(encoding="utf-8"))
            if _base_index(current) != expected:
                raise ValueError("cache manifest SHA256, window IDs, or settings changed; choose a new output directory")
        else:
            if any(output_dir.glob("*.npz")):
                raise ValueError("cache shards exist without index.json; choose a clean output directory")
            _atomic_json(index_path, expected)
        made, reused, eligibility = _build_records(records, settings, output_dir, workers)
        summary = None
        if task == "forecast":
            indexed_records = [dict(record, eligibility_reason=eligibility[record["shard"]])
                               for record in records]
            summary = _eligibility_summary(indexed_records, manifest_rows)
            finished = dict(expected, records=indexed_records, forecast_eligibility=summary)
            _atomic_json(index_path, finished)
    result = {"output_dir": str(output_dir), "manifest_sha256": expected["manifest_sha256"],
            "n_windows": expected["n_windows"], "n_recordings": len(records),
            "built_recordings": made, "reused_recordings": reused, "settings": settings}
    if summary is not None:
        result["forecast_eligibility"] = summary
    return result


class GraphCacheReader:
    """Validate cache provenance and read recording shards lazily per worker."""

    def __init__(self, manifest_path, cache_dir, task="classification", forecast_cut_seconds=5,
                 forecast_horizons=(1, 3, 5), max_missing_fraction=0.05, max_open_recordings=2):
        self.directory = Path(cache_dir)
        self.settings = _settings(task, forecast_cut_seconds, forecast_horizons, max_missing_fraction)
        index = json.loads((self.directory / "index.json").read_text(encoding="utf-8"))
        manifest_rows = _manifest_rows(manifest_path)
        expected = _metadata(manifest_path, self.settings, _record_specs(manifest_rows))
        if _base_index(index) != expected:
            raise ValueError("graph cache manifest SHA256, window IDs, or settings mismatch")
        self.eligibility = None
        self.eligibility_summary = None
        if task == "forecast":
            if "forecast_eligibility" not in index or any(
                    "eligibility_reason" not in record for record in index["records"]):
                raise ValueError("forecast cache is incomplete: eligibility index missing")
            for record in index["records"]:
                reasons = record["eligibility_reason"]
                if (len(reasons) != len(record["window_ids"]) or
                        not set(reasons).issubset(FORECAST_REASONS)):
                    raise ValueError("forecast eligibility reasons invalid")
            computed = _eligibility_summary(index["records"], manifest_rows)
            if index["forecast_eligibility"] != computed:
                raise ValueError("forecast eligibility summary or window ID hash mismatch")
            self.eligibility_summary = computed
            self.eligibility = {window_id: reason for record in index["records"]
                                for window_id, reason in zip(record["window_ids"], record["eligibility_reason"])}
        self.locations = {}
        for record in index["records"]:
            eligible = [(window_id, reason) for window_id, reason in zip(
                record["window_ids"], record.get("eligibility_reason", ["ok"] * len(record["window_ids"])))
                        if reason == "ok"]
            for position, (window_id, _) in enumerate(eligible):
                if window_id in self.locations:
                    raise ValueError("duplicate window ID in graph cache")
                self.locations[window_id] = record, position
        self.open_records = OrderedDict()
        self.max_open_recordings = max(1, int(max_open_recordings))

    def get(self, window_id):
        window_id = str(window_id)
        if window_id not in self.locations:
            if self.eligibility is not None and window_id in self.eligibility:
                raise KeyError("forecast window %s ineligible: %s" %
                               (window_id, self.eligibility[window_id]))
            raise KeyError("window ID absent from graph cache: %s" % window_id)
        record, position = self.locations[window_id]
        key = record["shard"]
        if key not in self.open_records:
            arrays = _read_shard(self.directory / key, record, self.settings)
            self.open_records[key] = arrays
            if len(self.open_records) > self.max_open_recordings:
                self.open_records.popitem(last=False)
        self.open_records.move_to_end(key)
        arrays = self.open_records[key]
        return {name: arrays[name][position] for name in arrays
                if name not in ("window_ids", "offset_seconds", "eligibility_reason")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--manifest", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--task", choices=("classification", "forecast"), default="classification")
    build.add_argument("--cut", type=int, default=5)
    build.add_argument("--horizons", default="1,3,5")
    build.add_argument("--max-missing-fraction", type=float, default=0.05)
    build.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    horizons = tuple(int(value) for value in args.horizons.split(","))
    result = build_graph_cache(args.manifest, args.output, task=args.task,
                               forecast_cut_seconds=args.cut, forecast_horizons=horizons,
                               max_missing_fraction=args.max_missing_fraction,
                               workers=args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
