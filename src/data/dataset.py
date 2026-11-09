"""Lazy PyTorch dataset for HMS EEG windows.

Use one dataset instance per worker. The parquet handle is opened only while
reading; no shared HDF5 writer or inherited file descriptor is involved.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data._utils.collate import default_collate

from .hms import (CLASSES, ELECTRODES, SAMPLE_RATE, clean_eeg, eeg_spectrogram,
                  graph_frames, row_target, read_eeg_window)


class HMSWindowDataset(Dataset):
    def __init__(self, manifest_csv, split, task="classification", max_rows=None,
                 max_missing_fraction=0.05, forecast_cut_seconds=5,
                 forecast_horizons=(1, 3, 5), graph_mean=None, graph_std=None,
                 include_spectrogram=True, include_graph=True, graph_cache_dir=None):
        table = pd.read_csv(manifest_csv)
        if "split" not in table:
            raise ValueError("Use patient split manifest, not unsplit metadata")
        self.rows = table[(table.split == split) & (table.audit_reason == "ok")].reset_index(drop=True)
        self.task = task
        self.include_spectrogram = bool(include_spectrogram)
        self.include_graph = bool(include_graph)
        self.max_missing_fraction = max_missing_fraction
        self.cut = int(forecast_cut_seconds)
        self.horizons = tuple(int(h) for h in forecast_horizons)
        if task not in ("classification", "forecast"):
            raise ValueError("task must be classification or forecast")
        if task == "forecast" and (self.cut < 1 or max(self.horizons) + self.cut > 50):
            raise ValueError("forecast target falls outside 50 s window")
        if task == "forecast" and not self.include_graph:
            raise ValueError("forecast task requires graph features")
        if graph_cache_dir is not None and not self.include_graph:
            raise ValueError("graph cache requires graph features")
        self.graph_cache = None
        self.forecast_exclusion_counts = {}
        if graph_cache_dir is not None:
            from .graph_cache import GraphCacheReader
            self.graph_cache = GraphCacheReader(
                manifest_csv, graph_cache_dir, task=task,
                forecast_cut_seconds=self.cut, forecast_horizons=self.horizons,
                max_missing_fraction=self.max_missing_fraction)
            if task == "forecast":
                reasons = self.rows.window_id.astype(str).map(self.graph_cache.eligibility)
                if reasons.isna().any():
                    raise ValueError("forecast cache eligibility missing split window IDs")
                self.forecast_exclusion_counts = {str(reason): int(count) for reason, count in
                                                  reasons[reasons != "ok"].value_counts().items()}
                self.rows = self.rows.loc[reasons == "ok"].reset_index(drop=True)
        if max_rows is not None:
            groups = [group.index.tolist() for _, group in self.rows.groupby("patient_id", sort=True)]
            chosen = []
            depth = 0
            while len(chosen) < max_rows and any(depth < len(group) for group in groups):
                for group in groups:
                    if depth < len(group):
                        chosen.append(group[depth])
                        if len(chosen) == max_rows:
                            break
                depth += 1
            self.rows = self.rows.iloc[chosen].reset_index(drop=True)
        self.graph_mean = None if graph_mean is None else np.asarray(graph_mean, dtype=np.float32)
        self.graph_std = None if graph_std is None else np.asarray(graph_std, dtype=np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        eeg, observed = read_eeg_window(row.eeg_path, row.eeg_label_offset_seconds)
        if self.task == "forecast":
            if self.graph_cache is None:
                from .graph_cache import ForecastIneligibleError, forecast_graph_features
                try:
                    computed = forecast_graph_features(
                        eeg, observed, self.cut, self.horizons, self.max_missing_fraction)
                except ForecastIneligibleError as exc:
                    raise ValueError("forecast window %s ineligible: %s" %
                                     (row.window_id, exc.reason)) from exc
                obs_eeg, obs_mask = computed["observed_eeg"], computed["observed_mask"]
                nodes, adj, mask = computed["graph_nodes"], computed["graph_adj"], computed["graph_mask"]
                future_nodes, future_mask = computed["future_graph_nodes"], computed["future_graph_mask"]
            else:
                # Classification window may be 50 s valid but early forecast history invalid.
                # Reader has already excluded such rows using the eligibility index.
                cut_sample = self.cut * SAMPLE_RATE
                try:
                    obs_eeg, obs_mask = clean_eeg(eeg[:, :cut_sample], observed[:, :cut_sample],
                                                  max_missing_fraction=self.max_missing_fraction)
                except ValueError as exc:
                    raise ValueError("forecast cache input changed for window %s: %s" %
                                     (row.window_id, exc)) from exc
                cached = self.graph_cache.get(row.window_id)
                nodes, adj, mask = cached["graph_nodes"], cached["graph_adj"], cached["graph_mask"]
                future_nodes, future_mask = cached["future_graph_nodes"], cached["future_graph_mask"]
        else:
            eeg, valid = clean_eeg(eeg, observed, max_missing_fraction=self.max_missing_fraction)
            obs_eeg, obs_mask = eeg, valid
            if self.include_graph:
                if self.graph_cache is None:
                    nodes, adj, mask = graph_frames(eeg, valid)
                else:
                    cached = self.graph_cache.get(row.window_id)
                    nodes, adj, mask = cached["graph_nodes"], cached["graph_adj"], cached["graph_mask"]
            future_nodes = future_mask = None
        if self.include_graph and self.graph_mean is not None:
            nodes = (nodes - self.graph_mean) / np.maximum(self.graph_std, 1e-6)
            if future_nodes is not None:
                future_nodes = (future_nodes - self.graph_mean) / np.maximum(self.graph_std, 1e-6)
        batch = {
            "eeg": torch.from_numpy(obs_eeg.copy()),
            "eeg_mask": torch.from_numpy(obs_mask.copy()),
            "target": torch.from_numpy(row_target(row)),
            "class_id": torch.tensor(int(row.class_id), dtype=torch.long),
            "patient_id": str(row.patient_id),
            "window_id": str(row.window_id),
            "sample_rate": SAMPLE_RATE,
        }
        if self.include_spectrogram:
            batch["spectrogram"] = torch.from_numpy(eeg_spectrogram(obs_eeg).copy())
        if self.include_graph:
            batch.update({"graph_nodes": torch.from_numpy(nodes.copy()),
                          "graph_adj": torch.from_numpy(adj.copy()),
                          "graph_mask": torch.from_numpy(mask.copy())})
        if future_nodes is not None:
            batch["observed_eeg"] = torch.from_numpy(obs_eeg.copy())
            batch["future_graph_nodes"] = torch.from_numpy(future_nodes.copy())
            batch["future_graph_mask"] = torch.from_numpy(future_mask.copy())
            batch["future_times_s"] = torch.tensor(self.horizons, dtype=torch.float32)
            batch["input_end_sample"] = self.cut * SAMPLE_RATE
        return batch


def hms_collate(samples):
    """Keep shared horizon vector 1D, unlike default tensor stacking."""
    batch = default_collate(samples)
    if "future_times_s" in batch:
        times = batch["future_times_s"]
        if not torch.all(times == times[0]):
            raise ValueError("Forecast horizons differ within a batch")
        batch["future_times_s"] = times[0]
    return batch


class RecordingBatchSampler(Sampler):
    """Shuffle recordings, keeping nearby windows together for read-cache reuse."""

    def __init__(self, dataset, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.groups = [group.index.tolist() for _, group in dataset.rows.groupby("eeg_path", sort=True)]
        self.batch_size = int(batch_size)
        self.count = len(dataset)

    def __iter__(self):
        batch = []
        for group_number in torch.randperm(len(self.groups)).tolist():
            group = self.groups[group_number]
            for position in torch.randperm(len(group)).tolist():
                batch.append(group[position])
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size


def fit_graph_normalizer(dataset, max_windows=None):
    """Fit observable graph feature moments using *training* windows only."""
    if "train" not in set(dataset.rows.split):
        raise ValueError("Normalizer must be fitted on train split")
    sums = np.zeros(5, dtype=np.float64)
    sums2 = np.zeros(5, dtype=np.float64)
    n = 0
    limit = len(dataset) if max_windows is None else min(len(dataset), max_windows)
    for i in range(limit):
        row = dataset.rows.iloc[i]
        if getattr(dataset, "graph_cache", None) is not None:
            cached = dataset.graph_cache.get(row.window_id)
            nodes, mask = cached["graph_nodes"], cached["graph_mask"]
        else:
            eeg, observed = read_eeg_window(row.eeg_path, row.eeg_label_offset_seconds)
            if dataset.task == "forecast":
                eeg = eeg[:, :dataset.cut*SAMPLE_RATE]
                observed = observed[:, :dataset.cut*SAMPLE_RATE]
            try:
                eeg, valid = clean_eeg(eeg, observed, max_missing_fraction=dataset.max_missing_fraction)
            except ValueError as exc:
                if dataset.task == "forecast" and str(exc) in (
                        "constant_channel", "too_many_missing_samples"):
                    raise ValueError("forecast window %s ineligible: history_%s" %
                                     (row.window_id, exc)) from exc
                raise
            nodes, _, mask = graph_frames(eeg, valid)
            if dataset.task == "forecast" and not mask.any():
                raise ValueError("forecast window %s ineligible: history_no_valid_graph_input" %
                                 row.window_id)
        arr = nodes[mask]
        sums += arr.sum(axis=0)
        sums2 += (arr.astype(np.float64)**2).sum(axis=0)
        n += len(arr)
    if n == 0:
        raise ValueError("No valid graph features for normalization")
    mean = sums / n
    std = np.sqrt(np.maximum(sums2/n - mean**2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)
