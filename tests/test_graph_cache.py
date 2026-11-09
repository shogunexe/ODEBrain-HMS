import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import HMSWindowDataset, fit_graph_normalizer
from src.data.graph_cache import GraphCacheReader, build_graph_cache
from src.data.hms import CLASSES, ELECTRODES


def _recording(path, length_seconds=52, future_shift=0):
    time = np.arange(length_seconds * 200, dtype=np.float32) / 200
    data = {}
    for i, name in enumerate(ELECTRODES):
        wave = np.sin(2 * np.pi * (2 + i % 4) * time + i * 0.1)
        wave += 0.15 * np.sin(2 * np.pi * 13 * time + i * 0.03)
        wave[1000:] += future_shift * np.sin(2 * np.pi * 3.5 * time[1000:])
        data[name] = wave.astype(np.float32)
    pd.DataFrame(data).to_parquet(path, index=False)


def _manifest(path, recordings):
    rows = []
    for number, (recording, offset) in enumerate(recordings):
        row = {"window_id": "row_%d" % number, "eeg_id": number // 2,
               "patient_id": number, "eeg_path": str(recording),
               "eeg_label_offset_seconds": offset, "split": "train",
               "audit_reason": "ok", "class_id": number % 6}
        row.update({"target_" + name: float(i == number % 6)
                    for i, name in enumerate(CLASSES)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_classification_cache_parity_resume_and_provenance(tmp_path, monkeypatch):
    first = tmp_path / "one.parquet"
    second = tmp_path / "two.parquet"
    _recording(first)
    _recording(second, length_seconds=50)
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [(first, 0), (first, 2), (second, 0)])
    uncached = HMSWindowDataset(manifest, "train", include_spectrogram=False)
    expected = [uncached[i] for i in range(3)]
    cache_dir = tmp_path / "graph_classification"

    import src.data.graph_cache as module
    original_read = module.pd.read_parquet
    reads = []
    def counted_read(*args, **kwargs):
        reads.append(str(args[0]))
        return original_read(*args, **kwargs)
    monkeypatch.setattr(module.pd, "read_parquet", counted_read)
    built = build_graph_cache(manifest, cache_dir, workers=1)
    assert built["built_recordings"] == 2 and len(reads) == 2
    assert len(set(reads)) == 2

    cached = HMSWindowDataset(manifest, "train", graph_cache_dir=cache_dir,
                              include_spectrogram=False)
    for i in range(3):
        sample = cached[i]
        for name in ("graph_nodes", "graph_adj", "graph_mask"):
            np.testing.assert_array_equal(sample[name].numpy(), expected[i][name].numpy())
    for actual, reference in zip(fit_graph_normalizer(cached), fit_graph_normalizer(uncached)):
        np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-7)

    shards = sorted(cache_dir.glob("*.npz"))
    before = {path.name: path.stat().st_mtime_ns for path in shards}
    resumed = build_graph_cache(manifest, cache_dir, workers=1)
    assert resumed["built_recordings"] == 0 and resumed["reused_recordings"] == 2
    assert before == {path.name: path.stat().st_mtime_ns for path in shards}

    shards[0].unlink()
    resumed = build_graph_cache(manifest, cache_dir, workers=1)
    assert resumed["built_recordings"] == 1 and resumed["reused_recordings"] == 1
    shards[0].write_bytes(b"interrupted write")
    resumed = build_graph_cache(manifest, cache_dir, workers=1)
    assert resumed["built_recordings"] == 1 and resumed["reused_recordings"] == 1
    table = pd.read_csv(manifest)
    table.loc[0, "split"] = "test"
    table.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="manifest SHA256"):
        GraphCacheReader(manifest, cache_dir)
    with pytest.raises(ValueError, match="manifest SHA256"):
        build_graph_cache(manifest, cache_dir, workers=1)

    with pytest.raises(ValueError, match="workers"):
        build_graph_cache(manifest, tmp_path / "invalid", workers=0)


def test_parallel_cache_matches_serial_and_resumes(tmp_path, capsys):
    paths = [tmp_path / ("record_%d.parquet" % i) for i in range(3)]
    for path in paths:
        _recording(path, length_seconds=50)
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [(path, 0) for path in paths])
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    one = build_graph_cache(manifest, serial, workers=1)
    two = build_graph_cache(manifest, parallel, workers=2)
    assert (one["built_recordings"], two["built_recordings"]) == (3, 3)
    assert "3/3 recordings" in capsys.readouterr().out
    for serial_shard in serial.glob("*.npz"):
        parallel_shard = parallel / serial_shard.name
        with np.load(serial_shard, allow_pickle=False) as left, np.load(parallel_shard, allow_pickle=False) as right:
            assert set(left.files) == set(right.files)
            for name in left.files:
                np.testing.assert_array_equal(left[name], right[name])
    resumed = build_graph_cache(manifest, parallel, workers=2)
    assert resumed["built_recordings"] == 0 and resumed["reused_recordings"] == 3
    assert not list(parallel.glob(".shard_*"))


def test_parallel_worker_error_keeps_index_resumable(tmp_path):
    missing = tmp_path / "missing.parquet"
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [(missing, 0)])
    output = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="missing.parquet"):
        build_graph_cache(manifest, output, workers=2)
    assert (output / "index.json").is_file()
    _recording(missing, length_seconds=50)
    result = build_graph_cache(manifest, output, workers=2)
    assert result["built_recordings"] == 1


def test_forecast_cache_preserves_cut_and_future_supervision(tmp_path):
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    _recording(a, length_seconds=50)
    _recording(b, length_seconds=50, future_shift=3)
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [(a, 0), (b, 0)])
    cache_dir = tmp_path / "graph_forecast"
    built = build_graph_cache(manifest, cache_dir, task="forecast")
    assert built["n_windows"] == 2
    uncached = HMSWindowDataset(manifest, "train", task="forecast", include_spectrogram=False)
    cached = HMSWindowDataset(manifest, "train", task="forecast", graph_cache_dir=cache_dir,
                              include_spectrogram=False)
    for i in range(2):
        original = uncached[i]
        sample = cached[i]
        for name in ("graph_nodes", "graph_adj", "graph_mask",
                     "future_graph_nodes", "future_graph_mask", "observed_eeg"):
            np.testing.assert_array_equal(sample[name].numpy(), original[name].numpy())
        assert sample["input_end_sample"] == 1000
        assert sample["graph_nodes"].shape[0] == 5
    np.testing.assert_array_equal(cached[0]["graph_nodes"].numpy(),
                                  cached[1]["graph_nodes"].numpy())
    assert not np.array_equal(cached[0]["future_graph_nodes"].numpy(),
                              cached[1]["future_graph_nodes"].numpy())
    for actual, reference in zip(fit_graph_normalizer(cached), fit_graph_normalizer(uncached)):
        np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-7)
    with pytest.raises(ValueError, match="settings mismatch"):
        HMSWindowDataset(manifest, "train", task="forecast", forecast_cut_seconds=4,
                         graph_cache_dir=cache_dir)


def test_forecast_eligibility_exclusions_and_index(tmp_path):
    good = tmp_path / "good.parquet"
    bad_history = tmp_path / "bad_history.parquet"
    bad_target = tmp_path / "bad_target.parquet"
    for path in (good, bad_history, bad_target):
        _recording(path, length_seconds=50)
    history = pd.read_parquet(bad_history)
    history.iloc[:1000, :] = 0
    history.to_parquet(bad_history, index=False)
    target = pd.read_parquet(bad_target)
    target.iloc[1000:1200, :] = 0
    target.to_parquet(bad_target, index=False)
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [(good, 0), (bad_history, 0), (bad_target, 0)])
    table = pd.read_csv(manifest)
    table.loc[1, "split"] = "val"
    table.loc[2, "split"] = "test"
    table.to_csv(manifest, index=False)

    cache_dir = tmp_path / "forecast"
    result = build_graph_cache(manifest, cache_dir, task="forecast", workers=2)
    summary = result["forecast_eligibility"]
    assert summary["total_windows"] == 3
    assert summary["eligible_windows"] == 1
    assert summary["excluded_windows"] == 2
    assert summary["by_split"]["train"]["eligible"] == 1
    assert summary["by_split"]["val"]["reason_counts"] == {"history_constant_channel": 1}
    assert summary["by_split"]["test"]["reason_counts"] == {"future_no_valid_target": 1}
    assert summary["eligible_window_ids_sha256"] == hashlib.sha256(b"row_0").hexdigest()
    reader = GraphCacheReader(manifest, cache_dir, task="forecast")
    assert reader.eligibility == {"row_0": "ok", "row_1": "history_constant_channel",
                                  "row_2": "future_no_valid_target"}
    with pytest.raises(KeyError, match="history_constant_channel"):
        reader.get("row_1")
    with pytest.raises(KeyError, match="future_no_valid_target"):
        reader.get("row_2")
    train = HMSWindowDataset(manifest, "train", task="forecast", graph_cache_dir=cache_dir)
    val = HMSWindowDataset(manifest, "val", task="forecast", graph_cache_dir=cache_dir)
    test = HMSWindowDataset(manifest, "test", task="forecast", graph_cache_dir=cache_dir)
    assert (len(train), len(val), len(test)) == (1, 0, 0)
    assert val.forecast_exclusion_counts == {"history_constant_channel": 1}
    assert test.forecast_exclusion_counts == {"future_no_valid_target": 1}
    with pytest.raises(ValueError, match="row_1 ineligible: history_constant_channel"):
        HMSWindowDataset(manifest, "val", task="forecast")[0]
    with pytest.raises(ValueError, match="row_2 ineligible: future_no_valid_target"):
        HMSWindowDataset(manifest, "test", task="forecast")[0]
    assert len(HMSWindowDataset(manifest, "val", task="classification")) == 1
    resumed = build_graph_cache(manifest, cache_dir, task="forecast", workers=2)
    assert resumed["reused_recordings"] == 3
    assert resumed["forecast_eligibility"] == summary

    index_path = cache_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["forecast_eligibility"]["eligible_windows"] = 2
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="eligibility summary"):
        GraphCacheReader(manifest, cache_dir, task="forecast")


def test_real_constant_history_record_is_excluded_without_discarding_record(tmp_path):
    path = Path("data/raw/train_eegs/1001717358.parquet").resolve()
    full_manifest = Path("data/curated/split_manifest.csv")
    if not path.is_file() or not full_manifest.is_file():
        pytest.skip("full HMS data unavailable")
    rows = pd.read_csv(full_manifest)
    rows = rows[(rows.eeg_id == 1001717358) &
                (rows.eeg_label_offset_seconds.isin([0, 4, 12, 42]))].copy()
    assert len(rows) == 4
    manifest = tmp_path / "real_subset.csv"
    rows.to_csv(manifest, index=False)
    result = build_graph_cache(manifest, tmp_path / "forecast", task="forecast", workers=1)
    summary = result["forecast_eligibility"]
    assert summary["total_windows"] == 4
    assert summary["reason_counts"] == {"history_constant_channel": 3, "ok": 1}
    reader = GraphCacheReader(manifest, tmp_path / "forecast", task="forecast")
    assert len(reader.locations) == 1
    eligible_id = rows.loc[rows.eeg_label_offset_seconds == 42, "window_id"].iloc[0]
    assert reader.get(eligible_id)["graph_nodes"].shape[0] == 5
