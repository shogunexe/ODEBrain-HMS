import numpy as np
import pandas as pd
import pytest

from src.cli import require_full_data
from src.data import dataset as dataset_module
from src.data.hms import (CLASSES, ELECTRODES, audit_manifest, audit_window_quality, clean_eeg,
                          graph_frames, intervals_union_seconds, validate_metadata)
from src.data.splits import assign_patient_splits
from src.evaluation.metrics import sample_kl


def test_vote_order_and_patient_disjoint_split():
    rows = []
    for patient in range(20):
        for offset in (0, 10):
            row = {"eeg_id": patient, "patient_id": patient,
                   "eeg_label_offset_seconds": offset, "expert_consensus": CLASSES[patient % 6],
                   "eeg_path": "/missing", "file_exists": False, "window_id": "%d_%d" % (patient, offset)}
            row.update({name + "_vote": int(i == patient % 6) for i, name in enumerate(CLASSES)})
            rows.append(row)
    table = validate_metadata(pd.DataFrame(rows))
    split = assign_patient_splits(table)
    sets = [set(split.loc[split.split == name, "patient_id"]) for name in ("train", "val", "test")]
    assert all(not (sets[i] & sets[j]) for i in range(3) for j in range(i + 1, 3))
    assert list(table.loc[0, ["target_" + x for x in CLASSES]]) == [1, 0, 0, 0, 0, 0]


def test_overlap_and_missing_mask():
    assert intervals_union_seconds([(0, 50), (10, 60)]) == 60
    x = np.tile(np.sin(np.arange(400) / 10), (len(ELECTRODES), 1)).astype("float32")
    observed = np.ones_like(x, dtype=bool)
    x[0, 10:12] = np.nan
    observed[0, 10:12] = False
    x[0, 100:130] = np.nan
    observed[0, 100:130] = False
    cleaned, mask = clean_eeg(x, observed, max_missing_fraction=0.1)
    assert mask[0, 10:12].all()
    assert not mask[0, 100:130].any()
    assert np.all(cleaned[0, 100:130] == 0)


def test_graph_history_only_and_kl():
    rng = np.random.RandomState(42)
    past = rng.randn(19, 1000).astype("float32")
    future_a = rng.randn(19, 1000).astype("float32")
    future_b = rng.randn(19, 1000).astype("float32") + 3
    first = graph_frames(np.concatenate([past, future_a], axis=1), np.ones((19, 2000), bool))[0][:5]
    second = graph_frames(np.concatenate([past, future_b], axis=1), np.ones((19, 2000), bool))[0][:5]
    np.testing.assert_allclose(first, second, rtol=0, atol=0)
    y = np.eye(6)[[0, 1]]
    q = y * .9 + .1 / 6
    np.testing.assert_allclose(sample_kl(y, q), [-np.log(q[0, 0]), -np.log(q[1, 1])], rtol=1e-10)


def test_formal_run_rejects_partial_or_unaudited_data(tmp_path):
    metadata = pd.DataFrame({"eeg_id": [1, 1, 2]})
    paths = [tmp_path / "1.parquet", tmp_path / "2.parquet"]
    for path in paths:
        path.touch()
    manifest = pd.DataFrame({"eeg_id": [1, 1, 2], "file_exists": [True, True, True],
                             "eeg_path": [str(paths[0]), str(paths[0]), str(paths[1])],
                             "audit_reason": ["ok"] * 3, "missing_fraction": [0.0] * 3,
                             "raw_recording_seconds": [50.0] * 3})
    metadata_path = tmp_path / "train.csv"
    manifest_path = tmp_path / "manifest.csv"
    metadata.to_csv(metadata_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    require_full_data(manifest_path, metadata_path)
    manifest.loc[2, "file_exists"] = False
    manifest.to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match="every EEG parquet"):
        require_full_data(manifest_path, metadata_path)
    manifest.loc[2, "file_exists"] = True
    manifest.loc[2, "missing_fraction"] = np.nan
    manifest.to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match="quality checks"):
        require_full_data(manifest_path, metadata_path)


def test_forecast_normalizer_never_reads_future_samples(monkeypatch):
    class OneWindow:
        rows = pd.DataFrame([{"split": "train", "eeg_path": "unused", "eeg_label_offset_seconds": 0}])
        task = "forecast"
        cut = 5
        max_missing_fraction = .05

        def __len__(self):
            return 1

    x = np.tile(np.sin(np.arange(10000) / 13), (len(ELECTRODES), 1)).astype(np.float32)
    x[:, 995:1000] = np.nan
    def fitted(future_value):
        data = x.copy()
        data[:, 1000:] = future_value
        monkeypatch.setattr(dataset_module, "read_eeg_window",
                            lambda path, offset: (data, np.isfinite(data)))
        return dataset_module.fit_graph_normalizer(OneWindow())
    first = fitted(0.0)
    second = fitted(100.0)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_parallel_quality_audit_matches_serial(tmp_path):
    path = tmp_path / "eeg.parquet"
    samples = np.arange(10000, dtype=np.float32)
    pd.DataFrame({name: np.sin(samples / (13 + i)) for i, name in enumerate(ELECTRODES)}).to_parquet(path)
    manifest = pd.DataFrame({"eeg_path": [str(path), str(path)],
                             "eeg_label_offset_seconds": [0, 1],
                             "audit_reason": ["ok", "ok"]})
    serial = audit_window_quality(manifest, workers=1)
    parallel = audit_window_quality(manifest, workers=2)
    pd.testing.assert_frame_equal(serial, parallel)
    assert serial.audit_reason.tolist() == ["ok", "window_out_of_bounds"]
