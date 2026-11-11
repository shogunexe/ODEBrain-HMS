"""Dashboard adapters retain Spark and ablation provenance."""

import json

import pandas as pd

from app import main as dashboard


def _fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/curated").mkdir(parents=True)
    (tmp_path / "data/features").mkdir(parents=True)
    (tmp_path / "results/spark_baselines").mkdir(parents=True)
    for name in ("lr", "rf"):
        (tmp_path / ("results/spark_baselines/" + name + "_model")).mkdir()
    table = pd.DataFrame([
        {"window_id": "row_1", "eeg_id": 11, "patient_id": 1,
         "eeg_label_offset_seconds": 0.0, "split": "train", "audit_reason": "ok", "file_exists": True},
        {"window_id": "row_2", "eeg_id": 22, "patient_id": 2,
         "eeg_label_offset_seconds": 0.0, "split": "val", "audit_reason": "ok", "file_exists": True},
        {"window_id": "row_3", "eeg_id": 33, "patient_id": 3,
         "eeg_label_offset_seconds": 0.0, "split": "test", "audit_reason": "ok", "file_exists": True},
    ])
    table.to_csv(tmp_path / "data/curated/split_manifest.csv", index=False)
    table[["eeg_id"]].to_csv(tmp_path / "data/raw/train.csv", index=False)
    summary = {"manifest": "data/curated/split_manifest.csv",
               "output": "data/features/eeg_features_v1", "experiment_status": "formal",
               "by_split": {split: {"valid": 1} for split in ("train", "val", "test")}}
    (tmp_path / "data/features/eeg_features_v1_summary.json").write_text(json.dumps(summary))
    confusion = [[1 if row == col == 0 else 0 for col in range(6)] for row in range(6)]
    models = {name: {"model_path": "results/spark_baselines/" + name + "_model",
                     "splits": {split: {"n": 1, "kl_window_mean": 0.3,
                                        "macro_f1": 0.5, "confusion": confusion}
                                for split in ("train", "val", "test")}}
              for name in ("lr", "rf")}
    spark = {"features_path": "data/features/eeg_features_v1",
             "experiment_status": "formal", "target_source": "expert_votes",
             "label_order": list(dashboard.CLASSES),
             "counts": {split: 1 for split in ("train", "val", "test")},
             "models": models}
    path = tmp_path / "results/spark_baselines/metrics.json"
    path.write_text(json.dumps(spark))
    return path, spark, dashboard.manifest_hash(table)


def test_spark_nested_metrics_are_visible_and_formal_only_when_verified(tmp_path, monkeypatch):
    path, spark, _ = _fixture(tmp_path, monkeypatch)
    rows = dashboard._metric_rows(tmp_path / "results")
    test_rows = [row for row in rows if row["split"] == "test"]
    assert {row["model"] for row in test_rows} == {"Spark LR", "Spark RF"}
    assert all(row["provenance"] == "formal" and row["windows"] == 1
               for row in test_rows)
    spark["experiment_status"] = "real_data_smoke"
    path.write_text(json.dumps(spark))
    assert {row["provenance"] for row in dashboard._spark_metric_rows(path, spark)} == {
        "real_hms_smoke"}
    spark["experiment_status"] = "formal"
    spark["counts"]["test"] = 2
    assert next(row for row in dashboard._spark_metric_rows(path, spark)
                if row["split"] == "test")["provenance"] == "unverified"


def test_ablation_json_respects_split_and_provenance(tmp_path, monkeypatch):
    _, _, split_hash = _fixture(tmp_path, monkeypatch)
    path = tmp_path / "results/ablation.json"
    result = {"schema_version": "ode_ablation_formal_v1", "experiment_status": "formal",
              "synthetic": False, "smoke_subset": False, "split": "test",
              "split_hash": split_hash, "n_windows": 1, "n_patients": 1,
              "rows": [{"name": "pretrained", "kl_window_mean": 0.5}]}
    path.write_text(json.dumps(result))
    frame = dashboard._ablation_rows(path, split_hash, "test", 1, 1)
    assert frame.iloc[0]["证据状态"] == "formal"
    assert dashboard._ablation_rows(path, "other", "test", 1, 1) is None
    result["smoke_subset"] = True
    path.write_text(json.dumps(result))
    assert dashboard._ablation_rows(path, split_hash, "test", 1, 1).iloc[0]["证据状态"] == "real_hms_smoke"


def test_performance_curve_keeps_methods_and_subsets_separate():
    rows = pd.DataFrame([
        {"method": "spark", "subset": "10%", "workers": 1, "end_to_end_seconds": 30},
        {"method": "spark", "subset": "10%", "workers": 4, "end_to_end_seconds": 12},
        {"method": "spark", "subset": "100%", "workers": 1, "end_to_end_seconds": 300},
        {"method": "spark", "subset": "100%", "workers": 4, "end_to_end_seconds": 120},
        {"method": "serial", "subset": "10%", "workers": 1, "end_to_end_seconds": 35},
    ])
    figure = dashboard._performance_figure(rows, "workers", "end_to_end_seconds")
    traces = {trace.name: trace for trace in figure.data}
    assert len(traces) == 3
    assert list(traces["method=spark · subset=10%"].x) == [1, 4]
    assert list(traces["method=spark · subset=10%"].y) == [30, 12]
    assert list(traces["method=spark · subset=100%"].y) == [300, 120]
    assert traces["method=serial · subset=10%"].mode == "markers"
