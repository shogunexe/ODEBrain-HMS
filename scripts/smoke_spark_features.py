"""Synthetic interface smoke; these numbers are never scientific results."""

import csv
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

from src.features.eeg import CHANNELS, feature_names
from src.spark_jobs.features import build_spark_features, serial_features


def main():
    with tempfile.TemporaryDirectory(prefix="odebrain_synthetic_spark_") as tmp:
        root = Path(tmp)
        rate = 200
        t = np.arange(55 * rate) / rate
        rng = np.random.RandomState(42)
        rows = []
        for file_index in range(2):
            signals = np.stack([np.sin(2 * np.pi * (7 + i % 8) * t + i) +
                                0.02 * rng.randn(len(t)) for i in range(len(CHANNELS))], axis=1)
            path = root / ("synthetic_%d.npz" % file_index)
            np.savez_compressed(str(path), signals=signals, channels=np.array(CHANNELS))
            for window_index, offset in enumerate((0, 5)):
                rows.append({"window_id": "synthetic_%d_%d" % (file_index, window_index),
                             "eeg_id": "synthetic_%d" % file_index,
                             "patient_id": "synthetic_patient_%d" % file_index,
                             "eeg_path": str(path), "split": "train" if file_index == 0 else "val",
                             "label_id": "synthetic_label_%d_%d" % (file_index, window_index),
                             "class_id": file_index, "eeg_label_offset_seconds": offset,
                             "sample_rate": rate, "audit_reason": "ok"})
        rows.append({"window_id": "synthetic_excluded", "eeg_id": "synthetic_0",
                     "patient_id": "synthetic_patient_0", "eeg_path": str(root / "synthetic_0.npz"),
                     "split": "excluded", "label_id": "synthetic_bad_label",
                     "class_id": "", "eeg_label_offset_seconds": 0,
                     "sample_rate": rate, "audit_reason": "invalid_votes"})
        manifest = root / "synthetic_manifest.csv"
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        reference = {r["window_id"]: r for r in serial_features(rows)}
        assert all(r["quality_ok"] for r in reference.values())
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.master("local[2]").appName("ODEBrain-Synthetic-Features-Smoke").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        try:
            result = build_spark_features(spark, str(manifest), str(root / "features"), partitions=2)
            actual = {r["window_id"]: r.asDict() for r in spark.read.parquet(str(root / "features")).collect()}
        finally:
            spark.stop()
        assert set(actual) == set(reference)
        for key in reference:
            for name in feature_names():
                a, b = actual[key][name], reference[key][name]
                assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (key, name, a, b)
            assert actual[key]["class_id"] == reference[key]["class_id"]
            assert actual[key]["label_id"] == reference[key]["label_id"]
        print("PASS synthetic Spark/serial agreement: %d windows, %d numeric features" %
              (len(actual), len(feature_names())))
        print("Synthetic smoke summary:", result)


if __name__ == "__main__":
    main()
