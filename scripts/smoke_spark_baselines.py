"""Synthetic six-class Spark ML smoke; no scientific metric is inferred."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

from src.features.eeg import feature_names
from src.spark_jobs.baselines import train_baselines


def main():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.master("local[2]").appName("ODEBrain-Synthetic-Baseline-Smoke").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        with tempfile.TemporaryDirectory(prefix="odebrain_synthetic_baselines_") as tmp:
            root = Path(tmp)
            names = feature_names()
            rows = []
            for split, count in (("train", 3), ("val", 1), ("test", 1)):
                for cls in range(6):
                    for j in range(count):
                        row = {"window_id": "synthetic_%s_%d_%d" % (split, cls, j),
                               "eeg_id": "synthetic_%s_%d" % (split, cls),
                               "patient_id": "synthetic_patient_%s_%d" % (split, cls),
                               "label_id": "synthetic_label_%s_%d_%d" % (split, cls, j),
                               "split": split, "class_id": cls, "quality_ok": True}
                        row.update({name: float(cls * (i % 5 + 1) + j * 0.01 + i * 0.001)
                                    for i, name in enumerate(names)})
                        rows.append(row)
            feature_path = root / "features"
            spark.createDataFrame(rows).write.parquet(str(feature_path))
            result = train_baselines(spark, str(feature_path), str(root / "models"),
                                     rf_trees=3, lr_max_iter=5)
            for kind in ("lr", "rf"):
                pred = spark.read.parquet(str(root / "models" / (kind + "_test_predictions")))
                assert pred.count() == 6
                assert all(len(r["probabilities"]) == 6 for r in pred.select("probabilities").collect())
                assert result["models"][kind]["splits"]["test"]["n"] == 6
            print("PASS synthetic MLlib LR/RF six-class schema and prediction smoke")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
