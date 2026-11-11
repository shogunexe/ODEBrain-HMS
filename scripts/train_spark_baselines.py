"""Fit MLlib LR/RF on versioned features, using only train split for fitting."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

from src.spark_jobs.baselines import train_baselines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--master", default="local[2]")
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--lr-max-iter", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--experiment-status", default="unverified",
                        help="For example real_data_smoke or formal")
    parser.add_argument("--manifest", default="data/curated/split_manifest.csv")
    args = parser.parse_args()
    expected_windows = None
    if args.experiment_status == "formal":
        import pandas as pd
        from src.cli import require_full_data
        require_full_data(args.manifest)
        expected_windows = int((pd.read_csv(args.manifest, usecols=["audit_reason"]).audit_reason == "ok").sum())
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("ODEBrain-HMS-Spark-Baselines").master(args.master).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        result = train_baselines(spark, args.features, args.output,
            rf_trees=args.rf_trees, lr_max_iter=args.lr_max_iter,
            overwrite=args.overwrite, experiment_status=args.experiment_status,
            expected_windows=expected_windows)
        print(json.dumps(result["models"], indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
