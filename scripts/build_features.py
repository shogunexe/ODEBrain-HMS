"""Build versioned EEG feature Parquet from a patient split manifest.

Example: spark-submit scripts/build_features.py --manifest artifacts/curated/manifest.csv
    --output artifacts/features/eeg_features_v1 --master local[2]
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

from src.spark_jobs.features import build_spark_features, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Optional YAML/JSON with manifest, output, master, partitions")
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--master", default=None)
    parser.add_argument("--partitions", type=int)
    parser.add_argument("--seconds", type=float, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary")
    parser.add_argument("--experiment-status", default="unverified")
    parser.add_argument("--event-log-dir")
    args = parser.parse_args()
    config = {}
    if args.config:
        source = Path(args.config)
        if source.suffix in (".yaml", ".yml"):
            import yaml
            config = yaml.safe_load(source.read_text()) or {}
        else:
            config = json.loads(source.read_text())
        config = config.get("spark_features", config)
    manifest = args.manifest or config.get("manifest")
    output = args.output or config.get("output")
    if not manifest or not output:
        parser.error("--manifest and --output (or matching config keys) required")
    if args.experiment_status == "formal":
        from src.cli import require_full_data
        require_full_data(manifest)
    from pyspark.sql import SparkSession
    builder = SparkSession.builder.appName("ODEBrain-HMS-Features")
    master = args.master or config.get("master")
    if master:
        builder = builder.master(master)
    event_dir = args.event_log_dir or config.get("event_log_dir")
    if event_dir:
        Path(event_dir).mkdir(parents=True, exist_ok=True)
        builder = builder.config("spark.eventLog.enabled", "true").config(
            "spark.eventLog.dir", Path(event_dir).resolve().as_uri())
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        result = build_spark_features(spark, manifest, output,
            partitions=args.partitions or config.get("partitions"),
            seconds=args.seconds, mode="overwrite" if args.overwrite else "error")
        result["master"] = spark.sparkContext.master
        result["experiment_status"] = args.experiment_status
        result["event_log_dir"] = event_dir
        summary = args.summary or str(Path(output).with_name(Path(output).name + "_summary.json"))
        write_json(summary, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
