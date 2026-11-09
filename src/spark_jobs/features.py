"""Partitioned EEG feature job and matching serial implementation."""

from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

import numpy as np

from src.features.eeg import SCHEMA_VERSION, extract_features, feature_names


IDENTIFIERS = ("window_id", "eeg_id", "patient_id", "eeg_path", "split", "label_id")
TARGET_COLS = ("target_seizure", "target_lpd", "target_gpd", "target_lrda", "target_grda", "target_other")


def read_manifest(path):
    """Read a small manifest for serial benchmarks; Spark reads it distributed."""
    import pandas as pd
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in (".jsonl", ".json"):
        frame = pd.read_json(path, lines=suffix == ".jsonl")
    elif suffix == ".csv":
        frame = pd.read_csv(path, dtype={key: "string" for key in
            ("eeg_id", "patient_id", "label_id", "window_id", "eeg_path", "split", "audit_reason")})
    else:
        raise ValueError("manifest must be CSV, JSONL, JSON, or Parquet")
    return normalize_rows(frame.to_dict("records"))


def normalize_rows(rows):
    normalized = []
    required = ("eeg_path", "eeg_id", "patient_id", "eeg_label_offset_seconds",
                "split", "class_id")
    for i, row in enumerate(rows):
        if str(row.get("split", "")).lower() not in ("train", "val", "test"):
            continue
        if "audit_reason" in row and str(row["audit_reason"]).lower() != "ok":
            continue
        for key in required:
            if key not in row or row[key] is None:
                raise ValueError("manifest row %d missing %s" % (i, key))
        r = dict(row)
        for key in IDENTIFIERS:
            if key in r and r[key] is not None:
                r[key] = str(r[key])
        r["eeg_path"] = os.path.abspath(str(r["eeg_path"]))
        r["eeg_label_offset_seconds"] = float(r["eeg_label_offset_seconds"])
        r["class_id"] = int(r["class_id"])
        r["sample_rate"] = int(r.get("sample_rate") or 200)
        for key in TARGET_COLS:
            if key in r and r[key] is not None:
                r[key] = float(r[key])
        if r["class_id"] not in range(6) or r["sample_rate"] <= 0:
            raise ValueError("invalid label/rate in manifest row %d" % i)
        if str(r.get("window_id")) in ("None", "<NA>", "nan", ""):
            r["window_id"] = "%s_%g" % (r["eeg_id"],
                                          r["eeg_label_offset_seconds"])
        normalized.append(r)
    return normalized


def read_eeg(path):
    """Load one raw recording, never one read per requested window."""
    suffix = Path(path).suffix.lower()
    if suffix == ".npz":  # synthetic smoke only
        with np.load(path, allow_pickle=False) as obj:
            samples = np.asarray(obj["signals"], dtype=np.float64)
            channels = [str(x) for x in obj["channels"].tolist()]
        return samples, channels
    if suffix == ".parquet":
        import pandas as pd
        frame = pd.read_parquet(path)
        return frame.to_numpy(dtype=np.float64, copy=False), list(frame.columns)
    raise ValueError("unsupported EEG format %s" % path)


def process_recording(path, rows, seconds=50.0):
    """Yield all window features from a single recording and one file read."""
    try:
        samples, channels = read_eeg(path)
        read_error = None
    except Exception as exc:
        samples, channels, read_error = None, None, "%s: %s" % (type(exc).__name__, exc)
    for row in rows:
        output = {key: row.get(key) for key in IDENTIFIERS}
        output.update({"eeg_label_offset_seconds": row["eeg_label_offset_seconds"],
                       "class_id": row["class_id"], "sample_rate": row["sample_rate"],
                       "schema_version": SCHEMA_VERSION, "error_reason": read_error})
        output.update({key: row.get(key) for key in TARGET_COLS})
        if read_error is None:
            start = int(round(row["eeg_label_offset_seconds"] * row["sample_rate"]))
            length = int(round(seconds * row["sample_rate"]))
            if start < 0 or start + length > len(samples):
                output["error_reason"] = "window_out_of_bounds"
            else:
                try:
                    output.update(extract_features(samples[start:start + length], channels,
                                                   row["sample_rate"]))
                except Exception as exc:
                    output["error_reason"] = "%s: %s" % (type(exc).__name__, exc)
        if output["error_reason"] is not None:
            output.update({name: None for name in feature_names()})
            output.update({"valid_channel_count": 0, "duration_seconds": 0.0,
                           "quality_ok": False})
        yield output


def serial_features(rows, seconds=50.0):
    """Matching non-parallel reference, grouped by file to avoid repeated IO."""
    rows = normalize_rows(rows)
    for path, group in itertools.groupby(sorted(rows, key=lambda r: r["eeg_path"]),
                                         key=lambda r: r["eeg_path"]):
        yield from process_recording(path, group, seconds)


def _partition_features(iterator, seconds):
    for path, group in itertools.groupby(iterator, key=lambda r: r["eeg_path"]):
        yield from process_recording(path, (r.asDict(recursive=True) for r in group), seconds)


def spark_schema():
    from pyspark.sql.types import (BooleanType, DoubleType, IntegerType, StringType,
                                   StructField, StructType)
    fields = [StructField(key, StringType()) for key in IDENTIFIERS]
    fields += [StructField("eeg_label_offset_seconds", DoubleType()),
               StructField("class_id", IntegerType()),
               StructField("sample_rate", IntegerType()),
               StructField("schema_version", StringType()),
               StructField("error_reason", StringType())]
    fields += [StructField(key, DoubleType()) for key in TARGET_COLS]
    fields += [StructField(name, DoubleType()) for name in feature_names()]
    fields += [StructField("valid_channel_count", IntegerType()),
               StructField("duration_seconds", DoubleType()),
               StructField("quality_ok", BooleanType())]
    return StructType(fields)


def load_spark_manifest(spark, path):
    from pyspark.sql import functions as F
    if path.endswith(".csv"):
        df = spark.read.option("header", True).csv(path)
    elif path.endswith(".json") or path.endswith(".jsonl"):
        df = spark.read.json(path)
    else:
        df = spark.read.parquet(path)
    required = ("eeg_path", "eeg_id", "patient_id", "eeg_label_offset_seconds",
                "split", "class_id")
    absent = set(required) - set(df.columns)
    if absent:
        raise ValueError("manifest missing: %s" % sorted(absent))
    if "sample_rate" not in df.columns:
        df = df.withColumn("sample_rate", F.lit(200))
    if "window_id" not in df.columns:
        df = df.withColumn("window_id", F.concat_ws("_", F.col("eeg_id"),
                                                   F.col("eeg_label_offset_seconds")))
    if "label_id" not in df.columns:
        df = df.withColumn("label_id", F.lit(None).cast("string"))
    for key in TARGET_COLS:
        if key not in df.columns:
            df = df.withColumn(key, F.lit(None).cast("double"))
    df = df.filter(F.lower(F.col("split")).isin("train", "val", "test"))
    if "audit_reason" in df.columns:
        df = df.filter(F.lower(F.col("audit_reason")) == "ok")
    for key in IDENTIFIERS:
        df = df.withColumn(key, F.col(key).cast("string"))
    return df.select(*[F.col(k) for k in IDENTIFIERS],
                     F.col("eeg_label_offset_seconds").cast("double").alias("eeg_label_offset_seconds"),
                     F.col("class_id").cast("int").alias("class_id"),
                     F.coalesce(F.col("sample_rate").cast("int"), F.lit(200)).alias("sample_rate"),
                     *[F.col(k).cast("double").alias(k) for k in TARGET_COLS])


def build_spark_features(spark, manifest_path, output_path, partitions=None,
                         seconds=50.0, mode="error"):
    """Distribute by recording, sort requests, and stream one file at a time."""
    from pyspark.sql import functions as F
    if mode not in ("error", "overwrite"):
        raise ValueError("mode must be error or overwrite")
    manifest = load_spark_manifest(spark, manifest_path)
    invalid = manifest.filter((F.col("eeg_path").isNull()) |
                              (F.col("class_id").isNull()) |
                              (F.col("eeg_label_offset_seconds").isNull()) |
                              (F.col("class_id") < 0) | (F.col("class_id") > 5) |
                              (F.col("sample_rate") <= 0)).limit(1).count()
    if invalid:
        raise ValueError("manifest has invalid path, label, or sample rate")
    n = partitions or spark.sparkContext.defaultParallelism
    grouped = manifest.repartition(n, "eeg_path").sortWithinPartitions("eeg_path")
    start = time.monotonic()
    result = spark.createDataFrame(grouped.rdd.mapPartitions(
        lambda it: _partition_features(it, seconds)), schema=spark_schema())
    result.write.mode(mode).partitionBy("split").parquet(output_path)
    elapsed = time.monotonic() - start
    table = spark.read.parquet(output_path)
    summary = {r["split"]: {"windows": r["windows"], "valid": r["valid"]}
               for r in table.groupBy("split").agg(F.count("*").alias("windows"),
                    F.sum(F.col("quality_ok").cast("int")).alias("valid")).collect()}
    return {"schema_version": SCHEMA_VERSION, "manifest": manifest_path,
            "output": output_path, "partitions": n, "seconds": seconds,
            "wall_seconds_including_write": elapsed, "by_split": summary,
            "spark_version": spark.version}


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
