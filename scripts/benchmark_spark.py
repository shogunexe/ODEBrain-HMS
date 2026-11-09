"""Reproducible serial / local multiprocessing / local Spark feature benchmark.

Fractions select nested sets of *recordings* by stable hash. Each timing reads
raw EEG, computes identical window features, and writes output. All workers on
one host are explicitly reported as such. Synthetic inputs are smoke only.
"""

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

from src.spark_jobs.features import (build_spark_features, normalize_rows,
                                     process_recording, read_manifest, write_json)


def recording_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["eeg_path"]].append(row)
    return groups


def stable_subsets(rows, fractions):
    groups = recording_groups(rows)
    paths = sorted(groups, key=lambda p: hashlib.sha256(p.encode()).hexdigest())
    for fraction in sorted(set(fractions)):
        chosen = paths[:max(1, round(len(paths) * fraction))]
        yield fraction, [row for path in chosen for row in groups[path]]


def unique_duration(rows, seconds):
    duration = 0.0
    for group in recording_groups(rows).values():
        intervals = sorted((r["eeg_label_offset_seconds"],
                            r["eeg_label_offset_seconds"] + seconds) for r in group)
        if not intervals:
            continue
        lo, hi = intervals[0]
        for a, b in intervals[1:]:
            if a > hi:
                duration += hi - lo
                lo, hi = a, b
            else:
                hi = max(hi, b)
        duration += hi - lo
    return duration


def subset_stats(rows, seconds):
    paths = set(r["eeg_path"] for r in rows)
    return {"files": len(paths), "windows": len(rows),
            "raw_bytes": sum(os.path.getsize(p) for p in paths if os.path.isfile(p)),
            "unique_seconds": unique_duration(rows, seconds)}


def _work_group(item):
    path, rows, seconds = item
    return list(process_recording(path, rows, seconds))


def run_python(rows, output, seconds, workers):
    groups = recording_groups(rows)
    items = [(path, group, seconds) for path, group in groups.items()]
    valid = 0
    with output.open("w") as handle:
        def consume(records):
            nonlocal valid
            for record in records:
                valid += int(record["quality_ok"])
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        if workers == 1:
            for item in items:
                consume(_work_group(item))
        else:
            with mp.Pool(processes=workers) as pool:
                for records in pool.imap_unordered(_work_group, items):
                    consume(records)
    return valid


def write_manifest(path, rows):
    fields = ("window_id", "eeg_id", "patient_id", "eeg_path", "split",
              "label_id", "class_id", "eeg_label_offset_seconds", "sample_rate")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.3, 1.0])
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=50)
    parser.add_argument("--methods", default="serial,multiprocessing,spark")
    parser.add_argument("--keep-outputs", action="store_true")
    parser.add_argument("--synthetic", action="store_true",
                        help="Mark output as synthetic smoke, never formal evidence")
    parser.add_argument("--experiment-status", choices=("formal", "real_data_smoke", "synthetic_smoke"),
                        default="real_data_smoke")
    args = parser.parse_args()
    if args.experiment_status == "formal":
        from src.cli import require_full_data
        require_full_data(args.manifest)
    if not all(0 < x <= 1 for x in args.fractions) or args.repeats < 1:
        parser.error("fractions must be (0,1] and repeats positive")
    if not all(x >= 1 for x in args.workers):
        parser.error("workers must be positive")
    methods = set(args.methods.split(","))
    if methods - {"serial", "multiprocessing", "spark"}:
        parser.error("unknown method")
    rows = read_manifest(args.manifest)
    args.synthetic = args.synthetic or any(r["eeg_path"].endswith(".npz") for r in rows)
    if args.synthetic:
        if args.experiment_status == "formal":
            parser.error("synthetic inputs cannot be formal benchmark evidence")
        args.experiment_status = "synthetic_smoke"
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    run_spec = {"manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                "fractions": args.fractions, "workers": args.workers,
                "repeats": args.repeats, "seconds": args.seconds,
                "methods": sorted(methods), "synthetic": args.synthetic,
                "experiment_status": args.experiment_status}
    spec_path = root / "run_spec.json"
    if spec_path.is_file():
        if json.loads(spec_path.read_text(encoding="utf-8")) != run_spec:
            raise ValueError("Existing benchmark uses a different manifest or run specification")
    else:
        if (root / "runs.json").exists():
            raise ValueError("Existing benchmark has runs but no run specification")
        write_json(spec_path, run_spec)
    runs_path = root / "runs.json"
    runs = json.loads(runs_path.read_text(encoding="utf-8")) if runs_path.is_file() else []
    completed = {(row["subset"], row["method"], row["workers"], row["repeat"])
                 for row in runs}
    if len(completed) != len(runs):
        raise ValueError("Benchmark has duplicate recorded runs")
    for fraction, subset in stable_subsets(rows, args.fractions):
        stats = subset_stats(subset, args.seconds)
        name = "subset_%03d" % round(fraction * 100)
        subset_path = root / (name + ".csv")
        write_manifest(subset_path, subset)
        configurations = []
        if "serial" in methods:
            configurations.append(("serial", 1))
        if "multiprocessing" in methods:
            configurations += [("multiprocessing", w) for w in args.workers if w > 1]
        if "spark" in methods:
            configurations += [("spark", w) for w in args.workers]
        for method, workers in configurations:
            for repeat in range(args.repeats):
                run_key = (name, method, workers, repeat)
                if run_key in completed:
                    continue
                dest = root / ("%s_%s_%d_r%d" % (name, method, workers, repeat))
                if method == "spark" and dest.exists():
                    shutil.rmtree(dest)
                startup = 0.0
                spark = None
                if method == "spark":
                    from pyspark.sql import SparkSession
                    started = time.monotonic()
                    event_dir = root / "spark_event_logs"
                    event_dir.mkdir(exist_ok=True)
                    spark = (SparkSession.builder.appName("ODEBrain-Benchmark-%s-%d" % (name, workers))
                             .master("local[%d]" % workers)
                             .config("spark.eventLog.enabled", "true")
                             .config("spark.eventLog.dir", event_dir.resolve().as_uri())
                             .getOrCreate())
                    spark.sparkContext.setLogLevel("ERROR")
                    startup = time.monotonic() - started
                started = time.monotonic()
                try:
                    if method == "spark":
                        result = build_spark_features(spark, str(subset_path), str(dest),
                                                      partitions=workers, seconds=args.seconds)
                        valid = sum(v["valid"] for v in result["by_split"].values())
                    else:
                        valid = run_python(read_manifest(str(subset_path)), dest.with_suffix(".jsonl"),
                                           args.seconds, workers)
                    elapsed = time.monotonic() - started
                finally:
                    if spark is not None:
                        spark.stop()
                record = {"method": method, "workers": workers, "repeat": repeat,
                          "fraction": fraction, "subset": name, **stats,
                          "valid_windows": valid, "compute_and_write_seconds": elapsed,
                          "spark_session_setup_seconds": startup,
                          "end_to_end_seconds": elapsed + startup,
                          "windows_per_second": len(subset) / max(elapsed, 1e-9),
                          "cache_state": "uncontrolled; run order recorded",
                          "physical_hosts": 1, "shared_disk": True,
                          "cpu_slots": workers, "memory_limit_bytes": None,
                          "timing_boundary": "same Python process; Spark JVM may be reused across runs",
                          "host_cpu_count": os.cpu_count(),
                          "spark_version": result["spark_version"] if method == "spark" else None,
                          "synthetic": args.synthetic, "experiment_status": args.experiment_status}
                runs.append(record)
                write_json(root / "runs.json", runs)
                completed.add(run_key)
                print(json.dumps(record))
                if not args.keep_outputs:
                    if method == "spark":
                        shutil.rmtree(dest)
                    else:
                        dest.with_suffix(".jsonl").unlink()
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["subset"], run["method"], run["workers"])].append(
            run["end_to_end_seconds"])
    summary = [{"subset": k[0], "method": k[1], "workers": k[2],
                "median_end_to_end_seconds": statistics.median(values),
                "min_seconds": min(values), "max_seconds": max(values),
                "repeats": len(values), "synthetic": args.synthetic,
                "experiment_status": args.experiment_status}
               for k, values in grouped.items()]
    write_json(root / "summary.json", summary)


if __name__ == "__main__":
    main()
