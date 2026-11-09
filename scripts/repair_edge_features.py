"""Repair only formerly invalid HMS windows with common edge NaN padding.

The uniform feature extractor is used for corrections. Existing valid rows are
copied byte-for-byte at the Arrow value level, since the new extractor's edge
trim branch cannot activate on a row that passed the previous quality rule.
The script verifies every audited valid window before swapping the dataset.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.spark_jobs.features import read_manifest, serial_features


def read_partition(path):
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise ValueError("empty feature partition: %s" % path)
    # Read physical Parquet schema; pq.read_table(path) also infers the
    # split=... directory as a virtual dictionary column.
    return pa.concat_tables([pq.ParquetFile(str(file)).read() for file in files])


def ids(table):
    return table.column("window_id").to_pylist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/curated/split_manifest.csv")
    parser.add_argument("--features", default="data/features/eeg_features_v1")
    parser.add_argument("--summary", default="data/features/eeg_features_v1_summary.json")
    parser.add_argument("--backup-root", default="artifacts/feature_repair_backup_20261109")
    args = parser.parse_args()
    source = Path(args.features)
    staging = source.with_name(source.name + "_repaired_tmp")
    backup = Path(args.backup_root)
    if not source.is_dir() or staging.exists() or backup.exists():
        raise FileExistsError("expected existing source and absent staging/backup")
    old = {split: read_partition(source / ("split=" + split))
           for split in ("train", "val", "test")}
    bad = {split: {row["window_id"] for row in table.filter(
        pc.invert(table.column("quality_ok"))).select(["window_id"]).to_pylist()}
        for split, table in old.items()}
    bad_ids = set().union(*bad.values())
    if not bad_ids:
        raise ValueError("no invalid rows to repair")
    manifest = read_manifest(args.manifest)
    manifest_by_split = {split: {r["window_id"] for r in manifest if r["split"] == split}
                         for split in old}
    if any(set(ids(old[split])) != manifest_by_split[split] for split in old):
        raise ValueError("source feature IDs differ from audited manifest")
    selected = [r for r in manifest if r["window_id"] in bad_ids]
    if len(selected) != len(bad_ids):
        raise ValueError("invalid IDs not unique in manifest")
    corrections = list(serial_features(selected))
    if len(corrections) != len(bad_ids) or not all(r["quality_ok"] for r in corrections):
        raise ValueError("some corrected windows still fail quality")
    if any(r["error_reason"] is not None for r in corrections):
        raise ValueError("correction includes read/extraction error")
    staging.mkdir(parents=True)
    updated = {}
    try:
        for split, table in old.items():
            wanted = bad[split]
            kept = table.filter(pc.invert(pc.is_in(table.column("window_id"),
                                                 pa.array(sorted(wanted), type=pa.string()))))
            records = [{key: value for key, value in row.items() if key != "split"}
                       for row in corrections if row["split"] == split]
            replacement = pa.Table.from_pylist(records, schema=table.schema)
            combined = pa.concat_tables([kept, replacement])
            if len(combined) != len(table) or set(ids(combined)) != manifest_by_split[split]:
                raise ValueError("coverage mismatch in %s" % split)
            if len(ids(combined)) != len(set(ids(combined))):
                raise ValueError("duplicate window_id in %s" % split)
            if pc.sum(pc.invert(combined.column("quality_ok"))).as_py() != 0:
                raise ValueError("quality failure remains in %s" % split)
            path = staging / ("split=" + split)
            path.mkdir()
            pq.write_table(combined, str(path / "part-00000.parquet"), compression="snappy")
            updated[split] = {"windows": len(combined), "valid": len(combined),
                              "repaired": len(wanted)}
        (staging / "_SUCCESS").touch()
        # Reopen the written dataset before replacing the formal table.
        for split in old:
            written = read_partition(staging / ("split=" + split))
            if len(written) != updated[split]["windows"] or set(ids(written)) != manifest_by_split[split]:
                raise ValueError("written Parquet verification failed for %s" % split)
        previous_summary = json.loads(Path(args.summary).read_text())
        if sum(x["windows"] for x in updated.values()) != len(manifest):
            raise ValueError("formal window total differs from audited manifest")
        backup.mkdir(parents=True)
        (backup / "original_summary.json").write_text(json.dumps(previous_summary, indent=2))
        shutil.move(str(source), str(backup / "features_v1_before_edge_repair"))
        try:
            shutil.move(str(staging), str(source))
        except Exception:
            shutil.move(str(backup / "features_v1_before_edge_repair"), str(source))
            raise
        summary = dict(previous_summary)
        summary["by_split"] = {split: {"windows": x["windows"], "valid": x["valid"]}
                               for split, x in updated.items()}
        summary["repair"] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "reason": "common edge NaN padding in 21 windows; no long-gap interpolation",
            "algorithm": "shared extract_features; common time trim only if >=12 edge-defective electrodes",
            "repaired_rows": sum(x["repaired"] for x in updated.values()),
            "per_split": updated,
            "backup": str(backup / "features_v1_before_edge_repair"),
            "audited_valid_windows": len(manifest),
        }
        Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(json.dumps(summary["repair"], indent=2))
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()
