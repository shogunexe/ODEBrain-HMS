#!/usr/bin/env python3
"""Recompute three weighted classification views for frozen ablation arms.

This post hoc audit never trains, evaluates a checkpoint, or selects an arm.
It reads only the formal ablation summary and its already frozen test CSVs.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.run_ablations import load_plan, sha256
from scripts.supplement_classification_metrics import checked_arrays, group_weights, weighted_view
from src.data.hms import manifest_hash


def _source_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _matching(value, expected, description):
    if not math.isclose(float(value), float(expected), rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError(description + " differs from frozen ablation result")


def supplement(ablation_path, output_path):
    ablation_path, output_path = Path(ablation_path).resolve(), Path(output_path).resolve()
    source = json.loads(ablation_path.read_text(encoding="utf-8"))
    if (source.get("schema_version") != "ode_ablation_formal_v1"
            or source.get("experiment_status") != "formal" or source.get("split") != "test"
            or source.get("synthetic") is not False or source.get("smoke_subset") is not False):
        raise ValueError("Source is not the formal full test ablation")
    plan_path = _source_path(source["plan"])
    if sha256(plan_path) != source.get("plan_sha256"):
        raise ValueError("Ablation plan has changed")
    _, plan = load_plan(plan_path)
    arm_names = [arm["name"] for arm in plan["arms"]]
    rows = source.get("rows", [])
    if (not isinstance(rows, list) or len(rows) != len(arm_names)
            or {row.get("name") for row in rows} != set(arm_names)):
        raise ValueError("Ablation rows differ from the declared plan")
    arm_by_name = {arm["name"]: arm for arm in plan["arms"]}
    for row in rows:
        arm = arm_by_name[row["name"]]
        if (row.get("initialization") != arm["initialization"]
                or row.get("seed") != plan["seed"]
                or not math.isclose(float(row.get("rk4_step_seconds")),
                                    float(arm["rk4_step_seconds"]), rel_tol=0, abs_tol=1e-12)):
            raise ValueError("Ablation row setup differs from declared plan: " + row["name"])
    manifest_path = _source_path(source["manifest"])
    manifest = pd.read_csv(manifest_path)
    test = manifest.loc[(manifest.split == "test") & (manifest.audit_reason == "ok")].copy()
    if (test.empty or test.window_id.astype(str).duplicated().any()
            or sha256(manifest_path) != source.get("manifest_sha256")
            or manifest_hash(manifest) != source.get("split_hash")
            or len(test) != source.get("n_windows")
            or int(test.patient_id.nunique()) != source.get("n_patients")):
        raise ValueError("Ablation manifest or test coverage differs from frozen summary")
    result = {
        "schema_version": "ode_ablation_weighted_metrics_v1",
        "experiment_status": "formal", "split": "test", "test_used_for_selection": False,
        "interpretation": "Each aggregation scores the same frozen test windows; group-equal views weight each window by inverse windows in its patient or EEG recording. There is no group-level vote or classification. These are descriptive ablation metrics and are not directly subtracted from train+val refit metrics.",
        "hard_label_source": "expert_consensus_class_id",
        "kl_target_source": "expert_vote_distribution",
        "manifest": str(manifest_path), "manifest_sha256": source["manifest_sha256"],
        "split_hash": source["split_hash"], "n_windows": len(test),
        "n_patients": int(test.patient_id.nunique()),
        "n_eeg_records": int(test.eeg_id.nunique()),
        "ablation_summary": str(ablation_path), "ablation_summary_sha256": sha256(ablation_path),
        "plan": str(plan_path), "plan_sha256": source["plan_sha256"],
        "rows": [],
    }
    for row in rows:
        name = row["name"]
        predictions_path = _source_path(row["predictions_file"])
        metrics_path = _source_path(row["metrics_file"])
        if (sha256(predictions_path) != row.get("predictions_sha256")
                or sha256(metrics_path) != row.get("metrics_sha256")):
            raise ValueError("Frozen prediction or metric file changed: " + name)
        metric = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (metric.get("split") != "test" or metric.get("split_hash") != source["split_hash"]
                or metric.get("n_windows") != len(test)):
            raise ValueError("Ablation metric provenance differs: " + name)
        predictions = pd.read_csv(predictions_path, dtype={"window_id": str, "patient_id": str})
        targets, scores, hard, patients, records = checked_arrays(predictions, test, "deep")
        if (len(np.unique(patients)) != result["n_patients"]
                or len(np.unique(records)) != result["n_eeg_records"]):
            raise ValueError("Ablation patient or EEG coverage differs: " + name)
        views = {
            "window_equal": weighted_view(targets, scores, hard, np.ones(len(hard))),
            "patient_equal": weighted_view(targets, scores, hard, group_weights(patients)),
            "eeg_record_equal": weighted_view(targets, scores, hard, group_weights(records)),
        }
        for view, field in (("window_equal", "kl_window_mean"),
                            ("patient_equal", "kl_patient_equal_mean")):
            _matching(views[view]["kl_mean"], row[field], name + " " + field)
            _matching(views[view]["kl_mean"], metric[field], name + " metric " + field)
        result["rows"].append({
            "name": name, "initialization": row["initialization"],
            "rk4_step_seconds": row["rk4_step_seconds"], "seed": row["seed"],
            "training_protocol": "train_only_validation_early_stopping",
            "predictions_file": str(predictions_path), "predictions_sha256": row["predictions_sha256"],
            "metrics_file": str(metrics_path), "metrics_sha256": row["metrics_sha256"],
            "aggregations": views,
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, output_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", default="results/ablation.json")
    parser.add_argument("--output", default="results/ablation_weighted_metrics.json")
    args = parser.parse_args()
    result = supplement(args.ablation, args.output)
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "rows": [{"name": row["name"],
                                "kl_eeg_record_equal_mean": row["aggregations"]["eeg_record_equal"]["kl_mean"]}
                               for row in result["rows"]],
                      "n_windows": result["n_windows"], "n_eeg_records": result["n_eeg_records"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
