#!/usr/bin/env python3
"""Prepare traceable PNG evidence for the 24-page formal PPTX.

The input bundle is documented in docs/slides_plan.md. This command is a
fail-closed packaging step, not an experiment runner. It never relabels a
small-subset result as formal and never invents an absent comparison.
"""

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAGES = (4, 5, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
DEEP_MODELS = ("tcn", "spectrogram_cnn", "fusion", "graph_gru", "odebrain_inspired_hms")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DYNAMICS_FIGURES = ("synchronized", "trajectory_direction", "dynamics_quantities",
                    "observed_correlation", "future_graph_nodes", "perturbation")
DYNAMICS_LAYOUT = {
    17: (("correct", "synchronized"), ("error", "trajectory_direction")),
    18: (("correct", "dynamics_quantities"), ("error", "perturbation")),
    19: (("correct", "observed_correlation"), ("error", "future_graph_nodes")),
}


class EvidenceError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise EvidenceError(message)


def path_from(value):
    need(isinstance(value, str) and value.strip(), "Missing source path")
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    path = path.resolve()
    need(path.is_file(), "Missing source file: {}".format(path))
    return path


def relative(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError("Invalid JSON {}: {}".format(path, exc))


def reject_smoke(obj, source):
    """Reject explicit smoke, synthetic and partial provenance at any depth."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("smoke_subset", "synthetic") and value is not False:
                raise EvidenceError("Smoke/synthetic evidence in {}: {}={}".format(source, key, value))
            if key in ("experiment_status", "provenance") and isinstance(value, str):
                if "smoke" in value.lower() or "synthetic" in value.lower():
                    raise EvidenceError("Smoke/synthetic evidence in {}: {}={}".format(source, key, value))
            reject_smoke(value, source)
    elif isinstance(obj, list):
        for item in obj:
            reject_smoke(item, source)


def formal_json(value, label, dependencies):
    path = path_from(value)
    obj = read_json(path)
    reject_smoke(obj, label)
    dependencies[relative(path)] = digest(path)
    return path, obj


def finite_number(value, label, minimum=None):
    need(isinstance(value, (int, float)) and not isinstance(value, bool)
         and math.isfinite(value), "{} must be finite".format(label))
    if minimum is not None:
        need(value >= minimum, "{} must be >= {}".format(label, minimum))
    return value


def positive_int(value, label):
    need(type(value) is int and value > 0, "{} must be a positive integer".format(label))
    return value


def same_hash(value, expected, label):
    need(isinstance(value, str) and HEX64.fullmatch(value) and value == expected,
         "{} split hash differs from formal manifest".format(label))


def _counts(manifest):
    return {name: (int((manifest.split == name).sum()),
                   int(manifest.loc[manifest.split == name, "patient_id"].nunique()))
            for name in ("train", "val", "test")}


def validate_manifest(bundle, dependencies):
    import pandas as pd
    from src.data.hms import manifest_hash
    manifest_path = path_from(bundle["manifest"])
    metadata_path = path_from(bundle["metadata"])
    manifest = pd.read_csv(manifest_path)
    metadata = pd.read_csv(metadata_path, usecols=["eeg_id"])
    required = {"window_id", "eeg_id", "patient_id", "eeg_label_offset_seconds", "split",
                "file_exists", "audit_reason", "missing_fraction", "raw_recording_seconds"}
    need(required.issubset(manifest.columns), "Formal manifest lacks audit/split columns")
    need(len(manifest) == len(metadata), "Manifest does not cover all train.csv rows")
    need(manifest.eeg_id.value_counts().sort_index().equals(metadata.eeg_id.value_counts().sort_index()),
         "Manifest does not cover all metadata EEG IDs")
    need(manifest.file_exists.fillna(False).all(), "Manifest contains missing EEG files")
    need(not manifest.audit_reason.astype(str).str.contains("missing_eeg_file|unreadable_eeg_file").any(),
         "Manifest contains missing or unreadable EEG files")
    valid = manifest.audit_reason == "ok"
    need(valid.any(), "No audited valid windows")
    need(not manifest.loc[valid, ["missing_fraction", "raw_recording_seconds"]].isna().any().any(),
         "Valid windows lack quality audit")
    need(manifest.loc[valid, "split"].isin(["train", "val", "test"]).all(),
         "Valid windows have unassigned split")
    need(manifest.loc[~valid, "split"].eq("excluded").all(),
         "Invalid windows must be excluded")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        left = set(manifest.loc[manifest.split == a, "patient_id"].astype(str))
        right = set(manifest.loc[manifest.split == b, "patient_id"].astype(str))
        need(not (left & right), "Patient leakage across {} and {}".format(a, b))
    split_hash = manifest_hash(manifest)
    dependencies[relative(manifest_path)] = digest(manifest_path)
    dependencies[relative(metadata_path)] = digest(metadata_path)
    audit_path, audit = formal_json(bundle["audit"], "audit", dependencies)
    need(audit.get("schema_version") == "hms_manifest_v1", "Wrong audit schema")
    need(audit.get("metadata_rows") == len(manifest), "Audit metadata row count differs")
    need(audit.get("valid_windows") == int(valid.sum()), "Audit valid window count differs")
    need(audit.get("raw_eeg_files_found") == manifest.eeg_id.nunique(),
         "Audit does not cover every EEG file")
    need(audit.get("quality_checked_windows", 0) >= int(valid.sum()),
         "Audit did not quality-check every valid window")
    split_path, split = formal_json(bundle["split_summary"], "split summary", dependencies)
    same_hash(split.get("sha256"), split_hash, "Split summary")
    counts = _counts(manifest)
    for name, (windows, patients) in counts.items():
        need(split.get("split_window_counts", {}).get(name) == windows,
             "Split summary window count differs for {}".format(name))
        need(split.get("split_patient_counts", {}).get(name) == patients,
             "Split summary patient count differs for {}".format(name))
        need(windows > 0 and patients > 0, "Empty formal {} split".format(name))
    return manifest_path, audit_path, split_path, audit, split_hash, counts


def validate_features(bundle, manifest_path, counts, dependencies):
    path, data = formal_json(bundle["feature_summary"], "Spark features", dependencies)
    need(data.get("experiment_status") == "formal", "Spark features must be formal")
    need(path_from(data.get("manifest")) == manifest_path, "Spark features use another manifest")
    need(data.get("spark_version"), "Spark version missing")
    for name, (windows, _) in counts.items():
        row = data.get("by_split", {}).get(name, {})
        need(row.get("valid") == windows, "Spark feature count differs for {}".format(name))
    finite_number(data.get("wall_seconds_including_write"), "Spark feature time", 0)
    return path, data


def validate_spark_models(bundle, counts, dependencies):
    path, data = formal_json(bundle["spark_metrics"], "MLlib LR/RF models", dependencies)
    need(data.get("experiment_status") == "formal", "MLlib LR/RF models must be formal")
    need(data.get("target_source") == "expert_votes", "Spark target must be expert votes")
    for name, (windows, _) in counts.items():
        need(data.get("counts", {}).get(name) == windows, "Spark count differs for {}".format(name))
    for name in ("lr", "rf"):
        row = data.get("models", {}).get(name, {})
        need(Path(row.get("model_path", "")).name == name + "_model", "Spark model missing: {}".format(name))
        model_dir = Path(row["model_path"])
        model_dir = model_dir if model_dir.is_absolute() else ROOT / model_dir
        need(model_dir.is_dir() and any(model_dir.rglob("part-*")), "Spark model directory missing: {}".format(model_dir))
        for split_name, (windows, _) in counts.items():
            metric = row.get("splits", {}).get(split_name, {})
            need(metric.get("n") == windows, "{} {} evaluation count differs".format(name, split_name))
            finite_number(metric.get("kl_window_mean"), "{} {} KL".format(name, split_name), 0)
    return path, data


def validate_spark_record_metrics(bundle, manifest_path, split_hash, counts, dependencies):
    """Recompute record-equal KL from frozen MLlib prediction Parquet."""
    from scripts.supplement_spark_record_metrics import supplement

    path, data = formal_json(bundle["spark_record_metrics"], "MLlib record-equal KL", dependencies)
    need(data.get("schema_version") == "hms_spark_eeg_record_equal_kl_v1" and
         data.get("experiment_status") == "formal" and data.get("split") == "test" and
         data.get("aggregation_unit") == "eeg_id" and
         data.get("test_used_for_selection") is False,
         "MLlib record-equal evidence has wrong protocol")
    same_hash(data.get("split_hash"), split_hash, "MLlib record-equal KL")
    need(data.get("manifest_sha256") == digest(manifest_path) and
         data.get("n_windows") == counts["test"][0] and
         data.get("n_patients") == counts["test"][1],
         "MLlib record-equal test coverage differs")
    spark_path = path_from(bundle["spark_metrics"])
    need(data.get("original_metrics_sha256") == digest(spark_path),
         "MLlib record-equal evidence uses another baseline result")
    try:
        with tempfile.TemporaryDirectory(prefix="formal-slides-spark-record-") as temporary:
            recomputed = supplement(manifest_path, spark_path,
                                    Path(temporary) / "eeg_record_equal_kl.json")
    except (OSError, ValueError, KeyError) as exc:
        raise EvidenceError("MLlib record-equal evidence failed revalidation: {}".format(exc))
    need(recomputed == data, "MLlib record-equal evidence differs from saved predictions")
    for model in ("lr", "rf"):
        row = data["models"][model]
        need(row.get("n_eeg_records") == data.get("n_eeg_records"),
             "MLlib record count differs: {}".format(model))
        finite_number(row.get("kl_eeg_record_equal_mean"), model + " record-equal KL", 0)
        for name, checksum in row["prediction_parts_sha256"].items():
            dependencies[relative(path_from(str(Path(row["predictions_path"]) / name)))] = checksum
    return path, data


def validate_weighted_classification(bundle, manifest_path, split_hash, counts, dependencies):
    """Recompute three weighting views and auxiliary classification metrics."""
    from scripts.supplement_classification_metrics import supplement

    path, data = formal_json(bundle["classification_weighted_metrics"],
                             "Weighted classification metrics", dependencies)
    need(data.get("schema_version") == "hms_classification_weighted_metrics_v1" and
         data.get("experiment_status") == "formal" and data.get("split") == "test" and
         data.get("test_used_for_selection") is False,
         "Weighted classification evidence has wrong protocol")
    same_hash(data.get("split_hash"), split_hash, "Weighted classification metrics")
    need(data.get("manifest_sha256") == digest(manifest_path) and
         data.get("n_windows") == counts["test"][0] and
         data.get("n_patients") == counts["test"][1],
         "Weighted classification test coverage differs")
    try:
        with tempfile.TemporaryDirectory(prefix="formal-slides-weighted-clf-") as temporary:
            regenerated = supplement(manifest_path, path_from(bundle["refit_comparison"]),
                                     path_from(bundle["spark_metrics"]),
                                     path_from(bundle["spark_record_metrics"]),
                                     Path(temporary) / "classification_weighted_metrics.json")
    except (OSError, ValueError, KeyError) as exc:
        raise EvidenceError("Weighted classification evidence failed revalidation: {}".format(exc))
    need(regenerated == data, "Weighted classification evidence differs from saved predictions")
    for row in data["models"].values():
        for name, checksum in row["prediction_sha256"].items():
            source = Path(row["predictions_path"])
            part = source if source.is_file() else source / name
            dependencies[relative(path_from(str(part)))] = checksum
    return path, data


def validate_deep(bundle, manifest_path, split_hash, counts, dependencies):
    """Recompute the five-model refit summary before accepting headline results."""
    from scripts.refit_final import summarize_refits

    need("deep_metrics" not in bundle,
         "Formal slides must use the refit comparison, not initial-run deep_metrics")
    path, comparison = formal_json(bundle["refit_comparison"], "Final refit comparison", dependencies)
    need(comparison.get("schema_version") == "hms_final_refit_comparison_v1" and
         comparison.get("experiment_status") == "formal" and
         comparison.get("training_protocol") == "train_val_fixed_epoch_refit" and
         comparison.get("split") == "test" and
         comparison.get("smoke_subset") is False and
         comparison.get("synthetic") is False and
         comparison.get("test_used_for_selection") is False,
         "Headline comparison is not a formal train+val fixed-epoch refit")
    same_hash(comparison.get("split_hash"), split_hash, "Final refit comparison")
    need(comparison.get("manifest_sha256") == digest(manifest_path),
         "Refit comparison uses another manifest")
    test_n, test_patients = counts["test"]
    need(comparison.get("n_windows") == test_n and comparison.get("n_patients") == test_patients,
         "Refit comparison does not cover the complete patient-disjoint test set")
    rows = comparison.get("rows")
    need(isinstance(rows, list) and len(rows) == len(DEEP_MODELS) and
         all(isinstance(row, dict) for row in rows) and
         {row.get("name") for row in rows} == set(DEEP_MODELS),
         "Refit comparison must contain exactly the five distinct model rows")
    run_dirs = {}
    for row in rows:
        name = row["name"]
        run_dir = Path(row.get("run_dir", ""))
        run_dir = (run_dir if run_dir.is_absolute() else ROOT / run_dir).resolve()
        try:
            run_dir.relative_to(ROOT.resolve())
            inside_project = True
        except ValueError:
            inside_project = False
        need(run_dir.is_dir() and inside_project,
             "Refit run is missing or outside the project: {}".format(name))
        run_dirs[name] = run_dir
    try:
        with tempfile.TemporaryDirectory(prefix="formal-slides-refit-") as temporary:
            regenerated_path = Path(temporary) / "final_comparison.json"
            summarize_refits(run_dirs, regenerated_path, results_root=path.parent)
            regenerated = read_json(regenerated_path)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise EvidenceError("Final refit evidence failed revalidation: {}".format(exc))
    need(regenerated == comparison,
         "Refit comparison differs from current checkpoints, predictions or metrics")
    output = {}
    for row in rows:
        name = row["name"]
        finite_number(row.get("kl_window_mean"), name + " refit KL", 0)
        run_dir = run_dirs[name]
        checkpoint = path_from(str(run_dir / "final.pt"))
        config = path_from(str(run_dir / "config.json"))
        configuration = read_json(config)
        need(configuration.get("refit", {}).get("training_split") == "train+val" and
             configuration.get("refit", {}).get("test_used_for_selection") is False,
             "{} is not a frozen train+val refit".format(name))
        metric_path = path_from(str(path.parent / run_dir.name / "test_metrics.json"))
        prediction_path = path_from(str(path.parent / run_dir.name / "test_predictions.csv"))
        need(digest(checkpoint) == row["checkpoint_sha256"] and
             digest(metric_path) == row["test_metrics_sha256"] and
             digest(prediction_path) == row["test_predictions_sha256"],
             "{} source file hash differs from refit summary".format(name))
        for source in (checkpoint, config, metric_path, prediction_path,
                       run_dir / "completed.json", run_dir / "validation_record.json"):
            dependencies[relative(source)] = digest(source)
        output[name] = (path, row, row["checkpoint_sha256"])
    return output


def validate_paired_refit(bundle, manifest_path, split_hash, counts, dependencies):
    """Recompute descriptive patient-cluster intervals from final test predictions."""
    from scripts.paired_refit_ci import analyze

    path, data = formal_json(bundle["paired_refit_ci"], "Final paired KL intervals", dependencies)
    need(data.get("schema_version") == "hms_paired_refit_kl_v1" and
         data.get("status") == "formal" and data.get("split") == "test" and
         data.get("training_protocol") == "train_val_fixed_epoch_refit" and
         data.get("test_used_for_selection") is False and
         data.get("test_used_for_model_or_configuration_selection") is False and
         data.get("method", {}).get("resampling_unit") == "patient" and
         data.get("method", {}).get("paired") is True,
         "Final paired intervals have wrong test protocol")
    same_hash(data.get("split_hash"), split_hash, "Final paired KL intervals")
    need(data.get("manifest_sha256") == digest(manifest_path) and
         data.get("n_windows") == counts["test"][0] and
         data.get("n_patients") == counts["test"][1] and
         data.get("final_comparison_sha256") == digest(path_from(bundle["refit_comparison"])),
         "Final paired intervals differ from frozen refit test evidence")
    try:
        regenerated = analyze(manifest_path, path_from(bundle["refit_comparison"]),
                              path_from(bundle["refit_comparison"]).parent,
                              n_boot=data["method"]["n_boot"], seed=data["method"]["seed"])
    except (OSError, ValueError, KeyError) as exc:
        raise EvidenceError("Final paired intervals failed revalidation: {}".format(exc))
    need(regenerated == data, "Final paired intervals differ from saved predictions")
    for row in data["runs"].values():
        prediction = path_from(row["predictions"])
        dependencies[relative(prediction)] = digest(prediction)
    return path, data


def validate_benchmark(bundle, counts, dependencies):
    path, rows = formal_json(bundle["benchmark_runs"], "Spark benchmark runs", dependencies)
    need(isinstance(rows, list) and rows, "Benchmark runs missing")
    need(all(row.get("experiment_status") == "formal" and row.get("synthetic") is False
             for row in rows), "Benchmark has non-formal run")
    full = [row for row in rows if row.get("fraction") == 1.0]
    need({"serial", "multiprocessing", "spark"}.issubset({row.get("method") for row in full}),
         "Benchmark needs serial, multiprocessing and Spark full-data runs")
    configurations = {(row.get("method"), row.get("workers")) for row in full}
    for method, workers in configurations:
        repeated = [row for row in full if row.get("method") == method and row.get("workers") == workers]
        need(len(repeated) >= 3 and len({row.get("repeat") for row in repeated}) >= 3,
             "Benchmark full-data configuration needs at least three repeats")
    total = sum(w for w, _ in counts.values())
    for row in full:
        need(row.get("windows") == total and row.get("valid_windows") == total,
             "Benchmark full-data run has partial windows")
        finite_number(row.get("end_to_end_seconds"), "Benchmark elapsed seconds", 0)
    summary_path, summary = formal_json(bundle["benchmark_summary"], "Spark benchmark summary", dependencies)
    need(isinstance(summary, list) and summary and all(
        row.get("experiment_status") == "formal" and row.get("synthetic") is False
        for row in summary), "Benchmark summary is not formal")
    need({"serial", "multiprocessing", "spark"}.issubset({
        row.get("method") for row in summary if row.get("subset") == "subset_100" and
        row.get("repeats", 0) >= 3}), "Benchmark summary lacks three-repeat full-data comparison")
    return path, rows, summary_path, summary


def validate_disagreement(bundle, manifest_path, split_hash, counts, dependencies):
    """Bind the existing initial-run expert-disagreement analysis to page 15."""
    path, data = formal_json(bundle["disagreement_analysis"], "Expert disagreement", dependencies)
    need(data.get("schema_version") == "hms_disagreement_analysis_v1" and
         data.get("status") == "formal" and data.get("split_hash") == split_hash and
         data.get("manifest_sha256") == digest(manifest_path) and
         data.get("method", {}).get("threshold_source") == "full_validation_split_median",
         "Expert disagreement analysis lacks formal validation-set provenance")
    threshold = finite_number(data["method"].get("threshold_nats"),
                              "Expert entropy threshold", 0)
    models = data.get("metrics", {})
    for name in ("tcn", "graph_gru", "odebrain_inspired_hms"):
        groups = models.get(name, {})
        need(groups.get("all", {}).get("n_windows") == counts["test"][0],
             "Expert disagreement does not cover the test set: {}".format(name))
        low, high = groups.get("low", {}), groups.get("high", {})
        need(isinstance(low.get("n_windows"), int) and isinstance(high.get("n_windows"), int)
             and low["n_windows"] + high["n_windows"] == counts["test"][0],
             "Expert disagreement strata do not cover test windows: {}".format(name))
        for group in (low, high):
            finite_number(group.get("kl_window_mean"), "Expert disagreement KL", 0)
    return path, data, threshold


def validate_comparison(bundle, key, split_hash, counts, dependencies, min_rows=2):
    path, data = formal_json(bundle[key], key, dependencies)
    need(data.get("experiment_status") == "formal" and data.get("split") == "test",
         "{} must be formal test evidence".format(key))
    same_hash(data.get("split_hash"), split_hash, key)
    if key == "forecast_comparison":
        need(data.get("schema_version") == "hms_forecast_comparison_v2",
             "Forecast comparison needs the eligible-test schema")
        eligibility = data.get("forecast_eligibility")
        need(isinstance(eligibility, dict), "Forecast eligibility evidence is missing")
        cache_dir = data.get("provenance", {}).get("graph_cache_dir")
        need(isinstance(cache_dir, str) and cache_dir, "Forecast cache path is missing")
        cache_index = path_from(str(Path(cache_dir) / "index.json"))
        cache = read_json(cache_index)
        dependencies[relative(cache_index)] = digest(cache_index)
        need(cache.get("schema_version") == "hms_graph_forecast_cache_v2",
             "Forecast cache must use the eligibility schema")
        need(cache.get("manifest_sha256") == digest(path_from(bundle["manifest"])),
             "Forecast cache manifest differs from formal split")
        expected = cache.get("forecast_eligibility", {}).get("by_split", {}).get("test", {})
        need(eligibility.get("cache_index_sha256") == digest(cache_index) and
             eligibility.get("original_test_windows") == counts["test"][0] and
             eligibility.get("eligible_test_windows") == expected.get("eligible") and
             eligibility.get("excluded_test_windows") == expected.get("excluded") and
             eligibility.get("reason_counts") == expected.get("reason_counts") and
             eligibility.get("eligible_window_ids_sha256") == expected.get("eligible_window_ids_sha256") and
             data.get("n_windows") == expected.get("eligible") and
             data.get("n_patients", 0) > 0 and data.get("n_patients", 0) <= counts["test"][1],
             "Forecast test counts or eligibility provenance differ")
        positive_int(data.get("n_eeg_records"), "Forecast EEG record count")
        need(isinstance(data.get("eeg_record_ids_sha256"), str) and
             HEX64.fullmatch(data["eeg_record_ids_sha256"]),
             "Forecast EEG record identity hash missing")
    else:
        need(data.get("n_windows") == counts["test"][0] and data.get("n_patients") == counts["test"][1],
             "{} test counts differ".format(key))
    rows = data.get("rows")
    need(isinstance(rows, list) and len(rows) >= min_rows and
         all(isinstance(row, dict) for row in rows),
         "{} needs measured comparison rows".format(key))
    if key == "ablation":
        from scripts.run_ablations import CODE_FILES, load_plan
        plan_path = path_from(data.get("plan"))
        _, plan = load_plan(plan_path)
        need(data.get("schema_version") == "ode_ablation_formal_v1" and
             data.get("plan_sha256") == digest(plan_path) and
             data.get("manifest_sha256") == digest(path_from(bundle["manifest"])) and
             len(rows) == len(plan["arms"]) and
             {row.get("name") for row in rows} == {arm["name"] for arm in plan["arms"]} and
             data.get("code_sha256") == {name: digest(ROOT / name) for name in CODE_FILES},
             "Ablation does not cover all frozen plan arms or source hashes")
        dependencies[relative(plan_path)] = digest(plan_path)
    for row in rows:
        need(isinstance(row.get("name"), str) and row["name"], "Unnamed {} row".format(key))
        if key == "ablation":
            finite_number(row.get("kl_window_mean"), "Ablation KL", 0)
            need(row.get("n_windows") == counts["test"][0] and
                 row.get("n_patients") == counts["test"][1] and
                 row.get("split_hash") == split_hash and row.get("seed") == plan["seed"],
                 "Ablation arm test scope differs: {}".format(row["name"]))
            for source_key, checksum_key in (("checkpoint", "checkpoint_sha256"),
                                             ("config", "config_sha256"),
                                             ("metrics_file", "metrics_sha256"),
                                             ("predictions_file", "predictions_sha256")):
                source = path_from(row.get(source_key))
                need(digest(source) == row.get(checksum_key),
                     "Ablation arm source hash differs: {} {}".format(row["name"], source_key))
                dependencies[relative(source)] = digest(source)
        else:
            for horizon in ("1", "3", "5"):
                finite_number(row.get("mae_by_horizon_s", {}).get(horizon),
                              "Forecast MAE at {} s".format(horizon), 0)
                finite_number(row.get("eeg_record_equal_mae_by_horizon_s", {}).get(horizon),
                              "Forecast EEG-record MAE at {} s".format(horizon), 0)
            method = data.get("methods", {}).get(row["name"])
            need(isinstance(method, list) and {item.get("horizon_seconds") for item in method} == {1, 3, 5},
                 "Forecast patient-equal horizon evidence missing")
            for item in method:
                finite_number(item.get("patient_macro_mae"), "Forecast patient-equal MAE", 0)
                finite_number(item.get("eeg_record_equal_mae"), "Forecast record-equal MAE", 0)
    return path, data


def validate_ablation_weighted(bundle, ablation_path, ablation, split_hash, counts,
                               dependencies):
    """Bind three test-set weighting views to every frozen ablation arm."""
    from scripts.supplement_ablation_metrics import supplement

    path, data = formal_json(bundle["ablation_weighted_metrics"],
                             "Weighted ablation metrics", dependencies)
    need(data.get("schema_version") == "ode_ablation_weighted_metrics_v1" and
         data.get("experiment_status") == "formal" and data.get("split") == "test" and
         data.get("test_used_for_selection") is False,
         "Weighted ablation evidence has wrong test protocol")
    same_hash(data.get("split_hash"), split_hash, "Weighted ablation metrics")
    need(data.get("ablation_summary_sha256") == digest(ablation_path) and
         path_from(data.get("ablation_summary")) == ablation_path and
         data.get("manifest_sha256") == ablation.get("manifest_sha256") and
         data.get("plan_sha256") == ablation.get("plan_sha256") and
         data.get("n_windows") == counts["test"][0] and
         data.get("n_patients") == counts["test"][1],
         "Weighted ablation source or full test coverage differs")
    source_rows = {row["name"]: row for row in ablation["rows"]}
    rows = data.get("rows")
    need(isinstance(rows, list) and len(rows) == len(source_rows) == 3 and
         all(isinstance(row, dict) for row in rows) and
         {row.get("name") for row in rows} == set(source_rows),
         "Weighted ablation must cover exactly the three frozen arms")
    for row in rows:
        source = source_rows[row["name"]]
        need(row.get("predictions_sha256") == source["predictions_sha256"] and
             row.get("metrics_sha256") == source["metrics_sha256"] and
             path_from(row.get("predictions_file")) == path_from(source["predictions_file"]) and
             path_from(row.get("metrics_file")) == path_from(source["metrics_file"]),
             "Weighted ablation source hash differs: {}".format(row["name"]))
        for unit in ("window_equal", "patient_equal", "eeg_record_equal"):
            finite_number(row.get("aggregations", {}).get(unit, {}).get("kl_mean"),
                          "{} {} KL".format(row["name"], unit), 0)
    try:
        with tempfile.TemporaryDirectory(prefix="formal-slides-ablation-weighted-") as temporary:
            regenerated = supplement(ablation_path,
                                     Path(temporary) / "ablation_weighted_metrics.json")
    except (OSError, ValueError, KeyError) as exc:
        raise EvidenceError("Weighted ablation evidence failed revalidation: {}".format(exc))
    need(regenerated == data,
         "Weighted ablation evidence differs from frozen test predictions")
    return path, data


def validate_dynamics(bundle, manifest_path, split_hash, counts, ode_checkpoint_sha, dependencies):
    import numpy as np
    import pandas as pd
    path, index = formal_json(bundle["dynamics_index"], "Dynamics index", dependencies)
    need(index.get("schema_version") == "ode_results_v1" and
         index.get("provenance") == "real_hms_checkpoint", "Dynamics is not formal")
    same_hash(index.get("split_hash"), split_hash, "Dynamics")
    need(index.get("checkpoint_sha256") == ode_checkpoint_sha, "Dynamics checkpoint differs from ODE evaluation")
    need(index.get("manifest_file_coverage") == 1.0, "Dynamics has partial EEG coverage")
    pca_count = positive_int(index.get("pca_train_window_count"), "Dynamics PCA train count")
    need(pca_count <= counts["train"][0] + counts["val"][0],
         "Dynamics PCA fit count exceeds development split")
    pca = path_from(str(path.parent / index["pca_file"]))
    dependencies[relative(pca)] = digest(pca)
    with np.load(str(pca), allow_pickle=False) as projection:
        need(str(projection["split_hash"]) == split_hash, "Dynamics PCA split differs")
        need(str(projection["model_id"]) == index.get("model_id"), "Dynamics PCA model differs")
        pca_ids = [str(v) for v in projection["train_window_ids"].tolist()]
    train_ids = set(pd.read_csv(manifest_path, usecols=["window_id", "split"])
                    .query("split in ['train', 'val']").window_id.astype(str))
    need(len(pca_ids) == pca_count and len(set(pca_ids)) == len(pca_ids)
         and set(pca_ids).issubset(train_ids), "Dynamics PCA contains non-development windows")
    lineage = index.get("training_lineage", {})
    need(lineage.get("classification_checkpoint") == ode_checkpoint_sha,
         "Dynamics training lineage differs")
    if lineage.get("forecast_head_source") not in (None, "same_checkpoint"):
        forecast_source = path_from(lineage["forecast_head_source"])
        need(digest(forecast_source) == lineage.get("forecast_head_source_sha256"),
             "Forecast source checkpoint hash differs")
        dependencies[relative(forecast_source)] = digest(forecast_source)
    cases = []
    seen_case_ids = set()
    for item in index.get("cases", []):
        if item.get("split") != "test":
            continue
        case_id = item.get("id")
        need(isinstance(case_id, str) and case_id.strip() and case_id not in seen_case_ids,
             "Dynamics test case ID is missing or duplicated")
        seen_case_ids.add(case_id)
        case_path, case = formal_json(str(path.parent / item["metadata"]), "Dynamics case", dependencies)
        need(case.get("provenance") == "real_hms_checkpoint", "Dynamics case is not formal")
        same_hash(case.get("split_hash"), split_hash, "Dynamics case")
        need(case.get("checkpoint_sha256") == ode_checkpoint_sha, "Dynamics case model differs")
        need(case.get("split") == "test" and case.get("window_id") == item.get("window_id"),
             "Dynamics index/case mismatch")
        need(case.get("forecast_cut_s") == 5 and case.get("time_unit") == "physical seconds",
             "Dynamics time protocol missing")
        tags = case.get("selection_tags")
        need(isinstance(tags, list) and tags and all(isinstance(tag, str) and tag.strip() for tag in tags)
             and tags == item.get("tags"), "Dynamics case selection labels differ from index")
        for field in ("true_class", "predicted_class"):
            value = case.get(field)
            need(type(value) is int and 0 <= value < 6,
                 "Dynamics case {} has invalid {}".format(case_id, field))
        figures = validate_dynamics_figures(case_path, case, dependencies)
        cases.append({"id": case_id, "metadata": case_path, "case": case, "figures": figures})
    return path, index, select_dynamics_cases(cases)


def validate_dynamics_figures(case_path, case, dependencies):
    """Require every original PNG before a case can enter formal slide evidence."""
    from PIL import Image, UnidentifiedImageError
    figures = {}
    for name in DYNAMICS_FIGURES:
        record = case.get("figures", {}).get(name, {})
        filename = record.get("png") if isinstance(record, dict) else None
        need(isinstance(filename, str) and filename.strip() and Path(filename).suffix.lower() == ".png",
             "Dynamics figure {} is missing a PNG path".format(name))
        figure_path = path_from(str(case_path.parent / filename))
        need(figure_path.parent == case_path.parent.resolve(),
             "Dynamics figure {} must be in its case directory".format(name))
        try:
            with Image.open(figure_path) as image:
                need(image.format == "PNG", "Dynamics figure {} is not PNG data".format(name))
                image.verify()
                need(image.width >= 100 and image.height >= 100,
                     "Dynamics figure {} is too small".format(name))
        except (OSError, UnidentifiedImageError) as exc:
            raise EvidenceError("Invalid dynamics PNG {}: {}".format(figure_path, exc))
        dependencies[relative(figure_path)] = digest(figure_path)
        figures[name] = figure_path
    return figures


def select_dynamics_cases(cases):
    """Select stable, actual correct/error test examples from validated cases."""
    need(cases, "No formal test dynamics cases")
    correct_rows = [row for row in cases if row["case"]["predicted_class"] ==
                    row["case"]["true_class"]]
    error_rows = [row for row in cases if row["case"]["predicted_class"] !=
                  row["case"]["true_class"]]
    correct = min(correct_rows, key=lambda row: (
        "typical_correct" not in row["case"]["selection_tags"], row["id"])) if correct_rows else None
    error = min(error_rows, key=lambda row: (
        "typical_error" not in row["case"]["selection_tags"], row["id"])) if error_rows else None
    need(correct is not None, "Formal dynamics needs a correctly predicted test case")
    need(error is not None, "Formal dynamics needs an incorrectly predicted test case")
    return {"correct": correct, "error": error}


def forecast_slide_scope(forecast, test_n, test_patients):
    """Bind page 16 counts to the forecast-eligible subset."""
    eligibility = forecast.get("forecast_eligibility", {})
    eligible = positive_int(eligibility.get("eligible_test_windows"), "Forecast eligible test count")
    excluded = eligibility.get("excluded_test_windows")
    need(type(excluded) is int and excluded >= 0 and eligible + excluded == test_n,
         "Forecast slide eligible/excluded counts differ from original test count")
    need(forecast.get("n_windows") == eligible, "Forecast slide count differs from comparison")
    patients = positive_int(forecast.get("n_patients"), "Forecast eligible patient count")
    need(patients <= test_patients, "Forecast eligible patients exceed original test patients")
    return {"n_windows": eligible, "n_patients": patients,
            "evaluation_scope": "eligible_test_forecast_windows",
            "original_test_n_windows": test_n,
            "original_test_n_patients": test_patients,
            "excluded_count": excluded}


def validate_ui(bundle, split_hash, dependencies):
    from PIL import Image, UnidentifiedImageError

    ui_path, ui = formal_json(bundle["ui_capture"], "UI capture", dependencies)
    need(ui.get("experiment_status") == "formal", "UI capture is not formal")
    same_hash(ui.get("split_hash"), split_hash, "UI capture")
    need(ui.get("formal_gate", {}).get("passed") is True and
         ui.get("smoke_watermark_added") is False and
         ui.get("page") == "ODE 动力学" and
         type(ui.get("rendered_charts")) is int and ui["rendered_charts"] >= 3,
         "UI capture lacks a rendered formal dynamics page")
    screenshot = path_from(ui.get("image"))
    need(screenshot.suffix.lower() == ".png", "UI screenshot must be PNG")
    screenshot_sha = digest(screenshot)
    need(ui.get("image_sha256") == screenshot_sha,
         "UI screenshot differs from capture metadata")
    try:
        with Image.open(screenshot) as image:
            need(image.format == "PNG" and image.width >= 900 and image.height >= 600,
                 "UI screenshot has invalid geometry or format")
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise EvidenceError("Invalid UI screenshot {}: {}".format(screenshot, exc))
    dependencies[relative(screenshot)] = screenshot_sha
    return ui_path, ui, screenshot


def validate_resources(bundle, dependencies):
    resources_path, resources = formal_json(bundle["resources"], "Resources", dependencies)
    need(resources.get("experiment_status") == "formal", "Resources are not verified")
    for key in ("repository", "weights"):
        row = resources.get(key, {})
        need(isinstance(row.get("url"), str) and row["url"].startswith("https://")
             and row.get("verified_http_status") == 200 and row.get("verified_at"),
             "{} URL lacks HTTP 200 verification".format(key))
    return resources_path, resources


def render_panel(title, lines, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    fig.patch.set_facecolor("#0b1d2e")
    ax.set_facecolor("#122c41")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.04, .90, title, color="#21c5ae", fontsize=18, fontweight="bold", va="center")
    wrapped = [part for line in lines for part in textwrap.wrap(str(line), width=66,
               break_long_words=True, break_on_hyphens=False)]
    n = len(wrapped)
    need(n <= 10, "Too many lines for slide panel: {}".format(title))
    for i, line in enumerate(wrapped):
        ax.text(.05, .77 - i * (.68 / max(n, 1)), line, color="#f8fbfc",
                fontsize=11 if n > 8 else 12, va="top", family="DejaVu Sans")
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=.1)
    plt.close(fig)


def classification_kl_chart_data(deep, spark_record):
    """Use the fixed model order, never a test-result-driven ranking."""
    names = ("tcn", "spectrogram_cnn", "fusion", "graph_gru",
             "odebrain_inspired_hms", "lr", "rf")
    labels = ("TCN", "CNN", "Fusion", "GraphGRU", "ODE", "Spark LR", "Spark RF")
    values = []
    for name in names:
        row = deep[name][1] if name in deep else spark_record["models"][name]
        values.append(tuple(row[field] for field in
            ("kl_window_mean", "kl_patient_equal_mean", "kl_eeg_record_equal_mean")))
    return labels, values


def ablation_kl_chart_data(ablation_weighted):
    labels = {"pretrained_rk4_025": "Pretrain · RK4 .25",
              "scratch_rk4_025": "Scratch · RK4 .25",
              "pretrained_rk4_050": "Pretrain · RK4 .50"}
    rows = ablation_weighted["rows"]
    return ([labels.get(row["name"], row["name"]) for row in rows],
            [tuple(row["aggregations"][unit]["kl_mean"] for unit in
                   ("window_equal", "patient_equal", "eeg_record_equal"))
             for row in rows])


def render_kl_chart(title, labels, values, output, note):
    """Show window/patient/EEG-record KL without hiding the aggregation unit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    fig.patch.set_facecolor("#0b1d2e")
    ax.set_facecolor("#122c41")
    y = np.arange(len(labels), dtype=float)
    palette = ("#21c5ae", "#f5bd6d", "#8ea9f4")
    for column, (name, color) in enumerate(zip(("Window", "Patient", "EEG record"), palette)):
        ax.barh(y + (column - 1) * .23, [row[column] for row in values],
                height=.20, color=color, label=name)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="#f8fbfc", fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(row) for row in values) * 1.12)
    ax.set_xlabel("KL(target || prediction), lower is better", color="#f8fbfc", fontsize=11)
    ax.tick_params(axis="x", colors="#b8d2dc", labelsize=10)
    ax.grid(axis="x", color="#577080", alpha=.40)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(title, x=.03, ha="left", color="#21c5ae", fontsize=17, weight="bold")
    fig.legend(loc="upper right", bbox_to_anchor=(.96, .99), ncol=3,
               facecolor="#122c41", edgecolor="none", labelcolor="#f8fbfc", fontsize=10)
    fig.text(.03, .018, note, color="#b8d2dc", fontsize=9, va="bottom")
    fig.subplots_adjust(left=.22, right=.96, top=.83, bottom=.18)
    fig.savefig(output, facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close(fig)


def render_forecast_chart(forecast, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    fig.patch.set_facecolor("#0b1d2e")
    ax.set_facecolor("#122c41")
    names = {"odebrain_inspired_hms": "ODE", "graph_gru": "GraphGRU",
             "persistence": "Persistence"}
    colors = {"odebrain_inspired_hms": "#21c5ae", "graph_gru": "#8ea9f4",
              "persistence": "#f5bd6d"}
    for row in forecast["rows"]:
        name = row["name"]
        values = [row["mae_by_horizon_s"][str(h)] for h in (1, 3, 5)]
        ax.plot((1, 3, 5), values, marker="o", markersize=8, linewidth=2.7,
                color=colors[name], label=names[name])
        ax.text(5.12, values[-1], "{:.3f}".format(values[-1]),
                color=colors[name], fontsize=11, va="center")
    ax.set_xlim(.6, 6.0)
    ax.set_ylim(.55, max(row["mae_by_horizon_s"]["5"] for row in forecast["rows"]) * 1.10)
    ax.set_xticks((1, 3, 5))
    ax.set_xlabel("Forecast horizon (seconds)", color="#f8fbfc", fontsize=11)
    ax.set_ylabel("Mean absolute error", color="#f8fbfc", fontsize=11)
    ax.tick_params(colors="#b8d2dc", labelsize=10)
    ax.grid(color="#577080", alpha=.40)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="upper left", facecolor="#122c41", edgecolor="none",
              labelcolor="#f8fbfc", fontsize=10)
    fig.suptitle("Future graph nodes · eligible test windows", x=.03,
                 ha="left", color="#21c5ae", fontsize=17, weight="bold")
    fig.text(.67, .77, "5 s MAE\nPatient / EEG record", color="#f8fbfc",
             fontsize=11, weight="bold", va="top")
    for index, name in enumerate(("odebrain_inspired_hms", "graph_gru", "persistence")):
        metric = next(item for item in forecast["methods"][name]
                      if item["horizon_seconds"] == 5)
        fig.text(.67, .58 - index * .15,
                 "{}\n{:.3f} / {:.3f}".format(names[name],
                     metric["patient_macro_mae"], metric["eeg_record_equal_mae"]),
                 color=colors[name], fontsize=11, va="top")
    eligible = forecast["forecast_eligibility"]
    fig.text(.03, .018, "Eligible {:,} / {:,} test windows; {:,} excluded. Raw-scale node MAE.".format(
        eligible["eligible_test_windows"], eligible["original_test_windows"],
        eligible["excluded_test_windows"]), color="#b8d2dc", fontsize=9, va="bottom")
    fig.subplots_adjust(left=.10, right=.61, top=.82, bottom=.19)
    fig.savefig(output, facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close(fig)


def render_benchmark_chart(full_bench, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ("Serial · 1", "Multiprocessing · 4", "Spark · 1", "Spark · 4")
    keys = (("serial", 1), ("multiprocessing", 4), ("spark", 1), ("spark", 4))
    values = [full_bench[key]["median_end_to_end_seconds"] for key in keys]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    fig.patch.set_facecolor("#0b1d2e")
    ax.set_facecolor("#122c41")
    bars = ax.barh(range(4), values,
                   color=("#8ea9f4", "#21c5ae", "#f5bd6d", "#d490ba"), height=.52)
    ax.set_yticks(range(4))
    ax.set_yticklabels(labels, color="#f8fbfc", fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("Median end-to-end time (seconds), lower is better",
                  color="#f8fbfc", fontsize=11)
    ax.tick_params(axis="x", colors="#b8d2dc", labelsize=10)
    ax.grid(axis="x", color="#577080", alpha=.40)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + max(values) * .015,
                bar.get_y() + bar.get_height() / 2, "{:.0f} s".format(value),
                color="#f8fbfc", fontsize=11, va="center")
    fig.suptitle("Spark batch processing · full data", x=.03,
                 ha="left", color="#21c5ae", fontsize=17, weight="bold")
    fig.text(.03, .018, "Single physical host · shared disk · observed runs; no multi-host claim",
             color="#b8d2dc", fontsize=9, va="bottom")
    fig.subplots_adjust(left=.25, right=.96, top=.83, bottom=.18)
    fig.savefig(output, facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close(fig)


def render_dynamics_pair(page, left, right, output):
    """Arrange two untouched, validated case PNGs with their source labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig = plt.figure(figsize=(10, 5.2), dpi=200, facecolor="#0b1d2e")
    fig.text(.025, .965, "{} | {}".format(page, TITLES[page]), color="#21c5ae",
             fontsize=17, weight="bold", va="top")
    for column, (kind, name, record) in enumerate((left, right)):
        x = .025 + column * .49
        case = record["case"]
        status = "Correct prediction" if kind == "correct" else "Incorrect prediction"
        fig.text(x, .878, "{}  |  {}".format(status, name.replace("_", " ")),
                 color="#f8fbfc", fontsize=12, weight="bold", va="top")
        fig.text(x, .825, "Case ID: {}".format(record["id"]), color="#f8fbfc",
                 fontsize=10, va="top")
        fig.text(x, .783, "Selection: {}".format(", ".join(case["selection_tags"])),
                 color="#b8d2dc", fontsize=9.5, va="top")
        ax = fig.add_axes([x, .145, .46, .59])
        ax.set_facecolor("white")
        with Image.open(record["figures"][name]) as source:
            ax.imshow(source.convert("RGB"))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.text(x, .108, "Actual class {}  |  predicted class {}".format(
            case["true_class"], case["predicted_class"]), color="#f8fbfc",
            fontsize=10, va="top")
    fig.text(.025, .025,
             "Class outcome: full 50 s input  |  forecast figures: observed 0–5 s input",
             color="#b8d2dc", fontsize=8.5, va="bottom")
    fig.savefig(output, dpi=200, facecolor=fig.get_facecolor(),
                bbox_inches=None, pad_inches=0)
    plt.close(fig)


def prepare(bundle_path, output_path, preview=False):
    dependencies = {}
    bundle_path = path_from(str(bundle_path))
    bundle = read_json(bundle_path)
    need(isinstance(bundle, dict), "Input bundle must be a JSON object")
    reject_smoke(bundle, "input bundle")
    dependencies[relative(bundle_path)] = digest(bundle_path)
    manifest_path, audit_path, split_path, audit, split_hash, counts = validate_manifest(bundle, dependencies)
    feature_path, feature = validate_features(bundle, manifest_path, counts, dependencies)
    spark_path, spark = validate_spark_models(bundle, counts, dependencies)
    spark_record_path, spark_record = validate_spark_record_metrics(
        bundle, manifest_path, split_hash, counts, dependencies)
    weighted_path, weighted = validate_weighted_classification(
        bundle, manifest_path, split_hash, counts, dependencies)
    deep = validate_deep(bundle, manifest_path, split_hash, counts, dependencies)
    paired_path, paired = validate_paired_refit(bundle, manifest_path, split_hash, counts, dependencies)
    bench_path, bench, bench_sum_path, bench_sum = validate_benchmark(bundle, counts, dependencies)
    ablation_path, ablation = validate_comparison(bundle, "ablation", split_hash, counts, dependencies)
    ablation_weighted_path, ablation_weighted = validate_ablation_weighted(
        bundle, ablation_path, ablation, split_hash, counts, dependencies)
    disagreement_path, disagreement, disagreement_threshold = validate_disagreement(
        bundle, manifest_path, split_hash, counts, dependencies)
    forecast_path, forecast = validate_comparison(bundle, "forecast_comparison", split_hash, counts, dependencies)
    dyn_path, dyn_index, dyn_cases = validate_dynamics(
        bundle, manifest_path, split_hash, counts, deep["odebrain_inspired_hms"][2], dependencies)
    ui_path, ui, screenshot = validate_ui(bundle, split_hash, dependencies)
    if preview:
        resources_path, resources = None, None
    else:
        resources_path, resources = validate_resources(bundle, dependencies)
    output_path = Path(output_path)
    output_path = output_path if output_path.is_absolute() else ROOT / output_path
    expected_output = "results/slides_evidence_preview.json" if preview else "results/slides_evidence.json"
    need(output_path.resolve() == (ROOT / expected_output).resolve(),
         "PPTX builder reads {}".format(expected_output))
    figure_dir = ROOT / ("results/figures/preview_slides" if preview else
                         "results/figures/formal_slides")
    figure_dir.mkdir(parents=True, exist_ok=True)
    test_n, test_patients = counts["test"]
    rows = {}

    def add(page, source, lines=None, original_image=None, model="data/spark",
            window_count=None, patient_count=None, extra=None):
        need(page in PAGES and page not in rows, "Duplicate or unexpected slide {}".format(page))
        try:
            source.resolve().relative_to(ROOT.resolve())
        except ValueError:
            raise EvidenceError("PPTX source must be inside project: {}".format(source))
        image_path = figure_dir / ("slide_{:02d}.png".format(page))
        if original_image:
            if original_image.resolve() != image_path.resolve():
                shutil.copyfile(original_image, image_path)
            need(image_path.is_file(), "Missing prepared slide image: {}".format(image_path))
        else:
            render_panel("{} | {}".format(page, TITLES[page]), lines, image_path)
        rows[str(page)] = {"image": relative(image_path), "source_file": relative(source),
                           "split_hash": split_hash,
                           "n_patients": test_patients if patient_count is None else patient_count,
                           "n_windows": test_n if window_count is None else window_count,
                           "model_version": model,
                           "experiment_status": "local_preview" if preview else "formal",
                           "smoke_subset": False,
                           "source_sha256": dependencies[relative(source)],
                           "image_sha256": digest(image_path),
                           "input_sha256": dict(sorted(dependencies.items()))}
        if extra:
            rows[str(page)].update(extra)

    import pandas as pd
    valid_classes = pd.read_csv(manifest_path, usecols=["split", "class_id"])
    valid_classes = valid_classes.loc[valid_classes["split"].isin(("train", "val", "test"))]
    class_counts = valid_classes["class_id"].value_counts().to_dict()
    need(set(class_counts) == set(range(6)) and sum(class_counts.values()) == audit["valid_windows"],
         "Class counts do not cover all audited valid windows")
    add(4, audit_path, ["Metadata rows: {:,}".format(audit["metadata_rows"]),
                        "EEG files: {:,}".format(audit["raw_eeg_files_found"]),
                        "Valid windows: {:,}".format(audit["valid_windows"]),
                        "Patients: {:,}".format(audit["patient_count"]),
                        "Unique annotated context: {:.1f} h".format(
                            audit["unique_annotated_context_seconds"] / 3600),
                        "Class 0/1/2: {:,} / {:,} / {:,}".format(*(class_counts[i] for i in (0, 1, 2))),
                        "Class 3/4/5: {:,} / {:,} / {:,}".format(*(class_counts[i] for i in (3, 4, 5)))])
    add(5, split_path, ["Patient-disjoint audited split", *(
        "{}: {:,} windows / {:,} patients".format(name, *counts[name])
        for name in ("train", "val", "test")), "Split SHA256: {}".format(split_hash[:20])])
    add(7, bench_path, ["Observed host count: {}".format(max(r["physical_hosts"] for r in bench)),
                        "Spark: {}".format(feature["spark_version"]),
                        "Full run windows: {:,}".format(sum(w for w, _ in counts.values())),
                        "No multi-host claim; see benchmark records"])
    add(8, feature_path, ["Spark feature processing", "Partitions: {}".format(feature["partitions"]),
                          "Processing + write: {:.1f} s".format(feature["wall_seconds_including_write"]),
                          "Valid windows: {:,}".format(sum(w for w, _ in counts.values())),
                          "Spark version: {}".format(feature["spark_version"])])
    add(9, deep["odebrain_inspired_hms"][0], ["Full test; deep models refit on train+val",
        "MLlib LR/RF models: training split only", "KL(target || prediction)", *(
        "{} (Spark MLlib): {:.4f}".format({"lr": "LR", "rf": "RF"}[name],
                                          spark["models"][name]["splits"]["test"]["kl_window_mean"])
        for name in ("lr", "rf")), *(
        "{}: {:.4f}".format(name, deep[name][1]["kl_window_mean"]) for name in DEEP_MODELS)],
        model=deep["odebrain_inspired_hms"][2][:16],
        extra={"headline_training_protocol": "train_val_fixed_epoch_refit"})
    add(13, deep["odebrain_inspired_hms"][0], ["Deep models: train+val fixed-epoch refit",
        "Epochs frozen from initial validation; no test selection",
        "Test: {:,} windows / {:,} patients".format(test_n, test_patients),
        "KL: window / patient / EEG-record equal means", "Split SHA256: {}".format(split_hash[:20]),
        "ODE checkpoint SHA256: {}".format(deep["odebrain_inspired_hms"][2][:20])],
        model=deep["odebrain_inspired_hms"][2][:16],
        extra={"headline_training_protocol": "train_val_fixed_epoch_refit"})
    kl_chart_path = figure_dir / "slide_14.png"
    kl_chart_labels, kl_chart_values = classification_kl_chart_data(deep, spark_record)
    render_kl_chart("Classification · seven measured models", kl_chart_labels,
                    kl_chart_values, kl_chart_path,
                    "Deep models: train+val refit; Spark LR/RF: train only. Same held-out patient split.")
    add(14, deep["odebrain_inspired_hms"][0], ["Final refit test KL: W / P / E, lower is better", *(
        "{}: {:.3f} / {:.3f} / {:.3f}".format(
            name, deep[name][1]["kl_window_mean"],
            deep[name][1]["kl_patient_equal_mean"],
            deep[name][1]["kl_eeg_record_equal_mean"]) for name in DEEP_MODELS),
        "MLlib train-only W/P/E LR: {:.3f}/{:.3f}/{:.3f}".format(*(
            spark_record["models"]["lr"][field] for field in
            ("kl_window_mean", "kl_patient_equal_mean", "kl_eeg_record_equal_mean"))),
        "MLlib train-only W/P/E RF: {:.3f}/{:.3f}/{:.3f}".format(*(
            spark_record["models"]["rf"][field] for field in
            ("kl_window_mean", "kl_patient_equal_mean", "kl_eeg_record_equal_mean"))),
        "ODE macro F1 W/P/E: {:.3f}/{:.3f}/{:.3f}".format(*(
            weighted["models"]["odebrain_inspired_hms"]["aggregations"][unit]["macro_f1"]
            for unit in ("window_equal", "patient_equal", "eeg_record_equal")))],
        original_image=kl_chart_path, model=deep["odebrain_inspired_hms"][2][:16],
        extra={"headline_training_protocol": "train_val_fixed_epoch_refit",
               "spark_record_metrics_source": relative(spark_record_path),
               "spark_record_metrics_sha256": dependencies[relative(spark_record_path)],
               "weighted_metrics_source": relative(weighted_path),
               "weighted_metrics_sha256": dependencies[relative(weighted_path)]})
    disagreement_rows = disagreement["metrics"]
    ablation_chart_path = figure_dir / "slide_15.png"
    ablation_labels, ablation_values = ablation_kl_chart_data(ablation_weighted)
    render_kl_chart("Ablation · frozen train-only runs",
                    ablation_labels, ablation_values,
                    ablation_chart_path,
                    "Validation entropy threshold {:.3f} nats; ODE KL low/high {:.3f}/{:.3f}. Initial-run protocol.".format(
                        disagreement_threshold,
                        disagreement_rows["odebrain_inspired_hms"]["low"]["kl_window_mean"],
                        disagreement_rows["odebrain_inspired_hms"]["high"]["kl_window_mean"]))
    add(15, disagreement_path, ["Initial-run only: ablation and expert disagreement",
        "Ablation KL W/P/E, separate protocol from refit:", *(
        "{}: {:.3f} / {:.3f} / {:.3f}".format(row["name"], *(
            row["aggregations"][unit]["kl_mean"] for unit in
            ("window_equal", "patient_equal", "eeg_record_equal")))
        for row in ablation_weighted["rows"]),
        "Entropy threshold from validation: {:.3f} nats".format(disagreement_threshold),
        "Low/high test windows: {:,} / {:,}".format(
            disagreement_rows["tcn"]["low"]["n_windows"],
            disagreement_rows["tcn"]["high"]["n_windows"]),
        "ODE KL low/high: {:.3f} / {:.3f}".format(
            disagreement_rows["odebrain_inspired_hms"]["low"]["kl_window_mean"],
            disagreement_rows["odebrain_inspired_hms"]["high"]["kl_window_mean"])],
        original_image=ablation_chart_path,
        model="initial_ablation_runs", extra={"ablation_source": relative(ablation_path),
        "ablation_source_sha256": dependencies[relative(ablation_path)],
        "ablation_weighted_source": relative(ablation_weighted_path),
        "ablation_weighted_sha256": dependencies[relative(ablation_weighted_path)],
        "disagreement_scope": "initial_train_only_models"})
    forecast_scope = forecast_slide_scope(forecast, test_n, test_patients)
    forecast_protocol = forecast.get("training_protocol")
    need(forecast_protocol in ("validation_selected_train_only", "train_val_fixed_epoch_refit"),
         "Forecast training protocol is missing or unknown")
    forecast_phase = ("Final-refit" if forecast_protocol == "train_val_fixed_epoch_refit"
                      else "Initial-run")
    forecast_5s = {name: next(item for item in forecast["methods"][name]
                              if item["horizon_seconds"] == 5)
                   for name in ("odebrain_inspired_hms", "graph_gru", "persistence")}
    forecast_chart_path = figure_dir / "slide_16.png"
    render_forecast_chart(forecast, forecast_chart_path)
    add(16, forecast_path, [forecast_phase + " future-node MAE; eligible test set",
        "Eligible {:,}/{:,}; excluded {:,} for invalid input/target".format(
            forecast["forecast_eligibility"]["eligible_test_windows"], test_n,
            forecast["forecast_eligibility"]["excluded_test_windows"]), *(
        "{}: 1s {:.3f} | 3s {:.3f} | 5s {:.3f}".format(
            row["name"], *(row["mae_by_horizon_s"][h] for h in ("1", "3", "5")))
        for row in forecast["rows"][:6]), *(
        "{} 5s P/E: {:.3f} / {:.3f}".format(name,
            forecast_5s[name]["patient_macro_mae"],
            forecast_5s[name]["eeg_record_equal_mae"])
        for name in ("odebrain_inspired_hms", "graph_gru", "persistence"))],
        original_image=forecast_chart_path, model=forecast_protocol,
        window_count=forecast_scope["n_windows"],
        patient_count=forecast_scope["n_patients"],
        extra={key: value for key, value in forecast_scope.items()
               if key not in ("n_windows", "n_patients")})
    for page, ((left_kind, left_name), (right_kind, right_name)) in DYNAMICS_LAYOUT.items():
        image_path = figure_dir / ("slide_{:02d}.png".format(page))
        render_dynamics_pair(page,
                             (left_kind, left_name, dyn_cases[left_kind]),
                             (right_kind, right_name, dyn_cases[right_kind]), image_path)
        add(page, dyn_path, original_image=image_path,
            model=deep["odebrain_inspired_hms"][2][:16],
            extra={"plotted_case_ids": [dyn_cases[left_kind]["id"], dyn_cases[right_kind]["id"]],
                   "plotted_case_count": 2,
                   "figure_types": [left_name, right_name],
                   "selection_scope": "selected formal test cases"})
    full_bench = {(r.get("method"), r.get("workers")): r for r in bench_sum
                  if r.get("subset") == "subset_100"}
    need(all(key in full_bench for key in (("serial", 1), ("multiprocessing", 4),
                                        ("spark", 1), ("spark", 4))),
         "Benchmark summary needs serial-1, multiprocessing-4, Spark-1 and Spark-4")
    benchmark_chart_path = figure_dir / "slide_20.png"
    render_benchmark_chart(full_bench, benchmark_chart_path)
    add(20, bench_sum_path, ["Full-data end-to-end median", *(
        "{} ({} workers): {:.1f} s".format(label, workers,
             full_bench[(method, workers)]["median_end_to_end_seconds"])
        for method, workers, label in (("serial", 1, "Serial"),
                                       ("multiprocessing", 4, "Multiprocessing"),
                                       ("spark", 1, "Spark"), ("spark", 4, "Spark"))),
        "Physical hosts: 1",
        "Do not infer multi-node scalability"], original_image=benchmark_chart_path)
    add(21, ui_path, original_image=screenshot, model=deep["odebrain_inspired_hms"][2][:16])
    ode_gru = paired["comparisons"]["ode_minus_graph_gru"]
    add(22, deep["odebrain_inspired_hms"][0], ["Final refit; 2 MLlib classifiers + 5 deep models",
        "Test: {:,} windows / {:,} patients".format(test_n, test_patients),
        "ODE minus GraphGRU KL; positive = ODE higher",
        "Patient equal: {:+.3f} | 95% [{:+.3f}, {:+.3f}]".format(
            ode_gru["patient_equal"]["difference"], *ode_gru["patient_equal"]["ci95"]),
        "EEG-record equal: {:+.3f} | 95% [{:+.3f}, {:+.3f}]".format(
            ode_gru["eeg_record_equal"]["difference"], *ode_gru["eeg_record_equal"]["ci95"]),
        "Paired patient bootstrap; descriptive only",
        "Observed dynamics cases: {}".format(len(dyn_index["cases"]))],
        model=deep["odebrain_inspired_hms"][2][:16],
        extra={"headline_training_protocol": "train_val_fixed_epoch_refit",
               "paired_refit_source": relative(paired_path),
               "paired_refit_sha256": dependencies[relative(paired_path)]})
    if preview:
        add(23, bundle_path, ["Local preview; pending publication review",
            "Project repository: unpublished", "Model weights: unpublished",
            "HMS data: Kaggle competition page"],
            model=deep["odebrain_inspired_hms"][2][:16],
            extra={"publication_status": "pending_review"})
    else:
        add(23, resources_path, ["Verified project resources", resources["repository"]["url"],
            resources["weights"]["url"], "Both links: HTTP 200 recorded",
            "HMS data: Kaggle competition page"], model=deep["odebrain_inspired_hms"][2][:16])
    need(set(map(int, rows)) == set(PAGES), "Missing experimental slide")
    payload = {"schema_version": "formal_slide_evidence_v1", "formal": not preview,
               "experiment_status": "local_preview" if preview else "formal",
               "publication_status": "pending_review" if preview else "verified",
               "split_hash": split_hash,
               "headline_training_protocol": "train_val_fixed_epoch_refit",
               "headline_refit_comparison": relative(deep["odebrain_inspired_hms"][0]),
               "headline_refit_comparison_sha256": dependencies[relative(deep["odebrain_inspired_hms"][0])],
               "manifest_split_hash": split_hash, "manifest_file_sha256": digest(manifest_path),
               "dependencies_sha256": dict(sorted(dependencies.items())), "slides": rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(output_path)
    return output_path


TITLES = {4: "HMS audit", 5: "Patient split", 7: "Deployment", 8: "Spark features",
          9: "Model comparison", 13: "Evaluation protocol", 14: "Classification",
          15: "Ablation", 16: "Future graph", 17: "Trajectory", 18: "Dynamics",
          19: "Observed correlation", 20: "Spark cost", 21: "Interface",
          22: "Outcomes", 23: "Resources"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="Formal input bundle JSON")
    parser.add_argument("--output", help="Evidence JSON; defaults by mode")
    parser.add_argument("--preview", action="store_true",
                        help="Verify experiments but mark local deck pending publication")
    args = parser.parse_args()
    try:
        output = args.output or ("results/slides_evidence_preview.json" if args.preview else
                                 "results/slides_evidence.json")
        print(prepare(args.inputs, output, preview=args.preview))
    except (EvidenceError, KeyError, TypeError) as exc:
        parser.exit(2, "Formal slide evidence rejected: {}\n".format(exc))


if __name__ == "__main__":
    main()
