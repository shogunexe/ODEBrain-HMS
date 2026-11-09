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
    path, data = formal_json(bundle["spark_metrics"], "Spark LR/RF", dependencies)
    need(data.get("experiment_status") == "formal", "Spark LR/RF must be formal")
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
         [row.get("name") for row in rows] == list(DEEP_MODELS),
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
    else:
        need(data.get("n_windows") == counts["test"][0] and data.get("n_patients") == counts["test"][1],
             "{} test counts differ".format(key))
    rows = data.get("rows")
    need(isinstance(rows, list) and len(rows) >= min_rows, "{} needs measured comparison rows".format(key))
    for row in rows:
        need(isinstance(row.get("name"), str) and row["name"], "Unnamed {} row".format(key))
        if key == "ablation":
            finite_number(row.get("kl_window_mean"), "Ablation KL", 0)
        else:
            for horizon in ("1", "3", "5"):
                finite_number(row.get("mae_by_horizon_s", {}).get(horizon),
                              "Forecast MAE at {} s".format(horizon), 0)
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
    ui_path, ui = formal_json(bundle["ui_capture"], "UI capture", dependencies)
    need(ui.get("experiment_status") == "formal", "UI capture is not formal")
    same_hash(ui.get("split_hash"), split_hash, "UI capture")
    screenshot = path_from(ui.get("image"))
    need(screenshot.suffix.lower() == ".png", "UI screenshot must be PNG")
    dependencies[relative(screenshot)] = digest(screenshot)
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
    deep = validate_deep(bundle, manifest_path, split_hash, counts, dependencies)
    bench_path, bench, bench_sum_path, bench_sum = validate_benchmark(bundle, counts, dependencies)
    ablation_path, ablation = validate_comparison(bundle, "ablation", split_hash, counts, dependencies)
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

    add(4, audit_path, ["Metadata rows: {:,}".format(audit["metadata_rows"]),
                        "EEG files: {:,}".format(audit["raw_eeg_files_found"]),
                        "Valid windows: {:,}".format(audit["valid_windows"]),
                        "Patients: {:,}".format(audit["patient_count"])])
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
        "Spark baselines: training split only", "KL(target || prediction)", *(
        "Spark {}: {:.4f}".format(name.upper(), spark["models"][name]["splits"]["test"]["kl_window_mean"])
        for name in ("lr", "rf")), *(
        "{}: {:.4f}".format(name, deep[name][1]["kl_window_mean"]) for name in DEEP_MODELS)],
        model=deep["odebrain_inspired_hms"][2][:16],
        extra={"headline_training_protocol": "train_val_fixed_epoch_refit"})
    add(13, deep["odebrain_inspired_hms"][0], ["Deep models: train+val fixed-epoch refit",
        "Epochs frozen from initial validation; no test selection",
        "Test: {:,} windows / {:,} patients".format(test_n, test_patients),
        "KL(target || prediction), window mean", "Split SHA256: {}".format(split_hash[:20]),
        "ODE checkpoint SHA256: {}".format(deep["odebrain_inspired_hms"][2][:20])],
        model=deep["odebrain_inspired_hms"][2][:16],
        extra={"headline_training_protocol": "train_val_fixed_epoch_refit"})
    add(14, deep["odebrain_inspired_hms"][0], ["Final refit test KL, lower is better", *(
        "{}: {:.4f}".format(name, deep[name][1]["kl_window_mean"]) for name in DEEP_MODELS),
        "Spark (train only) LR: {:.4f} | RF: {:.4f}".format(*(
            spark["models"][n]["splits"]["test"]["kl_window_mean"] for n in ("lr", "rf")))],
        model=deep["odebrain_inspired_hms"][2][:16],
        extra={"headline_training_protocol": "train_val_fixed_epoch_refit"})
    add(15, ablation_path, ["Initial-run ablations; test KL lower is better", *(
        "{}: {:.4f}".format(row["name"], row["kl_window_mean"]) for row in ablation["rows"][:7])],
        model="initial_ablation_runs")
    forecast_scope = forecast_slide_scope(forecast, test_n, test_patients)
    add(16, forecast_path, ["Initial-run future-node MAE; eligible test set",
        "Eligible {:,}/{:,}; excluded {:,} for invalid input/target".format(
            forecast["forecast_eligibility"]["eligible_test_windows"], test_n,
            forecast["forecast_eligibility"]["excluded_test_windows"]), *(
        "{}: 1s {:.3f} | 3s {:.3f} | 5s {:.3f}".format(
            row["name"], *(row["mae_by_horizon_s"][h] for h in ("1", "3", "5")))
        for row in forecast["rows"][:6])], model="initial_forecast_runs",
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
    full_spark = [r for r in bench_sum if r.get("subset") == "subset_100" and r.get("method") == "spark"]
    full_serial = [r for r in bench_sum if r.get("subset") == "subset_100" and r.get("method") == "serial"]
    need(full_serial and full_spark, "Benchmark summary needs full serial and Spark timings")
    add(20, bench_sum_path, ["Full-data end-to-end median", "Serial: {:.1f} s".format(
        full_serial[0]["median_end_to_end_seconds"]), "Spark: {:.1f} s".format(
        full_spark[0]["median_end_to_end_seconds"]), "Physical hosts: 1",
        "Do not infer multi-node scalability"])
    add(21, ui_path, original_image=screenshot, model=deep["odebrain_inspired_hms"][2][:16])
    add(22, deep["odebrain_inspired_hms"][0], ["Final refit test evidence", "Models: 2 Spark + 5 deep",
        "Test: {:,} windows / {:,} patients".format(test_n, test_patients),
        "ODE KL: {:.4f}".format(deep["odebrain_inspired_hms"][1]["kl_window_mean"]),
        "Observed dynamics cases: {}".format(len(dyn_index["cases"]))],
        model=deep["odebrain_inspired_hms"][2][:16],
        extra={"headline_training_protocol": "train_val_fixed_epoch_refit"})
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
