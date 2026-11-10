#!/usr/bin/env python3
"""Formal HMS expert-disagreement analysis from frozen, paired test predictions.

The validation split alone fixes the expert-entropy median. No output is
written until all three complete runs and their predictions pass validation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.write_model_cards import (CardError, check_run,
                                       manifest_context, path_string, require)
from src.data.hms import CLASSES
from src.evaluation.metrics import sample_kl

MODELS = ("tcn", "graph_gru", "odebrain_inspired_hms")
TARGET = ["target_" + name for name in CLASSES]
PREDICTED = ["pred_" + name for name in CLASSES]


def vote_entropy(values):
    """Natural-log entropy of six normalized expert vote fractions."""
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and values.shape[1] == len(CLASSES)
            and np.isfinite(values).all() and (values >= 0).all(),
            "Invalid expert targets")
    sums = values.sum(axis=1)
    require((sums > 0).all(), "Zero-vote expert target")
    p = values / sums[:, None]
    return -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0), axis=1)


def frozen_strata(validation_targets, test_targets):
    validation = vote_entropy(validation_targets)
    test = vote_entropy(test_targets)
    require(len(validation) > 0 and len(test) > 0, "Empty validation or test split")
    threshold = float(np.median(validation))
    low = test <= threshold
    high = ~low
    require(low.any() and high.any(), "Frozen entropy threshold yields an empty test stratum")
    return threshold, {"all": np.ones(len(test), dtype=bool),
                       "low": low, "high": high}


def aligned_predictions(path, expected):
    """Require every held-out window exactly once, with identical patient/label."""
    required = ["window_id", "patient_id", "class_id", *TARGET, *PREDICTED]
    try:
        frame = pd.read_csv(path, dtype={"window_id": str, "patient_id": str})
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise CardError("Cannot read predictions {}: {}".format(path, exc))
    require(set(required).issubset(frame.columns), "Prediction columns missing: {}".format(path))
    require(frame.window_id.is_unique and len(frame) == len(expected),
            "Predictions do not cover exact held-out test: {}".format(path))
    frame = frame.set_index("window_id", verify_integrity=True)
    require(set(frame.index) == set(expected.index),
            "Prediction window IDs differ from held-out test: {}".format(path))
    frame = frame.loc[expected.index]
    require(frame.patient_id.astype(str).eq(expected.patient_id.astype(str)).all(),
            "Prediction patient mismatch: {}".format(path))
    try:
        hard = pd.to_numeric(frame.class_id, errors="raise").to_numpy(dtype=np.float64)
        truth = pd.to_numeric(expected.class_id, errors="raise").to_numpy(dtype=np.float64)
        target = frame[TARGET].to_numpy(dtype=np.float64)
        predicted = frame[PREDICTED].to_numpy(dtype=np.float64)
        expected_target = expected[TARGET].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise CardError("Invalid prediction numbers {}: {}".format(path, exc))
    require(np.array_equal(hard, truth), "Prediction hard labels differ from manifest: {}".format(path))
    require(np.isfinite(target).all() and np.allclose(target, expected_target, atol=1e-6, rtol=0),
            "Prediction expert targets differ from manifest: {}".format(path))
    require(np.isfinite(predicted).all() and (predicted >= 0).all()
            and (predicted <= 1).all()
            and np.allclose(predicted.sum(axis=1), 1, atol=1e-5, rtol=0),
            "Invalid prediction probabilities: {}".format(path))
    return predicted


def patient_bootstrap_mean(values, patients, mask, n_boot=2000, seed=42):
    values = np.asarray(values, dtype=float)[mask]
    patient = np.asarray(patients, dtype=str)[mask]
    require(type(n_boot) is int and n_boot >= 100, "n_boot must be >= 100")
    require(len(values) == len(patient) and np.isfinite(values).all(),
            "Invalid values for patient bootstrap")
    unique, inverse = np.unique(patient, return_inverse=True)
    require(len(unique) >= 2, "At least two patients required in each bootstrap stratum")
    patient_means = np.bincount(inverse, weights=values) / np.bincount(inverse)
    rng = np.random.RandomState(seed)
    indices = rng.randint(len(unique), size=(n_boot, len(unique)))
    draws = patient_means[indices].mean(axis=1)
    return {"n_patients": int(len(unique)), "patient_equal_mean": float(patient_means.mean()),
            "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
            "resampling_unit": "patient", "n_boot": n_boot, "seed": seed}


def group_metrics(target, predicted, hard, patients, mask, n_boot=2000, seed=42):
    kl = sample_kl(target[mask], predicted[mask])
    patient = patients[mask]
    unique, inverse = np.unique(patient, return_inverse=True)
    per_patient = np.bincount(inverse, weights=kl) / np.bincount(inverse)
    correct = (predicted[mask].argmax(axis=1) == hard[mask]).astype(float)
    return {"n_windows": int(mask.sum()), "n_patients": int(len(unique)),
            "kl_window_mean": float(kl.mean()),
            "kl_patient_equal_mean": float(per_patient.mean()),
            "accuracy_window": float(correct.mean()),
            "kl_patient_bootstrap": patient_bootstrap_mean(
                sample_kl(target, predicted), patients, mask, n_boot, seed),
            "accuracy_patient_bootstrap": patient_bootstrap_mean(
                (predicted.argmax(axis=1) == hard).astype(float), patients, mask, n_boot, seed)}


def paired_patient_bootstrap(values_a, values_b, patients, mask, n_boot=2000, seed=42):
    """Paired A-B difference, each patient equally weighted in each resample."""
    difference = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    bootstrap = patient_bootstrap_mean(difference, patients, mask, n_boot, seed)
    bootstrap["difference_patient_equal_mean"] = bootstrap.pop("patient_equal_mean")
    return bootstrap


def analyze(runs, manifest, metadata, split_summary, results_root, n_boot, seed):
    require(set(runs) == set(MODELS), "Need TCN, graph GRU and ODE runs")
    context = manifest_context(manifest, metadata, split_summary)
    table = pd.read_csv(context["path"], dtype={"window_id": str, "patient_id": str})
    require(set(["class_id", *TARGET]).issubset(table.columns), "Manifest target columns missing")
    val = table.loc[table.split == "val"]
    test = table.loc[table.split == "test"].set_index("window_id", verify_integrity=True).sort_index()
    target = test[TARGET].to_numpy(dtype=float)
    hard = test.class_id.to_numpy(dtype=int)
    patients = test.patient_id.astype(str).to_numpy()
    threshold, strata = frozen_strata(val[TARGET].to_numpy(dtype=float), target)
    require(all(len(np.unique(patients[mask])) >= 2 for mask in strata.values()),
            "Each entropy stratum needs at least two test patients")

    cards = {name: check_run(name, runs[name], context, results_root) for name in MODELS}
    predictions = {}
    for name, card in cards.items():
        path = ROOT / card["evaluation"]["test"]["predictions"]
        predicted = aligned_predictions(path, test)
        actual_kl = float(sample_kl(target, predicted).mean())
        reported_kl = card["evaluation"]["test"]["kl_window_mean"]
        require(abs(actual_kl - reported_kl) <= 1e-6,
                "Prediction KL differs from formal metrics for {}".format(name))
        predictions[name] = predicted

    metrics = {name: {group: group_metrics(target, pred, hard, patients, mask, n_boot, seed)
                      for group, mask in strata.items()} for name, pred in predictions.items()}
    comparisons = {}
    ode = predictions["odebrain_inspired_hms"]
    ode_kl = sample_kl(target, ode)
    ode_correct = (ode.argmax(axis=1) == hard).astype(float)
    for baseline in ("graph_gru", "tcn"):
        base = predictions[baseline]
        base_kl = sample_kl(target, base)
        base_correct = (base.argmax(axis=1) == hard).astype(float)
        comparisons["ode_minus_" + baseline] = {
            group: {"kl_patient_equal_difference": paired_patient_bootstrap(
                        ode_kl, base_kl, patients, mask, n_boot, seed),
                    "accuracy_patient_equal_difference": paired_patient_bootstrap(
                        ode_correct, base_correct, patients, mask, n_boot, seed)}
            for group, mask in strata.items()}

    return {"schema_version": "hms_disagreement_analysis_v1", "status": "formal",
            "method": {"entropy": "natural_log_of_normalized_six_class_expert_votes",
                       "threshold_source": "full_validation_split_median",
                       "threshold_nats": threshold,
                       "low_rule": "test_entropy <= threshold", "high_rule": "test_entropy > threshold",
                       "threshold_validation_windows": int(len(val)),
                       "bootstrap": "paired_patient_resampling_of_within_patient_mean_differences",
                       "difference_direction": "ODE minus baseline"},
            "split_hash": context["split_hash"], "manifest": path_string(context["path"]),
            "manifest_sha256": context["sha256"],
            "split_summary_sha256": context["summary_sha256"],
            "runs": {name: {"run_dir": card["run_dir"], "checkpoint_sha256": card["weights_sha256"],
                             "test_metrics_sha256": card["evaluation"]["test"]["sha256"],
                             "test_predictions_sha256": card["evaluation"]["test"]["predictions_sha256"]}
                     for name, card in cards.items()},
            "metrics": metrics, "paired_comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcn-run", default="artifacts/runs/tcn")
    parser.add_argument("--gru-run", default="artifacts/runs/graph_gru")
    parser.add_argument("--ode-run", default="artifacts/runs/ode_finetune")
    parser.add_argument("--manifest", default="data/curated/split_manifest.csv")
    parser.add_argument("--metadata", default="data/raw/train.csv")
    parser.add_argument("--split-summary", default="data/curated/split_manifest.json")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="results/disagreement_analysis.json")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = analyze({"tcn": args.tcn_run, "graph_gru": args.gru_run,
                          "odebrain_inspired_hms": args.ode_run},
                         args.manifest, args.metadata, args.split_summary,
                         args.results_root, args.n_boot, args.seed)
        output = Path(args.output)
        output = output if output.is_absolute() else ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(output.suffix + ".tmp")
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
        temp.replace(output)
        print(output)
    except (CardError, KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, "Formal disagreement analysis rejected: {}\n".format(exc))


if __name__ == "__main__":
    main()
