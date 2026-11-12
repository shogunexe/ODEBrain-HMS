"""Verify fixed-split refit repeats and report initialization-seed variation."""

import argparse
import json
import statistics
from pathlib import Path

from refit_final import CLASSIFIERS, _check_final_run, summarize_refits
from src.cli import file_sha256

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("kl_window_mean", "kl_patient_equal_mean",
           "kl_eeg_record_equal_mean", "macro_f1")


def aggregate(seeds=(42, 43, 44), output=ROOT / "results/seed_stability/summary.json"):
    if len(seeds) < 2 or len(seeds) != len(set(seeds)) or 42 not in seeds:
        raise ValueError("Report needs distinct seeds including frozen primary seed 42")
    records = {}
    shared = None
    for seed in seeds:
        if seed == 42:
            run_root = ROOT / "artifacts/refit"
            results_root = ROOT / "results/refit"
        else:
            run_root = ROOT / "artifacts/seed_stability" / ("seed%d" % seed)
            results_root = ROOT / "results/seed_stability" / ("seed%d" % seed)
        run_dirs = {name: run_root / name for name in CLASSIFIERS}
        for name, path in run_dirs.items():
            checked = _check_final_run(path)
            if checked["config"]["refit"]["seed"] != seed:
                raise ValueError("Refit seed differs from requested repeat: %s" % name)
        comparison_path = results_root / "final_comparison.json"
        summarize_refits(run_dirs, comparison_path, results_root)
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        identity = (comparison["split_hash"], comparison["manifest_sha256"],
                    comparison["n_windows"], comparison["n_patients"],
                    comparison["n_eeg_records"])
        if shared is None:
            shared = identity
        elif identity != shared:
            raise ValueError("Seed repeats changed the patient split or test coverage")
        records[seed] = {"comparison": str(comparison_path),
                         "sha256": file_sha256(comparison_path),
                         "rows": {row["name"]: row for row in comparison["rows"]}}
    rows = []
    for name in CLASSIFIERS:
        model = {"name": name, "per_seed": {}}
        for metric in METRICS:
            values = [float(records[seed]["rows"][name][metric]) for seed in seeds]
            model[metric] = {"mean": statistics.mean(values),
                             "sample_std": statistics.stdev(values),
                             "values_by_seed": {str(seed): value for seed, value in zip(seeds, values)}}
        for seed in seeds:
            row = records[seed]["rows"][name]
            if row["n_eeg_records"] != shared[4]:
                raise ValueError("Seed repeats changed EEG record coverage")
            model["per_seed"][str(seed)] = {
                "checkpoint_sha256": row["checkpoint_sha256"],
                "test_metrics_sha256": row["test_metrics_sha256"],
                "test_predictions_sha256": row["test_predictions_sha256"],
                "fixed_refit_epochs": row["fixed_refit_epochs"],
                "run_dir": row["run_dir"],
            }
        if len({model["per_seed"][str(seed)]["fixed_refit_epochs"] for seed in seeds}) != 1:
            raise ValueError("Seed repeats changed the validation-selected epoch budget")
        rows.append(model)
    result = {
        "schema_version": "hms_fixed_split_seed_stability_v1",
        "experiment_status": "formal", "smoke_subset": False, "synthetic": False,
        "variability_source": "initialization_seed_on_one_fixed_patient_split",
        "training_protocol": "train_val_fixed_epoch_refit",
        "test_used_for_selection": False,
        "seeds": list(seeds), "n_seeds": len(seeds),
        "split_hash": shared[0], "manifest_sha256": shared[1],
        "n_windows": shared[2], "n_patients": shared[3],
        "n_eeg_records": shared[4],
        "comparisons": {str(seed): {"path": record["comparison"],
                                      "sha256": record["sha256"]}
                        for seed, record in records.items()},
        "rows": rows,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                 allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output", default="results/seed_stability/summary.json")
    args = parser.parse_args()
    result = aggregate(tuple(args.seeds), args.output)
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "n_seeds": result["n_seeds"],
                      "n_models": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
