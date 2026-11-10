#!/usr/bin/env python3
"""Run and verify formal ODEBrain-inspired HMS ablations.

``train`` runs forecast pretraining and three classification arms, then their
validation evaluations. ``test`` evaluates the frozen arms on the full test
split and writes a provenance-checked ``results/ablation.json``. ``collect``
only checks existing artifacts and writes the same summary. No smoke result
can be promoted into a formal comparison.
"""

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CLASSES = ("seizure", "lpd", "gpd", "lrda", "grda", "other")
DYNAMIC_CONFIG_KEYS = frozenset(("smoke_subset", "manifest_sha256", "window_snapshot_sha256",
                                 "init_checkpoint", "init_checkpoint_sha256", "source_config_sha256"))
CODE_FILES = ("src/models/odebrain_hms.py", "src/training/engine.py", "src/cli.py",
              "src/data/dataset.py", "src/data/hms.py", "src/evaluation/metrics.py")


class AblationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AblationError(message)


def resolve(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_plan(path):
    plan_path = resolve(path)
    plan = read_yaml(plan_path)
    require(plan.get("schema_version") == "ode_ablation_plan_v1", "Unsupported ablation plan")
    require(type(plan.get("seed")) is int, "Plan requires an integer seed")
    arms = plan.get("arms")
    require(isinstance(arms, list) and len(arms) >= 3, "Need three or more ablation arms")
    names = [arm.get("name") for arm in arms]
    require(all(isinstance(name, str) and name and name.replace("_", "").isalnum()
                for name in names) and len(set(names)) == len(names), "Arm names must be unique safe strings")
    pairs = set()
    for arm in arms:
        step = arm.get("rk4_step_seconds")
        require(arm.get("initialization") in ("pretrain", "scratch"), "Invalid initialization")
        require(isinstance(step, (int, float)) and not isinstance(step, bool)
                and math.isfinite(step) and step > 0, "Invalid RK4 step")
        pairs.add((arm["initialization"], float(step)))
    require(any(("scratch", step) in pairs and ("pretrain", step) in pairs for _, step in pairs),
            "Plan lacks matched pretrained versus scratch arms")
    require(len({step for _, step in pairs}) >= 2, "Plan lacks RK4 step comparison")
    for key in ("pretrain_config", "finetune_config", "run_root", "generated_config_dir", "output"):
        require(isinstance(plan.get(key), str) and plan[key], "Plan missing " + key)
    return plan_path, plan


def expected_configs(plan_path, plan):
    """Derive immutable YAML inputs; CLI adds only recorded runtime hashes."""
    import copy
    from src.cli import input_modalities
    pretrain = copy.deepcopy(read_yaml(resolve(plan["pretrain_config"])))
    base = copy.deepcopy(read_yaml(resolve(plan["finetune_config"])))
    require(pretrain["model"]["name"] == base["model"]["name"] == "odebrain_inspired_hms",
            "Both configs must use ODEBrainInspiredHMS")
    require(pretrain["model"]["kwargs"] == base["model"]["kwargs"],
            "Pretraining and fine-tuning must use the same architecture")
    require(pretrain["data"].get("task") == "forecast" and
            float(pretrain["training"].get("forecast_weight", 0)) > 0 and
            float(pretrain["training"].get("classification_weight", 1)) == 0,
            "Pretraining must optimize future graph attributes only")
    require(base["data"].get("task", "classification") == "classification" and
            float(base["training"].get("classification_weight", 0)) > 0,
            "Fine-tuning must optimize classification")
    require(resolve(pretrain["data"]["manifest"]) == resolve(base["data"]["manifest"]),
            "Training stages use different manifests")
    require(input_modalities(base["model"]["name"]) == (False, True), "ODE model must consume graphs")
    plan_sha = sha256(plan_path)
    code_sha = {name: sha256(ROOT / name) for name in CODE_FILES}
    pretrain["ablation"] = {"plan_sha256": plan_sha, "stage": "pretrain", "seed": plan["seed"],
                            "code_sha256": code_sha}
    configs = {"pretrain": pretrain}
    for arm in plan["arms"]:
        config = copy.deepcopy(base)
        config["model"]["kwargs"]["rk4_step_seconds"] = float(arm["rk4_step_seconds"])
        config["ablation"] = {"plan_sha256": plan_sha, "stage": "finetune",
                              "arm": arm["name"], "initialization": arm["initialization"],
                              "seed": plan["seed"], "code_sha256": code_sha}
        configs[arm["name"]] = config
    return configs


def paths(plan, name):
    return (resolve(plan["generated_config_dir"]) / (name + ".yaml"),
            resolve(plan["run_root"]) / name)


def write_config_once(path, config):
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(config, sort_keys=True, allow_unicode=True)
    if path.is_file():
        require(path.read_text(encoding="utf-8") == content,
                "Generated config has changed; choose a new run_root: " + str(path))
    else:
        path.write_text(content, encoding="utf-8")


def command(argv):
    print("+ " + " ".join(argv), flush=True)
    subprocess.run(argv, cwd=str(ROOT), check=True)


def _run_train(name, plan, config, init_checkpoint, args):
    config_path, run_dir = paths(plan, name)
    write_config_once(config_path, config)
    if (run_dir / "completed.json").is_file():
        init_sha = sha256(init_checkpoint) if init_checkpoint is not None else None
        validate_checkpoint(name, plan, config, resolve(config["data"]["manifest"]), init_sha)
        print("+ verified completed run; skip training " + str(run_dir), flush=True)
        return
    argv = [args.python, "-m", "src.cli", "train", "--config", str(config_path),
            "--run-dir", str(run_dir), "--seed", str(plan["seed"]),
            "--workers", str(args.workers)]
    if args.device:
        argv += ["--device", args.device]
    if init_checkpoint is not None:
        require(init_checkpoint.is_file(), "Missing real pretrain checkpoint: " + str(init_checkpoint))
        argv += ["--init-checkpoint", str(init_checkpoint)]
    if (run_dir / "latest.pt").is_file():
        argv += ["--resume", str(run_dir / "latest.pt")]
    elif (run_dir / "best.pt").exists():
        raise AblationError("Run has best.pt without latest.pt: " + str(run_dir))
    command(argv)


def _evaluate(name, plan, split, args):
    _, run_dir = paths(plan, name)
    argv = [args.python, "-m", "src.cli", "evaluate", "--run-dir", str(run_dir),
            "--split", split, "--workers", str(args.workers),
            "--batch-size", str(args.batch_size)]
    if args.device:
        argv += ["--device", args.device]
    command(argv)


def ensure_full_manifest(configs):
    from src.cli import require_full_data
    import pandas as pd
    from src.data.hms import manifest_hash
    manifest_path = resolve(configs["pretrain"]["data"]["manifest"])
    require(manifest_path.is_file(), "Missing manifest")
    require_full_data(str(manifest_path), str(ROOT / "data/raw/train.csv"))
    table = pd.read_csv(manifest_path)
    valid = table.audit_reason == "ok"
    require(table.loc[valid, "split"].isin(("train", "val", "test")).all(),
            "Valid windows lack frozen splits")
    require(table.loc[~valid, "split"].eq("excluded").all(), "Invalid windows must be excluded")
    test_columns = ["window_id", "patient_id", "class_id"] + ["target_" + name for name in CLASSES]
    require(set(test_columns).issubset(table.columns), "Manifest lacks test labels")
    test = table.loc[table.split == "test", test_columns].copy()
    require(len(test) > 0 and test.window_id.is_unique, "Empty or duplicate test windows")
    require(test.patient_id.nunique() > 1, "Need multiple test patients")
    return manifest_path, table, test, manifest_hash(table)


def _checkpoint(path):
    import torch
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(str(path), **kwargs)


def validate_checkpoint(name, plan, expected, manifest_path, init_sha=None):
    config_path, run_dir = paths(plan, name)
    saved_path = run_dir / "config.json"
    best_path = run_dir / "best.pt"
    history_path = run_dir / "history.jsonl"
    for path in (config_path, saved_path, best_path, history_path,
                 run_dir / "graph_normalizer.npz", run_dir / "window_snapshot.json",
                 run_dir / "completed.json"):
        require(path.is_file(), "Missing run artifact: " + str(path))
    require(read_yaml(config_path) == expected, "Generated config differs from ablation plan: " + name)
    saved = read_json(saved_path)
    require(saved.get("smoke_subset") is False and saved.get("manifest_sha256") == sha256(manifest_path),
            "Run is smoke or bound to another manifest: " + name)
    require(saved.get("source_config_sha256") == sha256(config_path),
            "Run source config SHA differs from generated YAML: " + name)
    require({k: v for k, v in saved.items() if k not in DYNAMIC_CONFIG_KEYS} == expected,
            "Run configuration differs from planned arm: " + name)
    require(saved.get("window_snapshot_sha256") == sha256(run_dir / "window_snapshot.json"),
            "Training window snapshot changed: " + name)
    require(saved.get("init_checkpoint_sha256") == init_sha,
            "Initialization checkpoint differs: " + name)
    if init_sha is None:
        require("init_checkpoint" not in saved, "Scratch arm was initialized from a checkpoint")
    else:
        require(resolve(saved.get("init_checkpoint", "missing")) ==
                (paths(plan, "pretrain")[1] / "best.pt").resolve(),
                "Fine-tuning initialized from unexpected checkpoint")
    checkpoint = _checkpoint(best_path)
    require(checkpoint.get("config") == saved, "Checkpoint configuration differs from run JSON")
    require(type(checkpoint.get("epoch")) is int and checkpoint["epoch"] >= 0,
            "Checkpoint has no completed epoch")
    require(checkpoint.get("monitor") == ("forecast_mse" if name == "pretrain" else "kl"),
            "Checkpoint used wrong validation monitor")
    require(any(line.strip() for line in history_path.read_text(encoding="utf-8").splitlines()),
            "Run has no training history")
    completion = read_json(run_dir / "completed.json")
    require(completion.get("source_config_sha256") == sha256(config_path) and
            completion.get("manifest_sha256") == sha256(manifest_path) and
            completion.get("window_snapshot_sha256") == saved["window_snapshot_sha256"] and
            completion.get("best_checkpoint_sha256") == sha256(best_path) and
            completion.get("max_epochs") == int(expected["training"]["epochs"]),
            "Run completion marker differs from checkpoint or config: " + name)
    return {"run_dir": str(run_dir), "checkpoint": str(best_path),
            "checkpoint_sha256": sha256(best_path), "config": str(config_path),
            "config_sha256": sha256(config_path), "best_epoch": checkpoint["epoch"]}


def validate_metric(metric, test, split_hash, name, expected_coverage=1.0):
    require(metric.get("smoke_subset") is False and metric.get("synthetic", False) is False
            and metric.get("evaluation_scope") == "full_split",
            "Evaluation is smoke or partial: " + name)
    require(metric.get("split") == "test" and metric.get("model") == "odebrain_inspired_hms",
            "Wrong evaluation split/model: " + name)
    require(metric.get("split_hash") == split_hash, "Split hash differs: " + name)
    require(metric.get("n_windows") == len(test) and
            metric.get("n_patients") == int(test.patient_id.nunique()),
            "Test window/patient counts differ: " + name)
    require(isinstance(metric.get("metadata_eeg_coverage"), (int, float)) and
            math.isclose(metric["metadata_eeg_coverage"], expected_coverage,
                         rel_tol=0, abs_tol=1e-12),
            "Metric valid-recording coverage differs from audited manifest: " + name)
    for field in ("kl_window_mean", "kl_patient_equal_mean", "macro_f1"):
        value = metric.get(field)
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value), "Invalid " + field + ": " + name)
    require(metric["kl_window_mean"] >= 0, "Negative KL: " + name)


def validate_predictions(path, test, metric):
    import numpy as np
    import pandas as pd
    from src.evaluation.metrics import sample_kl
    frame = pd.read_csv(path, dtype={"window_id": str, "patient_id": str})
    required = {"window_id", "patient_id", "class_id"} | {prefix + name for prefix in ("target_", "pred_") for name in CLASSES}
    require(required.issubset(frame.columns), "Prediction CSV lacks target/prediction columns")
    require(len(frame) == len(test) and frame.window_id.is_unique, "Prediction CSV has missing/duplicate windows")
    expected = dict(zip(test.window_id.astype(str), test.patient_id.astype(str)))
    actual = dict(zip(frame.window_id.astype(str), frame.patient_id.astype(str)))
    require(actual == expected, "Prediction window/patient identity differs from manifest")
    if {"class_id"} | {"target_" + name for name in CLASSES} <= set(test.columns):
        truth = test.set_index(test.window_id.astype(str)).loc[frame.window_id.astype(str)]
        require(np.array_equal(frame.class_id.to_numpy(dtype=int), truth.class_id.to_numpy(dtype=int)),
                "Prediction hard labels differ from manifest")
        for name in CLASSES:
            require(np.allclose(frame["target_" + name].to_numpy(dtype=float),
                                truth["target_" + name].to_numpy(dtype=float), atol=1e-6, rtol=0),
                    "Prediction soft labels differ from manifest")
    target = frame[["target_" + name for name in CLASSES]].to_numpy(dtype=np.float64)
    predicted = frame[["pred_" + name for name in CLASSES]].to_numpy(dtype=np.float64)
    require(np.isfinite(target).all() and np.isfinite(predicted).all() and
            (target >= 0).all() and (predicted >= 0).all(), "Invalid test probabilities")
    require(np.allclose(target.sum(1), 1, atol=1e-5) and
            np.allclose(predicted.sum(1), 1, atol=1e-5), "Probabilities do not sum to one")
    kl = sample_kl(target, predicted)
    patient = frame.patient_id.astype(str).to_numpy()
    patient_equal = float(np.mean([kl[patient == pid].mean() for pid in np.unique(patient)]))
    require(abs(float(kl.mean()) - metric["kl_window_mean"]) <= 1e-7,
            "Metric KL differs from prediction CSV")
    require(abs(patient_equal - metric["kl_patient_equal_mean"]) <= 1e-7,
            "Metric patient KL differs from prediction CSV")


def collect(plan_path, plan, configs):
    manifest_path, table, test, split_hash = ensure_full_manifest(configs)
    expected_coverage = float(table.loc[table.audit_reason == "ok", "eeg_id"].nunique() /
                              table.eeg_id.nunique())
    pre = validate_checkpoint("pretrain", plan, configs["pretrain"], manifest_path)
    rows = []
    for arm in plan["arms"]:
        name = arm["name"]
        init_sha = pre["checkpoint_sha256"] if arm["initialization"] == "pretrain" else None
        run = validate_checkpoint(name, plan, configs[name], manifest_path, init_sha)
        run_dir = paths(plan, name)[1]
        metric_path = ROOT / "results" / run_dir.name / "test_metrics.json"
        predictions_path = ROOT / "results" / run_dir.name / "test_predictions.csv"
        require(metric_path.is_file() and predictions_path.is_file(), "Missing test evaluation for " + name)
        require(metric_path.stat().st_mtime >= Path(run["checkpoint"]).stat().st_mtime and
                predictions_path.stat().st_mtime >= Path(run["checkpoint"]).stat().st_mtime,
                "Test evaluation predates best checkpoint: " + name)
        metric = read_json(metric_path)
        validate_metric(metric, test, split_hash, name, expected_coverage)
        require(resolve(metric.get("run_dir", "missing")) == run_dir,
                "Metric refers to a different run: " + name)
        validate_predictions(predictions_path, test, metric)
        rows.append({
            "name": name, "initialization": arm["initialization"],
            "rk4_step_seconds": float(arm["rk4_step_seconds"]),
            "kl_window_mean": metric["kl_window_mean"],
            "kl_patient_equal_mean": metric["kl_patient_equal_mean"],
            "macro_f1": metric["macro_f1"], "n_windows": metric["n_windows"],
            "n_patients": metric["n_patients"], "split_hash": split_hash,
            "seed": plan["seed"], "checkpoint_sha256": run["checkpoint_sha256"],
            "checkpoint": run["checkpoint"], "config_sha256": run["config_sha256"],
            "config": run["config"], "best_epoch": run["best_epoch"],
            "init_checkpoint_sha256": init_sha,
            "metrics_file": str(metric_path), "metrics_sha256": sha256(metric_path),
            "predictions_file": str(predictions_path),
            "predictions_sha256": sha256(predictions_path),
        })
    summary = {
        "schema_version": "ode_ablation_formal_v1", "experiment_status": "formal",
        "synthetic": False, "smoke_subset": False, "split": "test",
        "split_hash": split_hash, "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path), "n_windows": int(len(test)),
        "n_patients": int(test.patient_id.nunique()),
        "plan": str(plan_path), "plan_sha256": sha256(plan_path),
        "code_sha256": configs["pretrain"]["ablation"]["code_sha256"],
        "pretrain_checkpoint": pre["checkpoint"],
        "pretrain_checkpoint_sha256": pre["checkpoint_sha256"],
        "rows": rows,
    }
    output = resolve(plan["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "split_hash": split_hash,
                      "n_windows": len(test), "n_patients": int(test.patient_id.nunique()),
                      "rows": [{"name": r["name"], "kl_window_mean": r["kl_window_mean"]} for r in rows]},
                     indent=2))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "test", "collect"))
    parser.add_argument("--plan", default="configs/ablations.yaml")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    require(args.workers >= 0 and args.batch_size > 0, "Invalid worker count or batch size")
    plan_path, plan = load_plan(args.plan)
    configs = expected_configs(plan_path, plan)
    ensure_full_manifest(configs)
    if args.stage == "train":
        _run_train("pretrain", plan, configs["pretrain"], None, args)
        pretrain_path = paths(plan, "pretrain")[1] / "best.pt"
        for arm in plan["arms"]:
            name = arm["name"]
            init = pretrain_path if arm["initialization"] == "pretrain" else None
            _run_train(name, plan, configs[name], init, args)
            _evaluate(name, plan, "val", args)
    elif args.stage == "test":
        manifest_path = resolve(configs["pretrain"]["data"]["manifest"])
        pre = validate_checkpoint("pretrain", plan, configs["pretrain"], manifest_path)
        for arm in plan["arms"]:
            init_sha = pre["checkpoint_sha256"] if arm["initialization"] == "pretrain" else None
            validate_checkpoint(arm["name"], plan, configs[arm["name"]], manifest_path, init_sha)
        for arm in plan["arms"]:
            _evaluate(arm["name"], plan, "test", args)
        collect(plan_path, plan, configs)
    else:
        collect(plan_path, plan, configs)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AblationError, ValueError, FileNotFoundError, KeyError) as exc:
        print("Ablation evidence rejected: " + str(exc), file=sys.stderr)
        sys.exit(2)
