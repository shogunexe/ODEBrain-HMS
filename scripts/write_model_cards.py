#!/usr/bin/env python3
"""Write model cards only for complete, auditable HMS classification runs.

Run from the project root with the project Python environment. Checkpoints are
loaded with torch's restricted weights_only loader; no weight is copied or
published. All requested runs are checked before any card is written.
"""

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = ("tcn", "spectrogram_cnn", "fusion", "graph_gru", "odebrain_inspired_hms")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CardError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CardError(message)


def source_path(value):
    require(isinstance(value, str) and value.strip(), "Missing path")
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    path = path.resolve()
    require(path.is_file(), "Missing file: {}".format(path))
    return path


def path_string(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_file(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CardError("Invalid JSON {}: {}".format(path, exc))
    require(isinstance(data, dict), "JSON must be an object: {}".format(path))
    return data


def reject_smoke(data, label):
    if isinstance(data, dict):
        for key, value in data.items():
            if key in ("smoke_subset", "synthetic") and value is not False:
                raise CardError("{} contains {}={}".format(label, key, value))
            if key in ("experiment_status", "provenance") and isinstance(value, str):
                if "smoke" in value.lower() or "synthetic" in value.lower():
                    raise CardError("{} contains {}={}".format(label, key, value))
            reject_smoke(value, label)
    elif isinstance(data, list):
        for value in data:
            reject_smoke(value, label)


def same_sha(actual, expected, label):
    require(isinstance(expected, str) and HEX64.fullmatch(expected) and actual == expected,
            "{} SHA256 mismatch".format(label))


def positive_int(value, label):
    require(type(value) is int and value > 0, "{} must be a positive integer".format(label))
    return value


def finite(value, label, minimum=None):
    require(type(value) in (float, int) and math.isfinite(value),
            "{} must be a finite number".format(label))
    if minimum is not None:
        require(value >= minimum, "{} must be >= {}".format(label, minimum))
    return value


def manifest_context(manifest_name, metadata_name, summary_name):
    import pandas as pd
    from src.data.hms import manifest_hash
    manifest_path = source_path(manifest_name)
    metadata_path = source_path(metadata_name)
    summary_path = source_path(summary_name)
    manifest = pd.read_csv(manifest_path, dtype={"window_id": str, "patient_id": str})
    metadata = pd.read_csv(metadata_path, usecols=["eeg_id"])
    summary = json_file(summary_path)
    required = {"window_id", "patient_id", "eeg_id", "split", "audit_reason",
                "file_exists", "missing_fraction", "raw_recording_seconds"}
    require(required.issubset(manifest.columns), "Manifest lacks full audit/split fields")
    require(len(manifest) == len(metadata), "Manifest does not cover full metadata")
    require(manifest.eeg_id.value_counts().sort_index().equals(
            metadata.eeg_id.value_counts().sort_index()),
            "Manifest EEG IDs differ from metadata")
    require(manifest.file_exists.fillna(False).all(), "EEG files are missing")
    require(not manifest.audit_reason.astype(str).str.contains(
            "missing_eeg_file|unreadable_eeg_file").any(), "EEG audit is incomplete")
    valid = manifest.audit_reason == "ok"
    require(valid.any() and not manifest.loc[valid, ["missing_fraction",
            "raw_recording_seconds"]].isna().any().any(), "Quality audit is incomplete")
    require(manifest.window_id.is_unique, "Duplicate window IDs")
    require(manifest.loc[valid, "split"].isin(["train", "val", "test"]).all(),
            "Valid window has no train/val/test split")
    require(manifest.loc[~valid, "split"].eq("excluded").all(),
            "Invalid window was included")
    groups = {name: set(manifest.loc[manifest.split == name, "patient_id"])
              for name in ("train", "val", "test")}
    require(not (groups["train"] & groups["val"] or groups["train"] & groups["test"]
                 or groups["val"] & groups["test"]), "Patient leakage across splits")
    split_hash = manifest_hash(manifest)
    same_sha(split_hash, summary.get("sha256"), "Split summary")
    counts = {}
    ids = {}
    for name in ("train", "val", "test"):
        section = manifest[manifest.split == name]
        count = (int(len(section)), int(section.patient_id.nunique()))
        require(count[0] > 0 and count[1] > 0, "Empty {} split".format(name))
        require(summary.get("split_window_counts", {}).get(name) == count[0]
                and summary.get("split_patient_counts", {}).get(name) == count[1],
                "Split summary count mismatch for {}".format(name))
        counts[name] = count
        ids[name] = set(zip(section.window_id.astype(str), section.patient_id.astype(str)))
    valid_eeg_coverage = float(manifest.loc[valid, "eeg_id"].nunique() /
                               manifest.eeg_id.nunique())
    return {"path": manifest_path, "sha256": sha256(manifest_path),
            "split_hash": split_hash, "summary_path": summary_path,
            "summary_sha256": sha256(summary_path), "metadata_path": metadata_path,
            "metadata_sha256": sha256(metadata_path), "counts": counts, "ids": ids,
            "valid_eeg_coverage": valid_eeg_coverage}


def check_prediction_file(path, expected_ids):
    import pandas as pd
    frame = pd.read_csv(path, usecols=["window_id", "patient_id"],
                        dtype={"window_id": str, "patient_id": str})
    actual = list(zip(frame.window_id, frame.patient_id))
    require(len(actual) == len(expected_ids) and len(set(actual)) == len(actual)
            and set(actual) == expected_ids,
            "Prediction file does not cover exact frozen split: {}".format(path))


def check_metric(path, model_name, run_dir, split, context):
    data = json_file(path)
    reject_smoke(data, "{} {} metrics".format(model_name, split))
    require(data.get("smoke_subset") is False and data.get("split") == split,
            "{} {} metrics are not formal".format(model_name, split))
    require(data.get("evaluation_scope") == "full_split", "Metrics are not full split")
    require(data.get("model") == model_name, "Metric model identity mismatch")
    same_sha(context["split_hash"], data.get("split_hash"), "Metric split")
    require(isinstance(data.get("metadata_eeg_coverage"), (int, float)) and
            math.isclose(data["metadata_eeg_coverage"],
                         context["valid_eeg_coverage"], rel_tol=0, abs_tol=1e-12),
            "Metric valid-recording coverage differs from audited manifest")
    require(data.get("n_windows") == context["counts"][split][0]
            and data.get("n_patients") == context["counts"][split][1],
            "{} {} count mismatch".format(model_name, split))
    reported_run = Path(data.get("run_dir", ""))
    reported_run = reported_run if reported_run.is_absolute() else ROOT / reported_run
    require(reported_run.resolve() == run_dir, "Metric points to another run")
    for key in ("kl_window_mean", "kl_patient_equal_mean", "macro_f1"):
        finite(data.get(key), "{} {}".format(split, key), 0)
    ci = data.get("kl_patient_bootstrap", {}).get("ci95")
    require(isinstance(ci, list) and len(ci) == 2, "Patient bootstrap CI missing")
    finite(ci[0], "CI lower", 0); finite(ci[1], "CI upper", 0)
    require(ci[0] <= ci[1], "Invalid patient bootstrap interval")
    predictions = source_path(str(path.parent / (split + "_predictions.csv")))
    check_prediction_file(predictions, context["ids"][split])
    return {"path": path_string(path), "sha256": sha256(path),
            "predictions": path_string(predictions),
            "predictions_sha256": sha256(predictions),
            "n_windows": data["n_windows"], "n_patients": data["n_patients"],
            "kl_window_mean": data["kl_window_mean"],
            "kl_patient_equal_mean": data["kl_patient_equal_mean"],
            "macro_f1": data["macro_f1"], "patient_bootstrap_ci95": ci}


def check_run(model_name, run_name, context, results_root):
    import numpy as np
    import torch
    from src.cli import make_model
    from src.data.hms import CLASSES, ELECTRODES, SAMPLE_RATE, WINDOW_SECONDS
    run_dir = Path(run_name)
    run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
    run_dir = run_dir.resolve()
    require(run_dir.is_dir(), "Missing run directory: {}".format(run_dir))
    require(run_dir.parent.resolve() == (ROOT / "artifacts/runs").resolve(),
            "Run must be a direct child of artifacts/runs")
    config_path = source_path(str(run_dir / "config.json"))
    completion_path = source_path(str(run_dir / "completed.json"))
    checkpoint_path = source_path(str(run_dir / "best.pt"))
    normalizer_path = source_path(str(run_dir / "graph_normalizer.npz"))
    snapshot_path = source_path(str(run_dir / "window_snapshot.json"))
    config = json_file(config_path)
    completion = json_file(completion_path)
    reject_smoke(config, "{} config".format(model_name))
    reject_smoke(completion, "{} completion".format(model_name))
    require(config.get("smoke_subset") is False, "Run is not explicitly formal")
    require(config.get("model", {}).get("name") == model_name,
            "Run model name mismatch")
    require(config.get("data", {}).get("task", "classification") == "classification",
            "Forecast-only run cannot have a classification model card")
    same_sha(context["sha256"], config.get("manifest_sha256"), "Training manifest")
    same_sha(context["sha256"], completion.get("manifest_sha256"), "Completion manifest")
    same_sha(sha256(checkpoint_path), completion.get("best_checkpoint_sha256"),
             "Completed checkpoint")
    same_sha(sha256(snapshot_path), config.get("window_snapshot_sha256"),
             "Training window snapshot")
    same_sha(sha256(snapshot_path), completion.get("window_snapshot_sha256"),
             "Completion window snapshot")
    require(completion.get("source_config_sha256") == config.get("source_config_sha256"),
            "Source config hash differs at completion")
    source_config = source_path("configs/models/{}.yaml".format(model_name))
    same_sha(sha256(source_config), config.get("source_config_sha256"),
             "Source YAML config")
    snapshot = json_file(snapshot_path)
    for split in ("train", "val"):
        ids = [str(value) for value in snapshot.get(split, [])]
        expected = {window for window, _ in context["ids"][split]}
        require(len(ids) == len(expected) and len(set(ids)) == len(ids)
                and set(ids) == expected,
                "Frozen {} windows differ from manifest".format(split))
    with np.load(str(normalizer_path), allow_pickle=False) as normalizer:
        mean, std = normalizer["mean"], normalizer["std"]
        require(mean.shape == (5,) and std.shape == (5,) and np.isfinite(mean).all()
                and np.isfinite(std).all() and (std > 0).all(),
                "Graph normalizer is invalid")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    require(isinstance(checkpoint, dict) and checkpoint.get("config") == config,
            "Checkpoint config differs from run config")
    require(checkpoint.get("monitor") == "kl" and type(checkpoint.get("epoch")) is int
            and checkpoint["epoch"] >= 0, "Checkpoint was not selected by classification KL")
    finite(checkpoint.get("best_val_score"), "Best validation KL", 0)
    best_epoch = checkpoint["epoch"]
    best_validation_kl = checkpoint["best_val_score"]
    model = make_model(config)
    model.load_state_dict(checkpoint.get("model"), strict=True)
    del model, checkpoint
    lineage = None
    if config.get("init_checkpoint"):
        init_path = source_path(config["init_checkpoint"])
        same_sha(sha256(init_path), config.get("init_checkpoint_sha256"),
                 "Pretraining checkpoint")
        init_checkpoint = torch.load(str(init_path), map_location="cpu", weights_only=True)
        init_config = init_checkpoint.get("config", {})
        reject_smoke(init_config, "Pretraining config")
        require(init_config.get("smoke_subset") is False
                and init_config.get("manifest_sha256") == context["sha256"],
                "Pretraining source is not formal on the same manifest")
        lineage = {"path": path_string(init_path), "sha256": sha256(init_path),
                   "model": init_config.get("model", {}).get("name")}
    result_dir = Path(results_root)
    result_dir = result_dir if result_dir.is_absolute() else ROOT / result_dir
    result_dir = result_dir.resolve() / run_dir.name
    validation = check_metric(source_path(str(result_dir / "val_metrics.json")),
                              model_name, run_dir, "val", context)
    test = check_metric(source_path(str(result_dir / "test_metrics.json")),
                        model_name, run_dir, "test", context)
    kwargs = config["model"].get("kwargs", {})
    require(kwargs.get("num_classes") == 6, "Model is not six-class")
    modalities = {"raw_eeg": model_name in ("tcn", "fusion", "odebrain_inspired_hms"),
                  "spectrogram": model_name in ("spectrogram_cnn", "fusion"),
                  "graph": model_name in ("graph_gru", "odebrain_inspired_hms")}
    return {"schema_version": "hms_model_card_v1", "model": model_name,
            "status": "formal", "task": "offline_six_class_eeg_classification",
            "run_dir": path_string(run_dir), "config": path_string(config_path),
            "config_sha256": sha256(config_path),
            "source_config": path_string(source_config),
            "source_config_sha256": sha256(source_config),
            "checkpoint": path_string(checkpoint_path),
            "weights_sha256": sha256(checkpoint_path),
            "completion": path_string(completion_path),
            "completion_sha256": sha256(completion_path),
            "window_snapshot": path_string(snapshot_path),
            "window_snapshot_sha256": sha256(snapshot_path),
            "graph_normalizer": path_string(normalizer_path),
            "graph_normalizer_sha256": sha256(normalizer_path),
            "manifest": path_string(context["path"]),
            "manifest_file_sha256": context["sha256"],
            "split_summary": path_string(context["summary_path"]),
            "split_summary_sha256": context["summary_sha256"],
            "split_hash": context["split_hash"],
            "metadata": path_string(context["metadata_path"]),
            "metadata_sha256": context["metadata_sha256"],
            "split_counts": {name: {"windows": value[0], "patients": value[1]}
                             for name, value in context["counts"].items()},
            "input_modalities": modalities,
            "input_channels": list(ELECTRODES),
            "sample_rate_hz": SAMPLE_RATE, "window_seconds": WINDOW_SECONDS,
            "label_region_seconds": [20, 30],
            "classes": list(CLASSES),
            "model_kwargs": kwargs, "training": config["training"],
            "pretraining": lineage, "best_epoch_zero_indexed": best_epoch,
            "best_validation_kl_during_training": best_validation_kl,
            "evaluation": {"val": validation, "test": test}}


def card_markdown(card):
    if card.get("training_protocol") == "train_val_fixed_epoch_refit":
        return refit_card_markdown(card)
    name = card["model"]
    modes = card["input_modalities"]
    count = card["split_counts"]
    train_command = (".venv/bin/python -m src.cli train --config {} --run-dir {}"
                     .format(card["source_config"], card["run_dir"]))
    if card["pretraining"]:
        train_command += " --init-checkpoint {}".format(card["pretraining"]["path"])
    evaluation_commands = [".venv/bin/python -m src.cli evaluate --run-dir {} --split {}"
                           .format(card["run_dir"], split) for split in ("val", "test")]
    lines = ["# {} — HMS 模型卡".format(name), "", "## 用途与身份", "",
             "- 状态：正式完整数据评估；六类离线 EEG 活动分类。",
             "- 模型版本：`{}`（checkpoint SHA256）。".format(card["weights_sha256"]),
             "- 输入来源：HMS 训练集；分类使用 50 秒上下文，专家标注对应中央 20–30 秒。",
             "- 本实现的 ODE 模型属于 ODEBrain-inspired HMS 改造；不代表官方模型已在 HMS 上验证。" if name == "odebrain_inspired_hms" else
             "- 模型属于本项目基线，结果只对应此处固定患者划分。", "",
             "## 输入与预处理", "",
             "- 采样率：200 Hz；19 个 EEG 导联（不含 EKG）：{}。".format(", ".join(card["input_channels"])),
             "- 通道：原始 EEG={}；同窗 STFT 频谱={}；1 秒图帧={}。".format(
                 modes["raw_eeg"], modes["spectrogram"], modes["graph"]),
             "- 以 `eeg_label_offset_seconds` 裁出 50 秒；短内部缺失最多插值 20 采样点，长缺失保留掩码并填零；超过 5% 缺失或恒定导联的窗口不进入有效集。",
             "- 频谱分支：STFT 256 采样点窗、64 采样点步长，幅值 `log1p`。图分支：每导联 5 个频带的 1 秒节点属性，相关系数作为统计关联代理边；节点标准化参数只从训练患者拟合。",
             "- 训练图归一化文件 SHA256：`{}`。".format(card["graph_normalizer_sha256"]),
             "", "## 数据与划分", "",
             "- 患者级互斥划分哈希：`{}`。".format(card["split_hash"]),
             "- Manifest 文件 SHA256：`{}`。".format(card["manifest_file_sha256"]),
             "- 训练：{} 窗口 / {} 患者；验证：{} 窗口 / {} 患者；测试：{} 窗口 / {} 患者。".format(
                 count["train"]["windows"], count["train"]["patients"],
                 count["val"]["windows"], count["val"]["patients"],
                 count["test"]["windows"], count["test"]["patients"]),
             "- 软标签由六类专家票数归一化；测试集按固定协议独立评估。",
             "", "## 训练与评估", "",
             "- 训练配置：`{}`；配置 SHA256：`{}`。".format(
                 card["config"], card["config_sha256"]),
             "- 训练参数：`{}`。".format(json.dumps(card["training"], ensure_ascii=False, sort_keys=True)),
             "- 主指标：KL(target || prediction)，逐窗口平均；患者等权 KL 与 macro F1 为补充。"]
    if card["pretraining"]:
        lines += ["- 初始化权重来源：`{}`；SHA256：`{}`。".format(
            card["pretraining"]["path"], card["pretraining"]["sha256"])]
    lines += ["", "| 划分 | 窗口 | 患者 | 窗口 KL ↓ | 患者等权 KL ↓ | Macro F1 |",
              "|---|---:|---:|---:|---:|---:|"]
    for split in ("val", "test"):
        metric = card["evaluation"][split]
        lines.append("| {} | {} | {} | {:.6f} | {:.6f} | {:.6f} |".format(
            split, metric["n_windows"], metric["n_patients"],
            metric["kl_window_mean"], metric["kl_patient_equal_mean"], metric["macro_f1"]))
    lines += ["", "- 验证和测试的逐窗口预测 CSV、指标 JSON 均与冻结划分逐 ID 核对；各文件 SHA256 见同名 JSON 模型卡。",
              "", "## 限制", "",
              "- 只用于离线研究展示；不能用于临床诊断、实时发作预警或患者治疗。",
              "- 专家票数存在分歧；脑电相关边与潜在 ODE 轨迹不构成生理因果解释。",
              "- 指标只适用于当前 HMS 划分、预处理和文件哈希；跨中心/设备/人群泛化未经本模型卡验证。",
              "", "## 本地复现", "", "```bash", train_command,
              *evaluation_commands, "```", "",
              "权重仅位于本地 `{}`；本脚本不发布权重或生成下载链接。".format(card["checkpoint"]), ""]
    return "\n".join(lines)


def verified_refit_comparison(runs, comparison_path, results_root, context):
    """Recompute the five-run summary before trusting a published comparison."""
    from scripts.refit_final import summarize_refits
    require(set(runs) == set(MODELS), "Specify exactly one run for each of five deep models")
    resolved = {name: str(Path(path).resolve() if Path(path).is_absolute()
                          else (ROOT / path).resolve()) for name, path in runs.items()}
    require(len(set(resolved.values())) == len(MODELS), "Refit runs must be distinct")
    comparison_path = source_path(comparison_path)
    reported = json_file(comparison_path)
    reject_smoke(reported, "Refit comparison")
    results_root = (Path(results_root) if Path(results_root).is_absolute()
                    else ROOT / results_root).resolve()
    with tempfile.TemporaryDirectory(prefix="hms_refit_card_verify_") as temporary:
        computed_path = Path(temporary) / "verified.json"
        summarize_refits(resolved, computed_path, results_root)
        computed = json_file(computed_path)
    require(reported == computed,
            "Refit comparison differs from independently recomputed test evidence")
    require(computed["split_hash"] == context["split_hash"] and
            computed["manifest_sha256"] == context["sha256"] and
            computed["n_windows"] == context["counts"]["test"][0] and
            computed["n_patients"] == context["counts"]["test"][1],
            "Refit comparison does not match the audited test manifest")
    return reported, comparison_path


def check_refit_run(model_name, run_name, row, context, results_root,
                    comparison_path):
    """Package a previously reverified final refit without mixing source metrics."""
    from src.data.hms import CLASSES, ELECTRODES, SAMPLE_RATE, WINDOW_SECONDS
    run_dir = (Path(run_name) if Path(run_name).is_absolute() else ROOT / run_name).resolve()
    require(run_dir == Path(row["run_dir"]).resolve() and row["name"] == model_name,
            "Refit comparison row points to another model or run")
    files = {name: source_path(str(run_dir / name)) for name in (
        "config.json", "completed.json", "window_snapshot.json",
        "validation_record.json", "normalizer_record.json", "graph_normalizer.npz",
        "final.pt")}
    config = json_file(files["config.json"])
    completed = json_file(files["completed.json"])
    selection = json_file(files["validation_record.json"])
    require(config.get("schema_version") == "hms_final_refit_v1" and
            config.get("smoke_subset") is False and
            config.get("model", {}).get("name") == model_name and
            config.get("refit", {}).get("task") == "classification" and
            config.get("refit", {}).get("training_split") == "train+val" and
            config.get("refit", {}).get("test_used_for_selection") is False and
            completed.get("experiment_status") == "formal_refit",
            "Refit card requires formal train+val classification evidence")
    same_sha(sha256(files["final.pt"]), row["checkpoint_sha256"], "Refit checkpoint")
    same_sha(sha256(files["final.pt"]), completed.get("final_checkpoint_sha256"),
             "Refit completion checkpoint")
    require(selection.get("test_used_for_selection") is False and
            selection.get("selection_split") == "val" and
            selection.get("fixed_refit_epochs") == config["refit"]["fixed_epochs"] and
            selection.get("source_best_checkpoint_sha256") ==
            config["refit"]["source_best_checkpoint_sha256"],
            "Refit selection record differs from final protocol")
    result_root = (Path(results_root) if Path(results_root).is_absolute()
                   else ROOT / results_root).resolve()
    result_dir = result_root / run_dir.name
    metric_path = source_path(str(result_dir / "test_metrics.json"))
    prediction_path = source_path(str(result_dir / "test_predictions.csv"))
    same_sha(sha256(metric_path), row["test_metrics_sha256"], "Refit test metrics")
    same_sha(sha256(prediction_path), row["test_predictions_sha256"],
             "Refit test predictions")
    metric = json_file(metric_path)
    reject_smoke(metric, "Refit test metrics")
    require(metric.get("training_protocol") == "train_val_fixed_epoch_refit" and
            metric.get("test_used_for_selection") is False and
            metric.get("split") == "test", "Refit test metric protocol mismatch")
    for key in ("kl_window_mean", "kl_patient_equal_mean", "macro_f1"):
        finite(metric.get(key), "Refit test " + key, 0)
    ci = metric.get("kl_patient_bootstrap", {}).get("ci95")
    require(isinstance(ci, list) and len(ci) == 2 and
            all(type(x) in (int, float) and math.isfinite(x) for x in ci) and
            0 <= ci[0] <= ci[1], "Refit patient bootstrap CI missing or invalid")
    source_run = Path(selection["source_run"]).resolve()
    source_best = source_path(selection["source_best_checkpoint"])
    same_sha(sha256(source_best), selection["source_best_checkpoint_sha256"],
             "Validation-selected source checkpoint")
    require(source_best.parent == source_run and
            source_run == Path(config["refit"]["source_run"]).resolve(),
            "Refit source run changed")
    init_run = config["refit"].get("init_refit_run")
    pretraining = None
    if init_run:
        init_path = source_path(str(Path(init_run) / "final.pt"))
        same_sha(sha256(init_path), config["refit"].get("init_refit_checkpoint_sha256"),
                 "Refitted forecast pretraining checkpoint")
        pretraining = {"run_dir": path_string(Path(init_run)),
                       "checkpoint": path_string(init_path), "sha256": sha256(init_path),
                       "task": "forecast", "training_split": "train+val"}
    kwargs = config["model"].get("kwargs", {})
    require(kwargs.get("num_classes") == 6, "Refit model is not six-class")
    modalities = {"raw_eeg": model_name in ("tcn", "fusion", "odebrain_inspired_hms"),
                  "spectrogram": model_name in ("spectrogram_cnn", "fusion"),
                  "graph": model_name in ("graph_gru", "odebrain_inspired_hms")}
    return {"schema_version": "hms_model_card_refit_v1", "model": model_name,
            "status": "formal", "task": "offline_six_class_eeg_classification",
            "training_protocol": "train_val_fixed_epoch_refit",
            "run_dir": path_string(run_dir), "checkpoint": path_string(files["final.pt"]),
            "weights_sha256": sha256(files["final.pt"]),
            "config": path_string(files["config.json"]),
            "config_sha256": sha256(files["config.json"]),
            "completion": path_string(files["completed.json"]),
            "completion_sha256": sha256(files["completed.json"]),
            "window_snapshot": path_string(files["window_snapshot.json"]),
            "window_snapshot_sha256": sha256(files["window_snapshot.json"]),
            "validation_record": path_string(files["validation_record.json"]),
            "validation_record_sha256": sha256(files["validation_record.json"]),
            "graph_normalizer": path_string(files["graph_normalizer.npz"]),
            "graph_normalizer_sha256": sha256(files["graph_normalizer.npz"]),
            "normalizer_record": path_string(files["normalizer_record.json"]),
            "normalizer_record_sha256": sha256(files["normalizer_record.json"]),
            "comparison": path_string(comparison_path),
            "comparison_sha256": sha256(comparison_path),
            "source_run": path_string(source_run),
            "source_checkpoint": path_string(source_best),
            "source_checkpoint_sha256": sha256(source_best),
            "selection": selection,
            "fixed_refit_epochs": config["refit"]["fixed_epochs"],
            "refit_seed": config["refit"]["seed"],
            "pretraining": pretraining,
            "manifest": path_string(context["path"]),
            "manifest_file_sha256": context["sha256"],
            "split_summary": path_string(context["summary_path"]),
            "split_summary_sha256": context["summary_sha256"],
            "split_hash": context["split_hash"],
            "metadata": path_string(context["metadata_path"]),
            "metadata_sha256": context["metadata_sha256"],
            "split_counts": {name: {"windows": value[0], "patients": value[1]}
                             for name, value in context["counts"].items()},
            "input_modalities": modalities, "input_channels": list(ELECTRODES),
            "sample_rate_hz": SAMPLE_RATE, "window_seconds": WINDOW_SECONDS,
            "label_region_seconds": [20, 30], "classes": list(CLASSES),
            "model_kwargs": kwargs, "training": config["training"],
            "evaluation": {"test": {"path": path_string(metric_path),
                                     "sha256": sha256(metric_path),
                                     "predictions": path_string(prediction_path),
                                     "predictions_sha256": sha256(prediction_path),
                                     "n_windows": metric["n_windows"],
                                     "n_patients": metric["n_patients"],
                                     "kl_window_mean": metric["kl_window_mean"],
                                     "kl_patient_equal_mean": metric["kl_patient_equal_mean"],
                                     "macro_f1": metric["macro_f1"],
                                     "patient_bootstrap_ci95": ci}}}


def refit_card_markdown(card):
    counts = card["split_counts"]
    metric = card["evaluation"]["test"]
    selection = card["selection"]
    lines = ["# {} — HMS 最终重训模型卡".format(card["model"]), "",
             "## 用途与身份", "",
             "- 状态：正式完整数据评估；六类离线 EEG 活动分类。",
             "- 模型版本：`{}`（最终 checkpoint SHA256）。".format(card["weights_sha256"]),
             "- 输入来源：HMS 训练集；分类使用 50 秒上下文，专家标注对应中央 20–30 秒。",
             "- 本实现的 ODE 模型属于 ODEBrain-inspired HMS 改造；不代表官方模型已在 HMS 上验证。" if card["model"] == "odebrain_inspired_hms" else
             "- 模型属于本项目基线，结果只对应此处固定患者划分。", "",
             "## 数据与划分", "",
             "- 患者级互斥划分哈希：`{}`。".format(card["split_hash"]),
             "- Manifest 文件 SHA256：`{}`。".format(card["manifest_file_sha256"]),
             "- 初训训练：{} 窗口 / {} 患者；初训验证：{} 窗口 / {} 患者；最终重训合计：{} 窗口 / {} 患者；独立测试：{} 窗口 / {} 患者。".format(
                 counts["train"]["windows"], counts["train"]["patients"],
                 counts["val"]["windows"], counts["val"]["patients"],
                 counts["train"]["windows"] + counts["val"]["windows"],
                 counts["train"]["patients"] + counts["val"]["patients"],
                 counts["test"]["windows"], counts["test"]["patients"]),
             "- 软标签由六类专家票数归一化；测试患者未进入轮数选择或最终重训。", "",
             "## 输入与预处理", "",
             "- 采样率：{} Hz；19 个 EEG 导联（不含 EKG）：{}。".format(
                 card["sample_rate_hz"], ", ".join(card["input_channels"])),
             "- 通道：原始 EEG={}；同窗 STFT 频谱={}；1 秒图帧={}。".format(
                 *[card["input_modalities"][key] for key in ("raw_eeg", "spectrogram", "graph")]),
             "- 以 `eeg_label_offset_seconds` 裁出 50 秒；短内部缺失最多插值 20 采样点，长缺失保留掩码并填零；超过 5% 缺失或恒定导联的窗口不进入有效集。",
             "- 频谱分支：STFT 256 采样点窗、64 采样点步长，幅值 `log1p`。图分支：每导联 5 个频带的 1 秒节点属性，相关系数作为统计关联代理边。",
             ("- 最终图归一化参数仅从训练+验证患者拟合；文件 SHA256：`{}`。" if card["input_modalities"]["graph"] else
              "- 此模型不使用图输入；恒等图归一化占位文件 SHA256：`{}`。")
             .format(card["graph_normalizer_sha256"]),
             "", "## 训练与评估", "",
             "- 初训从验证集选择第 {} 轮（从 1 计数），依据 `{}`；源 checkpoint SHA256：`{}`。".format(
                 card["fixed_refit_epochs"], selection["selection_monitor"],
                 card["source_checkpoint_sha256"]),
             "- 最终模型在训练+验证患者上固定训练 {} 轮；不在重训阶段重新选择轮数。".format(
                 card["fixed_refit_epochs"]),
             "- 重训随机种子：{}；配置：`{}`；配置 SHA256：`{}`。".format(
                 card["refit_seed"], card["config"], card["config_sha256"]),
             "- 训练参数：`{}`。".format(json.dumps(card["training"], ensure_ascii=False, sort_keys=True)),
             "- 主指标：KL(target || prediction)，逐窗口平均；患者等权 KL 与 macro F1 为补充。"]
    if card["pretraining"]:
        lines += ["- 初始化权重来自同样以训练+验证患者重训的预测任务：`{}`；SHA256：`{}`。".format(
            card["pretraining"]["checkpoint"], card["pretraining"]["sha256"])]
    lines += ["", "| 划分 | 窗口 | 患者 | 窗口 KL ↓ | 患者等权 KL ↓ | Macro F1 |",
              "|---|---:|---:|---:|---:|---:|",
              "| test | {} | {} | {:.6f} | {:.6f} | {:.6f} |".format(
                  metric["n_windows"], metric["n_patients"],
                  metric["kl_window_mean"], metric["kl_patient_equal_mean"],
                  metric["macro_f1"]), "",
              "- 测试预测逐窗口覆盖冻结测试集；指标与预测已由 `scripts/refit_final.py summarize` 重算核对。",
              "- 五模型正式对比表：`{}`；SHA256：`{}`。".format(
                  card["comparison"], card["comparison_sha256"]),
              "- 结果只对应此处固定患者划分、预处理和文件哈希；跨中心、设备、人群泛化未经验证。",
              "", "## 限制", "",
              "- 只用于离线研究展示；不能用于临床诊断、实时发作预警或患者治疗。",
              "- 专家票数存在分歧；脑电相关边与潜在 ODE 轨迹不构成生理因果解释。",
              "", "## 本地复现", "", "```bash",
              ".venv/bin/python scripts/refit_final.py train --source-run {} --run-dir artifacts/refit/reproduce_{}{}".format(
                  card["source_run"], card["model"],
                  " --init-refit-run " + card["pretraining"]["run_dir"] if card["pretraining"] else ""),
              ".venv/bin/python scripts/refit_final.py evaluate --run-dir artifacts/refit/reproduce_{}".format(card["model"]),
              "```", "",
              "权重仅位于本地 `{}`；本脚本不发布权重或生成下载链接。".format(card["checkpoint"]), ""]
    return "\n".join(lines)


def write_refit_cards(runs, manifest, metadata, split_summary, results_root,
                      output_dir, comparison):
    context = manifest_context(manifest, metadata, split_summary)
    summary, comparison_path = verified_refit_comparison(runs, comparison,
                                                         results_root, context)
    rows = {row["name"]: row for row in summary["rows"]}
    require(set(rows) == set(MODELS) and len(summary["rows"]) == len(MODELS),
            "Refit comparison does not contain five unique models")
    cards = {name: check_refit_run(name, runs[name], rows[name], context,
                                   results_root, comparison_path) for name in MODELS}
    return _write_card_files(cards, output_dir)


def _write_card_files(cards, output_dir):
    output = Path(output_dir)
    output = output if output.is_absolute() else ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    for name, card in cards.items():
        markdown = output / (name + ".md")
        json_path = output / (name + ".json")
        temp_md = markdown.with_suffix(".md.tmp")
        temp_json = json_path.with_suffix(".json.tmp")
        temp_md.write_text(card_markdown(card), encoding="utf-8")
        temp_json.write_text(json.dumps(card, ensure_ascii=False, indent=2, allow_nan=False),
                             encoding="utf-8")
        temp_md.replace(markdown)
        temp_json.replace(json_path)
    return output


def write_cards(runs, manifest, metadata, split_summary, results_root, output_dir):
    require(set(runs) == set(MODELS), "Specify exactly one run for each of five deep models")
    context = manifest_context(manifest, metadata, split_summary)
    cards = {name: check_run(name, runs[name], context, results_root) for name in MODELS}
    return _write_card_files(cards, output_dir)


def parse_runs(items):
    runs = {}
    for item in items:
        require("=" in item, "--run must be MODEL=PATH")
        name, path = item.split("=", 1)
        require(name in MODELS and name not in runs and path,
                "Unknown or repeated model: {}".format(name))
        runs[name] = path
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="MODEL=PATH")
    parser.add_argument("--protocol", choices=("initial", "refit"), default="initial",
                        help="Refit uses final.pt and an independently reverified comparison")
    parser.add_argument("--manifest", default="data/curated/split_manifest.csv")
    parser.add_argument("--metadata", default="data/raw/train.csv")
    parser.add_argument("--split-summary", default="data/curated/split_manifest.json")
    parser.add_argument("--results-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--comparison", default="results/refit/final_comparison.json")
    args = parser.parse_args()
    try:
        runs = parse_runs(args.run)
        if args.protocol == "refit":
            print(write_refit_cards(runs, args.manifest, args.metadata,
                                    args.split_summary,
                                    args.results_root or "results/refit",
                                    args.output_dir or "docs/model_cards_refit",
                                    args.comparison))
        else:
            print(write_cards(runs, args.manifest, args.metadata,
                              args.split_summary, args.results_root or "results",
                              args.output_dir or "docs/model_cards"))
    except (CardError, KeyError, TypeError, RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(2, "Formal model cards rejected: {}\n".format(exc))


if __name__ == "__main__":
    main()
