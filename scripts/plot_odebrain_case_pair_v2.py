"""Auditable correct/error HMS examples from the revised formal ODE model."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.dataset import HMSWindowDataset
from src.data.hms import CLASSES, ELECTRODES
from src.dynamics.analysis import TrainOnlyPCA, mean_valid_nodes
from src.dynamics.export import load_trained_model


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_cases(predictions):
    pred_cols = ["pred_" + name for name in CLASSES]
    target_cols = ["target_" + name for name in CLASSES]
    if not {"window_id", "class_id", *pred_cols, *target_cols}.issubset(predictions.columns):
        raise ValueError("Formal prediction CSV lacks class probabilities or labels")
    table = predictions.copy()
    table["predicted_class"] = table[pred_cols].to_numpy().argmax(axis=1)
    table["expert_fraction"] = table[target_cols].to_numpy().max(axis=1)
    table["correct"] = table.predicted_class == table.class_id
    chosen = []
    for correct in (True, False):
        subset = table.loc[table.correct == correct].copy()
        if subset.empty:
            raise ValueError("Need at least one correct and one incorrect held-out case")
        high = subset.loc[subset.expert_fraction >= .75]
        if not high.empty:
            subset = high
        subset = subset.sort_values("window_id")
        chosen.append(subset.iloc[0])
    return chosen


def plot(checkpoint, predictions, projector_path, manifest, output, device="cuda"):
    checkpoint, predictions, projector_path = map(Path, (checkpoint, predictions, projector_path))
    output = Path(output)
    model, saved = load_trained_model(checkpoint, device=device)
    if (model.architecture != "github_trajectory_v2" or
            saved.get("config", {}).get("smoke_subset") is not False or
            saved.get("config", {}).get("schema_version") != "hms_final_refit_v1"):
        raise ValueError("Case figures require the final formal ODEBrain v2 refit")
    if saved["config"]["manifest_sha256"] != sha256(manifest):
        raise ValueError("Checkpoint and manifest disagree")
    projector = TrainOnlyPCA.load(projector_path)
    if projector.model_id != "odebrain_github_v2_" + sha256(checkpoint)[:16]:
        raise ValueError("PCA belongs to another model checkpoint")
    with np.load(checkpoint.parent / "graph_normalizer.npz", allow_pickle=False) as values:
        mean, std = values["mean"], values["std"]
    dataset = HMSWindowDataset(manifest, "test", graph_mean=mean, graph_std=std,
                               include_spectrogram=False,
                               graph_cache_dir=saved["config"]["data"]["graph_cache_dir"])
    by_id = {str(window_id): index for index, window_id in enumerate(dataset.rows.window_id)}
    chosen = select_cases(pd.read_csv(predictions))
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for kind, row in zip(("correct", "error"), chosen):
        window_id = str(row.window_id)
        if window_id not in by_id:
            raise ValueError("Selected case is absent from audited test data")
        sample = dataset[by_id[window_id]]
        tensors = {key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor)
                   else value for key, value in sample.items()}
        with torch.no_grad():
            logits, dynamics = model(tensors, return_dynamics=True)
        probabilities = torch.softmax(logits, -1)[0].detach().cpu().numpy()
        expected = row[["pred_" + name for name in CLASSES]].to_numpy(dtype=float)
        if not np.allclose(probabilities, expected, rtol=1e-5, atol=1e-5):
            raise ValueError("Case model inference differs from frozen test predictions")
        mask = dynamics["observed_node_mask"][0].detach().cpu().numpy().astype(bool)
        states = dynamics["trajectory"][0].detach().cpu().numpy()
        derivatives = dynamics["vector_field"][0].detach().cpu().numpy()
        path = projector.transform(mean_valid_nodes(states, mask))
        velocity = projector.transform_derivative(mean_valid_nodes(derivatives, mask))
        times = dynamics["times_s"].detach().cpu().numpy()
        eeg = sample["eeg"].numpy()
        trace_t = np.arange(eeg.shape[1]) / int(sample["sample_rate"])
        labels = [name.upper() for name in CLASSES]
        target = sample["target"].numpy()
        fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.5), gridspec_kw={"height_ratios": [2, 1]})
        scale = max(float(np.percentile(np.abs(eeg[:4]), 95)) * 2.4, 1.)
        for channel in range(4):
            axes[0].plot(trace_t, eeg[channel] + channel * scale, lw=.5,
                         label=ELECTRODES[channel])
        axes[0].axvspan(20, 30, color="#ead1d8", alpha=.8, label="expert label interval")
        axes[0].set(xlim=(0, 50), xlabel="EEG window time (s)", ylabel="Offset channels",
                    title="50-s EEG; central 20-30 s has expert votes")
        axes[0].legend(ncol=5, fontsize=7, frameon=False)
        x = np.arange(len(CLASSES))
        axes[1].bar(x-.18, target, width=.36, label="Experts", color="#447c94")
        axes[1].bar(x+.18, probabilities, width=.36, label="ODEBrain v2", color="#a21d37")
        axes[1].set(xticks=x, xticklabels=labels, ylim=(0, 1), ylabel="Probability")
        axes[1].legend(frameon=False, ncol=2)
        fig.suptitle("%s | %s | expert %s | predicted %s" %
                     (kind.upper(), window_id, labels[int(sample["class_id"])],
                      labels[int(np.argmax(probabilities))]))
        for suffix in ("png", "svg"):
            fig.savefig(output / (kind + "_case." + suffix), dpi=180,
                        bbox_inches="tight", facecolor="white")
        plt.close(fig)
        cases.append({"kind": kind, "window_id": window_id,
                      "patient_id": sample["patient_id"],
                      "eeg_id": str(dataset.rows.iloc[by_id[window_id]].eeg_id),
                      "expert_class": int(sample["class_id"]),
                      "predicted_class": int(np.argmax(probabilities)),
                      "expert_fraction": float(target[int(sample["class_id"])]),
                      "probabilities": probabilities.tolist(),
                      "target": target.tolist(), "projected_path": path.tolist(),
                      "projected_velocity": velocity.tolist(), "ode_times_s": times.tolist()})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for case in cases:
        p = np.array(case["projected_path"])
        t = np.array(case["ode_times_s"])
        selected = t <= 1.00001
        ax = axes[0] if case["kind"] == "correct" else axes[1]
        ax.plot(p[selected, 0], p[selected, 1], marker="o", ms=3,
                color="#447c94" if case["kind"] == "correct" else "#a21d37")
        ax.scatter(p[0, 0], p[0, 1], color="#0d8f7c", s=65, zorder=4, label="start")
        ax.scatter(p[selected][-1, 0], p[selected][-1, 1], color="#a21d37",
                   marker="s", s=55, zorder=4, label="t=1 s")
        ax.set(title="%s: %s" % (case["kind"].title(), case["window_id"]),
               xlabel="Train PCA 1", ylabel="Train PCA 2")
        ax.grid(alpha=.2)
    axes[1].legend(frameon=False)
    fig.suptitle("Classification uses the mean of 12 states along each 0-1 s latent trajectory")
    for suffix in ("png", "svg"):
        fig.savefig(output / ("case_trajectories." + suffix), dpi=180,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    detail = cases[0]
    path = np.asarray(detail["projected_path"])
    velocity = np.asarray(detail["projected_velocity"])
    times = np.asarray(detail["ode_times_s"])
    chinese_font = Path("/export/home/limianchong/.local/share/fonts/NotoSansCJKsc-Regular.otf")
    if chinese_font.exists():
        font_manager.fontManager.addfont(str(chinese_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(chinese_font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 7.0),
                             gridspec_kw={"height_ratios": [2.1, 1]})
    fig.subplots_adjust(left=.16, right=.95, top=.88, bottom=.13, hspace=.55)
    class_mask = times <= 1.00001
    class_path = path[class_mask]
    axes[0].plot(path[:, 0], path[:, 1], color="#8da2ae", lw=2.0,
                 label="0–5 秒模型轨迹", zorder=1)
    axes[0].plot(class_path[:, 0], class_path[:, 1], color="#e5aa19", lw=3.4,
                 label="0–1 秒分类使用区间", zorder=2)
    axes[0].scatter(class_path[:, 0], class_path[:, 1], color="#e5aa19",
                    edgecolor="white", linewidth=.5, s=25, zorder=3)
    axes[0].scatter(path[0, 0], path[0, 1], c="#0d8f7c", s=85, marker="o",
                    edgecolors="white", linewidths=.8, zorder=5)
    axes[0].scatter(class_path[-1, 0], class_path[-1, 1], c="#a21d37",
                    s=78, marker="s", edgecolors="white", linewidths=.8, zorder=5)
    axes[0].scatter(path[-1, 0], path[-1, 1], c="#334e66",
                    s=75, marker="D", edgecolors="white", linewidths=.8, zorder=5)
    for point, label, offset in ((path[0], "t = 0", (10, 10)),
                                 (class_path[-1], "t = 1 s", (11, -13)),
                                 (path[-1], "t = 5 s", (-48, -18))):
        axes[0].annotate(label, point, xytext=offset, textcoords="offset points",
                         fontsize=8, color="#263646")
    axes[0].set(xlabel="训练集 PCA 第 1 轴", ylabel="训练集 PCA 第 2 轴",
                title="潜在状态轨迹（训练集 PCA 二维投影）")
    axes[0].legend(loc="lower right", fontsize=7.5, frameon=False)
    axes[0].grid(alpha=.18)
    speed = np.linalg.norm(velocity, axis=1)
    axes[1].axvspan(0, 1, color="#ead1d8", alpha=.75, zorder=0)
    axes[1].plot(times, speed, color="#a21d37", lw=2.0, marker="o",
                 ms=3.3, zorder=2)
    axes[1].set(xlim=(-.05, 5.05), xlabel="ODE 求解时间（秒）",
                ylabel="投影场强 ‖Wf(z)‖",
                title="沿轨迹的投影向量场强度")
    axes[1].grid(alpha=.18)
    fig.suptitle("%s：OTHER 预测正确" % detail["window_id"],
                 fontsize=14, fontweight="bold", y=.975)
    fig.text(.5, .045, "粉色区：分类用的 12 个状态；ODE 时间是模型计算时间，并非新输入的 EEG。",
             ha="center", fontsize=8.2, color="#52616b")
    for suffix in ("png", "svg"):
        fig.savefig(output / ("case_trajectory_detail." + suffix), dpi=180,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    metadata = {"schema_version": "odebrain_v2_case_pair_v1",
                "checkpoint_sha256": sha256(checkpoint),
                "test_predictions_sha256": sha256(predictions),
                "selection_rule": "lowest window_id among correct/error test windows with expert class fraction >=0.75, otherwise lowest window_id",
                "pca": str(projector_path), "cases": cases}
    (output / "case_pair.json").write_text(json.dumps(metadata, ensure_ascii=False,
                                                        indent=2, allow_nan=False), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--manifest", default="data/curated/split_manifest.csv")
    parser.add_argument("--output", default="results/odebrain_github_v2/cases")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    plot(args.checkpoint, args.predictions, args.pca, args.manifest, args.output, args.device)
