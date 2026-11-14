"""Plot formal ODEBrain HMS v2 comparisons from verified refit summaries.

Run only after the new classification and forecast refits have been evaluated.
The input JSONs are produced by refit_final.py summarize and
evaluate_forecast.py --protocol refit, respectively.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {"tcn": "TCN", "spectrogram_cnn": "Spectrogram CNN",
          "fusion": "Time-frequency fusion", "graph_gru": "GraphGRU",
          "odebrain_inspired_hms": "ODEBrain-HMS v2", "persistence": "Persistence"}
ORDER = ("tcn", "spectrogram_cnn", "fusion", "graph_gru", "odebrain_inspired_hms")
COLOR = {"odebrain_inspired_hms": "#a21d37", "graph_gru": "#447c94",
         "persistence": "#808080"}


def read_formal(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if (obj.get("experiment_status") != "formal" or obj.get("smoke_subset") is not False
            or obj.get("synthetic") is not False or obj.get("split") != "test"
            or obj.get("training_protocol") != "train_val_fixed_epoch_refit"):
        raise ValueError("Only formal, held-out, fixed-epoch refit comparisons may be plotted: %s" % path)
    return obj


def save(fig, output):
    for suffix in ("png", "svg"):
        fig.savefig(str(output.with_suffix("." + suffix)), dpi=180,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_training_curves(pretrain_run, classification_run, output_dir):
    runs = ((Path(pretrain_run), "forecast_mse", "Future graph-node MSE"),
            (Path(classification_run), "kl", "Six-class soft-label KL"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, (run_dir, key, title) in zip(axes, runs):
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if (config.get("smoke_subset") is not False or
                config.get("model", {}).get("kwargs", {}).get("architecture") != "github_trajectory_v2"):
            raise ValueError("Training curve must come from a formal ODEBrain v2 run")
        history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
        if not history:
            raise ValueError("Training history is empty")
        epochs = [int(row["epoch"]) + 1 for row in history]
        train = [float(row["train"][key]) for row in history]
        val = [float(row["val"][key]) for row in history]
        best = int(np.argmin(val))
        ax.plot(epochs, train, "o-", label="Train", color="#447c94")
        ax.plot(epochs, val, "o-", label="Validation", color="#a21d37")
        ax.scatter(epochs[best], val[best], marker="*", s=170,
                   color="#dda824", zorder=5, label="Selected epoch")
        ax.set(title=title, xlabel="Epoch", ylabel=key.upper(), xticks=epochs[::max(1, len(epochs)//8)])
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle("ODEBrain v2: validation selects the refit epoch count")
    save(fig, Path(output_dir) / "odebrain_training_curves")


def plot_pretraining_ablation(classification_run, scratch_run, output_dir):
    runs = ((Path(classification_run), "With graph forecast pretraining", "#a21d37"),
            (Path(scratch_run), "From random initialization", "#447c94"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    selected = []
    for run_dir, label, color in runs:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if (config.get("smoke_subset") is not False or
                config.get("model", {}).get("kwargs", {}).get("architecture") != "github_trajectory_v2" or
                config.get("data", {}).get("task", "classification") != "classification"):
            raise ValueError("Ablation must compare two formal revised classifier runs")
        history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
        values = [float(row["val"]["kl"]) for row in history]
        best = int(np.argmin(values))
        selected.append((label, values[best], best + 1, color))
        axes[0].plot(range(1, len(values)+1), values, marker="o", ms=3,
                     color=color, label=label)
        axes[0].scatter(best+1, values[best], marker="*", s=150,
                        color=color, zorder=5)
    axes[0].set(title="Validation KL by epoch", xlabel="Epoch", ylabel="Soft-label KL (lower is better)")
    axes[0].grid(alpha=.2)
    axes[0].legend(frameon=False)
    bars = axes[1].bar(range(2), [item[1] for item in selected],
                       color=[item[3] for item in selected])
    axes[1].set(title="Best validation KL", xticks=range(2),
                xticklabels=["Pretrained", "From scratch"],
                ylabel="Soft-label KL (lower is better)")
    axes[1].set_ylim(0, max(item[1] for item in selected) * 1.25)
    for bar, item in zip(bars, selected):
        axes[1].text(bar.get_x()+bar.get_width()/2, item[1] * 1.02,
                     "%.3f\n(epoch %d)" % (item[1], item[2]),
                     ha="center", fontsize=9)
    fig.suptitle("ODEBrain v2 ablation: same architecture and train/validation patients")
    save(fig, Path(output_dir) / "odebrain_pretraining_ablation")


def plot(classification_path, forecast_path, output_dir, weighted_path=None):
    classification = read_formal(classification_path)
    forecast = read_formal(forecast_path)
    if (classification.get("split_hash") != forecast.get("split_hash") or
            classification.get("manifest_sha256") != forecast.get("manifest_sha256")):
        raise ValueError("Classification and forecast results use different patient splits")
    rows = {row["name"]: row for row in classification["rows"]}
    if set(rows) != set(ORDER):
        raise ValueError("Classification comparison must contain five unique model names")
    ode_row = rows["odebrain_inspired_hms"]
    if "github_v2" not in ode_row["run_dir"]:
        raise ValueError("ODE classification row is not from the GitHub trajectory v2 refit")
    if "github_v2" not in forecast["provenance"]["ode"]["run_dir"]:
        raise ValueError("ODE forecast row is not from the GitHub trajectory v2 refit")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = (("kl_window_mean", "Window-equal KL"),
               ("kl_eeg_record_equal_mean", "EEG-record-equal KL"),
               ("kl_patient_equal_mean", "Patient-equal KL"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (key, title) in zip(axes, metrics):
        values = [float(rows[name][key]) for name in ORDER]
        bars = ax.barh(range(len(ORDER)), values,
                       color=[COLOR.get(name, "#88a6b5") for name in ORDER])
        ax.set(title=title, xlabel="KL divergence (lower is better)",
               yticks=range(len(ORDER)), yticklabels=[LABELS[name] for name in ORDER])
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.20)
        for bar, value in zip(bars, values):
            ax.text(value + max(values) * .015, bar.get_y() + bar.get_height() / 2,
                    "%.3f" % value, va="center", fontsize=9)
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("HMS six-class soft-label classification: fixed-epoch refit, held-out patients")
    save(fig, output_dir / "classification_kl_comparison")

    forecast_rows = {row["name"]: row for row in forecast["rows"]}
    if set(forecast_rows) != {"odebrain_inspired_hms", "graph_gru", "persistence"}:
        raise ValueError("Forecast comparison must contain ODE, GraphGRU and persistence")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in ((axes[0], "mae_by_horizon_s", "Window-equal MAE"),
                           (axes[1], "eeg_record_equal_mae_by_horizon_s", "EEG-record-equal MAE")):
        for name in ("odebrain_inspired_hms", "graph_gru", "persistence"):
            values = [float(forecast_rows[name][key][str(h)]) for h in (1, 3, 5)]
            ax.plot((1, 3, 5), values, marker="o", lw=2,
                    label=LABELS[name], color=COLOR[name])
        ax.set(xlabel="Forecast horizon after 5 s observed EEG (s)", ylabel="MAE (log1p band power)",
               title=title, xticks=(1, 3, 5))
        ax.grid(alpha=.2)
    axes[1].legend(frameon=False)
    fig.suptitle("Future EEG graph-node attributes: fixed-epoch refit, held-out patients")
    save(fig, output_dir / "forecast_mae_comparison")

    if weighted_path is not None:
        weighted = json.loads(Path(weighted_path).read_text(encoding="utf-8"))
        if (weighted.get("experiment_status") != "formal" or
                weighted.get("split_hash") != classification.get("split_hash") or
                weighted.get("source_sha256", {}).get("deep_comparison") !=
                hashlib.sha256(Path(classification_path).read_bytes()).hexdigest()):
            raise ValueError("Weighted metrics do not match the formal revised comparison")
        views = {name: weighted["models"][name]["aggregations"]["window_equal"]
                 for name in ORDER}
        scores = {
            "Macro F1": [views[name]["macro_f1"] for name in ORDER],
            "Macro recall": [np.mean([c["recall"] for c in views[name]["per_class"].values()])
                             for name in ORDER],
            "Macro one-vs-rest AUC": [np.mean([c["ovr_auroc"] for c in views[name]["per_class"].values()])
                                      for name in ORDER],
        }
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, (metric, values) in zip(axes, scores.items()):
            ax.bar(range(len(ORDER)), values,
                   color=[COLOR.get(name, "#88a6b5") for name in ORDER])
            ax.set(title=metric, ylim=(0, 1), xticks=range(len(ORDER)),
                   xticklabels=[LABELS[name] for name in ORDER])
            ax.tick_params(axis="x", rotation=35)
            for i, value in enumerate(values):
                ax.text(i, value + .015, "%.2f" % value, ha="center", fontsize=8)
        fig.suptitle("Hard expert-consensus labels: auxiliary window-equal metrics")
        save(fig, output_dir / "classification_auxiliary_metrics")

        class_names = list(weighted["class_order"])
        matrix = np.array([[views[name]["per_class"][label]["f1"] for label in class_names]
                           for name in ORDER])
        fig, ax = plt.subplots(figsize=(10, 4.7))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set(yticks=range(len(ORDER)), yticklabels=[LABELS[name] for name in ORDER],
               xticks=range(len(class_names)), xticklabels=[c.upper() for c in class_names],
               title="Per-class F1 on expert-consensus labels")
        for i in range(len(ORDER)):
            for j in range(len(class_names)):
                ax.text(j, i, "%.2f" % matrix[i, j], ha="center", va="center",
                        color="white" if matrix[i, j] > .55 else "#23313b", fontsize=9)
        fig.colorbar(im, ax=ax, label="F1")
        save(fig, output_dir / "classification_per_class_f1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification", required=True)
    parser.add_argument("--forecast", required=True)
    parser.add_argument("--weighted", help="Formal weighted hard-label metrics for extra figures")
    parser.add_argument("--pretrain-run", help="Formal ODEBrain v2 forecast pretraining run")
    parser.add_argument("--classification-run", help="Formal ODEBrain v2 classifier run")
    parser.add_argument("--scratch-run", help="Formal ODEBrain v2 classifier trained from scratch")
    parser.add_argument("--output-dir", default="results/odebrain_github_v2/figures")
    args = parser.parse_args()
    plot(args.classification, args.forecast, args.output_dir, args.weighted)
    if bool(args.pretrain_run) != bool(args.classification_run):
        parser.error("--pretrain-run and --classification-run must be supplied together")
    if args.pretrain_run:
        plot_training_curves(args.pretrain_run, args.classification_run, args.output_dir)
    if args.scratch_run:
        if not args.classification_run:
            parser.error("--scratch-run requires --classification-run")
        plot_pretraining_ablation(args.classification_run, args.scratch_run, args.output_dir)
