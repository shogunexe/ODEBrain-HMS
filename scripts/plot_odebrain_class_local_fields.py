"""Six class-centred slices of one fitted ODEBrain v2 vector field.

Each slice passes through the class's mean train-patient initial state in the
full latent space. All slices use the same train-fitted PCA basis and one
checkpoint. They are descriptive views, not six independently trained fields.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

from src.data.dataset import HMSWindowDataset
from src.data.hms import CLASSES
from src.dynamics.analysis import TrainOnlyPCA
from src.dynamics.export import load_trained_model
from scripts.plot_odebrain_six_class_field import infer, sha256


def main(checkpoint, manifest, source_dir, output_dir, device):
    checkpoint, manifest = Path(checkpoint), Path(manifest)
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    source = json.loads((source_dir / "six_class_ode_field.json").read_text(encoding="utf-8"))
    if source["checkpoint_sha256"] != sha256(checkpoint) or source["manifest_sha256"] != sha256(manifest):
        raise ValueError("Checkpoint or manifest differs from the audited PCA source")
    projector = TrainOnlyPCA.load(source_dir / "six_class_train_pca.npz")
    if tuple(source["pca_train_window_ids"]) != projector.train_window_ids:
        raise ValueError("PCA training examples differ")
    model, saved = load_trained_model(checkpoint, device=device)
    if model.architecture != "github_trajectory_v2":
        raise ValueError("Expected formal ODEBrain v2")
    config = saved["config"]
    if config.get("schema_version") != "hms_final_refit_v1" or config.get("smoke_subset") is not False:
        raise ValueError("Expected a formal refit checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "six_class_train_trajectories.npz"
    ids = source["pca_train_window_ids"]
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if (str(cache["checkpoint_sha256"]) != source["checkpoint_sha256"] or
                list(cache["window_ids"]) != ids):
                raise ValueError("Existing trajectory cache has different provenance")
            states = cache["states"]
            classes = cache["classes"].astype(int)
    else:
        with np.load(checkpoint.parent / "graph_normalizer.npz", allow_pickle=False) as values:
            mean, std = values["mean"], values["std"]
        train = HMSWindowDataset(manifest, "train", graph_mean=mean, graph_std=std,
                                 include_spectrogram=False,
                                 graph_cache_dir=config["data"]["graph_cache_dir"])
        index_by_id = {str(value): i for i, value in enumerate(train.rows.window_id)}
        states, classes = [], []
        for count, window_id in enumerate(ids, 1):
            sample = train[index_by_id[window_id]]
            trajectory, _, _ = infer(model, sample, device)
            states.append(trajectory[:12])
            classes.append(int(sample["class_id"]))
            if count % 16 == 0:
                print("Read %d/%d train-patient windows" % (count, len(ids)), flush=True)
        states = np.stack(states)
        classes = np.asarray(classes, dtype=int)
        np.savez_compressed(cache_path, checkpoint_sha256=source["checkpoint_sha256"],
                            window_ids=np.asarray(ids), states=states, classes=classes)
    if states.shape[0] != 96 or states.shape[1] != 12 or len(classes) != 96:
        raise ValueError("Expected 16 train-patient trajectories for each of six classes")

    groups = []
    for class_id in range(6):
        group = states[classes == class_id]
        if len(group) != 16:
            raise ValueError("Wrong class count: %s" % CLASSES[class_id])
        anchor = group[:, 0].mean(axis=0)
        path = (group - anchor) @ projector.components.T
        groups.append({"class_id": class_id, "anchor": anchor,
                       "paths": path, "mean_path": path.mean(axis=0)})
    pooled_paths = np.concatenate([group["paths"].reshape(-1, 2) for group in groups])
    half = float(np.clip(np.quantile(np.abs(pooled_paths), .985) + .8, 3.5, 8.0))
    n = 90
    x = np.linspace(-half, half, n)
    y = np.linspace(-half, half, n)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    for group in groups:
        # Parallel 2D planes have different class-dependent offsets in the
        # latent dimensions omitted by PCA.
        lifted = group["anchor"] + grid @ projector.components
        chunks = []
        with torch.no_grad():
            for batch in np.array_split(lifted, 8):
                field = model.vector_field(torch.tensor(0., device=device),
                                           torch.as_tensor(batch, dtype=torch.float32, device=device))
                chunks.append(field.detach().cpu().numpy())
        field = np.concatenate(chunks) @ projector.components.T
        group["vx"] = field[:, 0].reshape(n, n)
        group["vy"] = field[:, 1].reshape(n, n)
        group["speed"] = np.hypot(group["vx"], group["vy"])
    pooled_speed = np.concatenate([group["speed"].ravel() for group in groups])
    norm = Normalize(vmin=float(np.quantile(pooled_speed, .02)),
                     vmax=float(np.quantile(pooled_speed, .98)))
    font_path = Path("/export/home/limianchong/.local/share/fonts/NotoSansCJKsc-Regular.otf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 10.6))
    fig.subplots_adjust(left=.035, right=.965, top=.91, bottom=.19,
                        wspace=.08, hspace=.23)
    for ax, group in zip(axes.flat, groups):
        ax.imshow(group["speed"], origin="lower", extent=(-half, half, -half, half),
                  cmap="plasma", norm=norm, interpolation="bilinear", aspect="equal")
        ax.streamplot(x, y, group["vx"], group["vy"], density=1.9,
                      color="#1d1431", linewidth=.52, arrowsize=.53,
                      arrowstyle="->", maxlength=3.0, zorder=2)
        for path in group["paths"]:
            ax.plot(path[:, 0], path[:, 1], color="white", lw=.75,
                    alpha=.33, zorder=3)
        mean_path = group["mean_path"]
        ax.plot(mean_path[:, 0], mean_path[:, 1], color="white", lw=3.0, zorder=4)
        ax.plot(mean_path[:, 0], mean_path[:, 1], color="#ffca46", lw=1.6, zorder=5)
        ax.scatter(mean_path[0, 0], mean_path[0, 1], marker="*", s=240,
                   c="#d10c2f", edgecolors="white", linewidths=.9, zorder=7)
        ax.scatter(mean_path[-1, 0], mean_path[-1, 1], marker="o", s=38,
                   c="#ffdc5e", edgecolors="#46223c", linewidths=.7, zorder=6)
        ax.set(xlim=(-half, half), ylim=(-half, half))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("%s  ·  16 位训练患者" % CLASSES[group["class_id"]].upper(),
                     fontsize=14, pad=8)
        for spine in ax.spines.values():
            spine.set_linewidth(.7)
            spine.set_color("#cab6c2")
    fig.suptitle("六类脑电的局部动力形态：同一 ODE，不同类别的潜在状态切面",
                 fontsize=19, fontweight="bold", y=.985, color="#762037")
    cax = fig.add_axes([.27, .126, .46, .014])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="plasma"),
                      cax=cax, orientation="horizontal")
    cb.set_label("二维切面中的场速度  ‖Wf(z)‖", fontsize=10, labelpad=3)
    cb.ax.tick_params(labelsize=8, length=2)
    fig.text(.5, .066,
             "黑线：模型流线   ★：该类平均初始状态   浅线：16 条训练轨迹   黄线：平均轨迹（0–1 秒）",
             ha="center", fontsize=10, color="#493243")
    fig.text(.5, .034,
             "六个切面共用模型、PCA 方向与色标；每个切面通过该类高维初始状态中心。仅为类别条件下的模型可视化。",
             ha="center", fontsize=8.6, color="#655968")
    for ext in ("png", "svg"):
        fig.savefig(output_dir / ("six_class_local_fields." + ext), dpi=210,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    anchors = np.stack([group["anchor"] for group in groups])
    offsets = anchors - projector.mean - (anchors - projector.mean) @ projector.components.T @ projector.components
    residual_distances = np.linalg.norm(offsets[:, None] - offsets[None, :], axis=-1)
    field_vectors = np.stack([np.stack([g["vx"], g["vy"]], axis=-1) for g in groups])
    difference = np.linalg.norm(field_vectors[:, None] - field_vectors[None, :], axis=-1).mean(axis=(-2, -1))
    summary = {"schema_version": "odebrain_class_local_fields_v1",
               "checkpoint_sha256": source["checkpoint_sha256"],
               "manifest_sha256": source["manifest_sha256"],
               "class_counts": {CLASSES[g["class_id"]]: int((classes == g["class_id"]).sum()) for g in groups},
               "field": "one shared autonomous f(z), six parallel PCA planes through class mean full-dimensional initial states",
               "axis": "relative to each class's mean initial state, common train-only PCA basis and equal span",
               "not_class_specific_models": True,
               "max_abs_axis": half,
               "mean_pairwise_field_difference": float(difference[np.triu_indices(6, 1)].mean()),
               "mean_pairwise_anchor_residual_distance": float(residual_distances[np.triu_indices(6, 1)].mean())}
    (output_dir / "six_class_local_fields.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/refit/odebrain_github_v2_hms/final.pt")
    parser.add_argument("--manifest", default="data/curated/split_manifest.csv")
    parser.add_argument("--source-dir", default="results/odebrain_github_v2/six_class_field")
    parser.add_argument("--output-dir", default="results/odebrain_github_v2/six_class_field")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args.checkpoint, args.manifest, args.source_dir, args.output_dir, args.device)
