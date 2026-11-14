"""Paper-style local streamlines from the fitted ODEBrain v2 field.

All six views use the same autonomous model and train-only PCA projection.
The held-out examples determine only the centre and overlaid trajectory.
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
        raise ValueError("Original figure, checkpoint and manifest must agree")
    model, saved = load_trained_model(checkpoint, device=device)
    if model.architecture != "github_trajectory_v2":
        raise ValueError("Expected the formal ODEBrain v2 trajectory model")
    config = saved["config"]
    if config.get("schema_version") != "hms_final_refit_v1" or config.get("smoke_subset") is not False:
        raise ValueError("Expected a formal final refit checkpoint")
    projector = TrainOnlyPCA.load(source_dir / "six_class_train_pca.npz")
    if projector.model_id != "odebrain_github_v2_" + sha256(checkpoint)[:16]:
        raise ValueError("Train-only PCA belongs to another model")
    with np.load(checkpoint.parent / "graph_normalizer.npz", allow_pickle=False) as values:
        mean, std = values["mean"], values["std"]
    test = HMSWindowDataset(manifest, "test", graph_mean=mean, graph_std=std,
                            include_spectrogram=False,
                            graph_cache_dir=config["data"]["graph_cache_dir"])
    index_by_id = {str(value): i for i, value in enumerate(test.rows.window_id)}
    cases = []
    for entry in source["cases"]:
        sample = test[index_by_id[entry["window_id"]]]
        if int(sample["class_id"]) != entry["class_id"]:
            raise ValueError("Held-out class changed")
        states, _, _ = infer(model, sample, device)
        path = projector.transform(states)
        if not np.isfinite(path).all():
            raise ValueError("Non-finite latent trajectory")
        cases.append({"window_id": entry["window_id"], "class_id": entry["class_id"],
                      "path": path})

    # The classification grid has 12 RK4 trajectory samples from t=0 to t=1.
    # Use a common physical size for all six local views.
    first_second = [case["path"][:12] for case in cases]
    movement = [np.ptp(path, axis=0).max() for path in first_second]
    width = float(np.clip(3.0 * np.quantile(movement, .85), 5.0, 9.0))
    n = 96
    half = width / 2
    for case, short_path in zip(cases, first_second):
        centre = short_path.mean(axis=0)
        x = np.linspace(centre[0] - half, centre[0] + half, n)
        y = np.linspace(centre[1] - half, centre[1] + half, n)
        xx, yy = np.meshgrid(x, y)
        lifted = projector.mean + np.stack([xx.ravel(), yy.ravel()], axis=1) @ projector.components
        chunks = []
        with torch.no_grad():
            for batch in np.array_split(lifted, 8):
                result = model.vector_field(torch.tensor(0., device=device),
                                            torch.as_tensor(batch, dtype=torch.float32, device=device))
                chunks.append(result.detach().cpu().numpy())
        field = np.concatenate(chunks) @ projector.components.T
        case.update(x=x, y=y, vx=field[:, 0].reshape(n, n), vy=field[:, 1].reshape(n, n))
        case["speed"] = np.hypot(case["vx"], case["vy"])

    pooled = np.concatenate([case["speed"].ravel() for case in cases])
    norm = Normalize(vmin=float(np.quantile(pooled, .03)),
                     vmax=float(np.quantile(pooled, .97)))
    font_path = Path("/export/home/limianchong/.local/share/fonts/NotoSansCJKsc-Regular.otf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 10.2))
    fig.subplots_adjust(left=.035, right=.965, top=.91, bottom=.19,
                        wspace=.08, hspace=.23)
    fig.patch.set_facecolor("white")
    for ax, case in zip(axes.flat, cases):
        x, y = case["x"], case["y"]
        ax.imshow(case["speed"], origin="lower",
                  extent=(x[0], x[-1], y[0], y[-1]), cmap="plasma", norm=norm,
                  interpolation="bilinear", aspect="equal")
        ax.streamplot(x, y, case["vx"], case["vy"], density=2.0,
                      color="#1d1431", linewidth=.48, arrowsize=.53,
                      arrowstyle="->", maxlength=3.0, zorder=2)
        short_path = case["path"][:12]
        ax.plot(short_path[:, 0], short_path[:, 1], color="white", lw=2.0,
                alpha=.95, zorder=4)
        ax.plot(short_path[:, 0], short_path[:, 1], color="#ffcf4a", lw=.9,
                zorder=5)
        ax.scatter(short_path[0, 0], short_path[0, 1], marker="*", s=250,
                   c="#d10c2f", edgecolors="white", linewidths=.9, zorder=7)
        ax.scatter(short_path[-1, 0], short_path[-1, 1], marker="o", s=35,
                   c="#ffdc5e", edgecolors="#46223c", linewidths=.7, zorder=6)
        ax.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("%s  ·  %s" % (CLASSES[case["class_id"]].upper(),
                                    case["window_id"]), fontsize=14, pad=9)
        for spine in ax.spines.values():
            spine.set_linewidth(.7)
            spine.set_color("#cab6c2")
    fig.suptitle("ODEBrain v2：同一个连续动力场中的六类 EEG 测试窗口",
                 fontsize=19, fontweight="bold", y=.985, color="#762037")
    cax = fig.add_axes([.27, .126, .46, .014])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="plasma"),
                      cax=cax, orientation="horizontal")
    cb.set_label("二维 PCA 投影中的场速度  ‖Wf(z)‖", fontsize=10, labelpad=3)
    cb.ax.tick_params(labelsize=8, length=2)
    fig.text(.5, .066,
             "黑线：模型流线   ★：测试窗口初始状态   黄线：前 1 秒潜在轨迹   ●：1 秒末状态",
             ha="center", fontsize=10, color="#493243")
    fig.text(.5, .034,
             "六个视野共用同一模型与训练集 PCA；颜色不是脑电传播速度。二维切片会丢失信息，轨迹不必与流线完全重合。",
             ha="center", fontsize=8.4, color="#655968")
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(output_dir / ("six_class_streamfield." + ext), dpi=210,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    metadata = {"schema_version": "odebrain_six_class_streamfield_v1",
                "checkpoint_sha256": source["checkpoint_sha256"],
                "manifest_sha256": source["manifest_sha256"],
                "pca_split_hash": projector.split_hash,
                "field": "one autonomous f(z); local views centred on six held-out test paths",
                "background": "norm of model field projected to train-fitted 2D PCA plane",
                "overlays": "first second of each held-out sample's model latent trajectory",
                "local_width_pca_units": width,
                "window_ids": [case["window_id"] for case in cases]}
    (output_dir / "six_class_streamfield.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_dir / "six_class_streamfield.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/refit/odebrain_github_v2_hms/final.pt")
    parser.add_argument("--manifest", default="data/curated/split_manifest.csv")
    parser.add_argument("--source-dir", default="results/odebrain_github_v2/six_class_field")
    parser.add_argument("--output-dir", default="results/odebrain_github_v2/six_class_field")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args.checkpoint, args.manifest, args.source_dir, args.output_dir, args.device)
