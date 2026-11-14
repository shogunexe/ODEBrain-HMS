"""Matching latent-trajectory detail for the audited held-out error case."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "results/odebrain_github_v2/cases"
CHECKPOINT = ROOT / "artifacts/refit/odebrain_github_v2_hms/final.pt"


def main():
    import hashlib
    source = json.loads((CASES / "case_pair.json").read_text(encoding="utf-8"))
    checkpoint_hash = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if source["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Case data do not belong to the final checkpoint")
    case = next(item for item in source["cases"] if item["kind"] == "error")
    if case["window_id"] != "row_100028" or case["expert_class"] != 5 or case["predicted_class"] != 0:
        raise ValueError("Held-out error case has changed")
    path = np.asarray(case["projected_path"], dtype=float)
    velocity = np.asarray(case["projected_velocity"], dtype=float)
    times = np.asarray(case["ode_times_s"], dtype=float)
    if path.shape != velocity.shape or path.shape[0] != len(times) or not np.isfinite(path).all():
        raise ValueError("Invalid saved trajectory")
    font_path = Path("/export/home/limianchong/.local/share/fonts/NotoSansCJKsc-Regular.otf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
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
    axes[0].scatter(path[0, 0], path[0, 1], c="#0d8f7c", s=85,
                    edgecolors="white", linewidths=.8, zorder=5)
    axes[0].scatter(class_path[-1, 0], class_path[-1, 1], c="#a21d37",
                    s=78, marker="s", edgecolors="white", linewidths=.8, zorder=5)
    axes[0].scatter(path[-1, 0], path[-1, 1], c="#334e66",
                    s=75, marker="D", edgecolors="white", linewidths=.8, zorder=5)
    for point, label, offset in ((path[0], "t = 0", (10, 10)),
                                 (class_path[-1], "t = 1 s", (10, -14)),
                                 (path[-1], "t = 5 s", (-48, -18))):
        axes[0].annotate(label, point, xytext=offset, textcoords="offset points",
                         fontsize=8, color="#263646")
    axes[0].set(xlabel="训练集 PCA 第 1 轴", ylabel="训练集 PCA 第 2 轴",
                title="潜在状态轨迹（训练集 PCA 二维投影）")
    axes[0].legend(loc="lower right", fontsize=7.5, frameon=False)
    axes[0].grid(alpha=.18)
    speed = np.linalg.norm(velocity, axis=1)
    axes[1].axvspan(0, 1, color="#ead1d8", alpha=.75, zorder=0)
    axes[1].plot(times, speed, color="#a21d37", lw=2.0,
                 marker="o", ms=3.3, zorder=2)
    axes[1].set(xlim=(-.05, 5.05), xlabel="ODE 求解时间（秒）",
                ylabel="投影场强 ‖Wf(z)‖",
                title="沿轨迹的投影向量场强度")
    axes[1].grid(alpha=.18)
    fig.suptitle("row_100028：专家 OTHER 76.5% → 模型 SEIZURE 40.0%",
                 fontsize=12.5, fontweight="bold", y=.975)
    fig.text(.5, .045, "粉色区：分类用的 12 个状态；ODE 时间是模型计算时间，并非新输入的 EEG。",
             ha="center", fontsize=8.2, color="#52616b")
    for ext in ("png", "svg"):
        fig.savefig(CASES / ("error_trajectory_detail." + ext), dpi=180,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(CASES / "error_trajectory_detail.png")


if __name__ == "__main__":
    main()
