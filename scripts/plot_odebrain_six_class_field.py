"""Six-class latent ODE field on a train-fitted two-dimensional PCA slice.

The field is evaluated on a shared PCA plane. Each panel overlays one
prespecified, high-consensus held-out HMS window of that expert class.
Curl describes this projected model field, not physical EEG circulation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.dataset import HMSWindowDataset
from src.data.hms import CLASSES, manifest_hash
from src.dynamics.analysis import TrainOnlyPCA, mean_valid_nodes
from src.dynamics.export import load_trained_model


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def planar_curl(vx, vy, x, y):
    """Return d(vy)/dx - d(vx)/dy on a regular XY grid."""
    return np.gradient(vy, x, axis=1) - np.gradient(vx, y, axis=0)


def infer(model, sample, device):
    tensors = {key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor)
               else value for key, value in sample.items()}
    with torch.no_grad():
        logits, dynamics = model(tensors, return_dynamics=True)
    mask = dynamics["observed_node_mask"][0].detach().cpu().numpy().astype(bool)
    states = dynamics["trajectory"][0].detach().cpu().numpy()
    field = dynamics["vector_field"][0].detach().cpu().numpy()
    return (mean_valid_nodes(states, mask), mean_valid_nodes(field, mask),
            torch.softmax(logits, -1)[0].detach().cpu().numpy())


def choose_train_indices(dataset, per_class=16):
    """Patient-diverse, deterministic train-only PCA subset."""
    selected = []
    for class_id in range(len(CLASSES)):
        rows = dataset.rows.loc[dataset.rows.class_id == class_id].copy()
        rows = rows.sort_values(["patient_id", "window_id"]).drop_duplicates("patient_id")
        selected.extend(rows.index[:per_class].tolist())
    if len(selected) < 12:
        raise ValueError("Too few distinct train patients to fit six-class PCA")
    return selected


def choose_test_indices(dataset):
    selected = []
    for class_id in range(len(CLASSES)):
        rows = dataset.rows.loc[dataset.rows.class_id == class_id].copy()
        if rows.empty:
            raise ValueError("No held-out examples for %s" % CLASSES[class_id])
        vote_columns = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
        votes = rows[vote_columns].to_numpy(dtype=float)
        consensus = votes[:, class_id] / np.maximum(votes.sum(axis=1), 1)
        rows["consensus_fraction"] = consensus
        rows = rows.sort_values(["consensus_fraction", "window_id"],
                                ascending=[False, True])
        selected.append(int(rows.index[0]))
    return selected


def plot(checkpoint, manifest, output, device="cuda", train_per_class=16,
         grid_size=23):
    checkpoint, manifest, output = Path(checkpoint), Path(manifest), Path(output)
    model, saved = load_trained_model(checkpoint, device=device)
    if model.architecture != "github_trajectory_v2":
        raise ValueError("Six-class field requires the revised ODEBrain model")
    config = saved.get("config", {})
    if config.get("schema_version") != "hms_final_refit_v1" or config.get("smoke_subset") is not False:
        raise ValueError("Six-class field requires a formal final refit checkpoint")
    if config.get("manifest_sha256") != sha256(manifest):
        raise ValueError("Checkpoint and audited manifest differ")
    with np.load(checkpoint.parent / "graph_normalizer.npz", allow_pickle=False) as values:
        mean, std = values["mean"], values["std"]
    cache = config["data"]["graph_cache_dir"]
    train = HMSWindowDataset(manifest, "train", graph_mean=mean, graph_std=std,
                             include_spectrogram=False, graph_cache_dir=cache)
    test = HMSWindowDataset(manifest, "test", graph_mean=mean, graph_std=std,
                            include_spectrogram=False, graph_cache_dir=cache)
    train_ids, train_states = [], []
    for index in choose_train_indices(train, train_per_class):
        sample = train[index]
        states, _, _ = infer(model, sample, device)
        train_ids.append(sample["window_id"])
        train_states.append(states)
    split_hash = manifest_hash(pd.read_csv(manifest))
    projector = TrainOnlyPCA.fit(train_states, "odebrain_github_v2_" + sha256(checkpoint)[:16],
                                  split_hash, train_ids, n_components=2)
    selected = []
    for class_id, index in enumerate(choose_test_indices(test)):
        sample = test[index]
        states, derivatives, probability = infer(model, sample, device)
        selected.append({"class_id": class_id, "window_id": sample["window_id"],
                         "patient_id": sample["patient_id"],
                         "expert_fraction": float(sample["target"][class_id]),
                         "predicted_class": int(np.argmax(probability)),
                         "projected": projector.transform(states),
                         "projected_derivative": projector.transform_derivative(derivatives)})
    train_xy = np.concatenate([projector.transform(states) for states in train_states])
    all_xy = np.concatenate([train_xy] + [item["projected"] for item in selected])
    lower = np.quantile(all_xy, .02, axis=0)
    upper = np.quantile(all_xy, .98, axis=0)
    lower = np.minimum(lower, all_xy.min(axis=0))
    upper = np.maximum(upper, all_xy.max(axis=0))
    span = np.maximum(upper - lower, 1e-3)
    lower -= .15 * span
    upper += .15 * span
    x = np.linspace(lower[0], upper[0], grid_size)
    y = np.linspace(lower[1], upper[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    lifted = projector.mean + np.stack([xx.ravel(), yy.ravel()], axis=1) @ projector.components
    with torch.no_grad():
        field = model.vector_field(torch.tensor(0., device=device),
                                   torch.as_tensor(lifted, dtype=torch.float32, device=device))
    projected_field = field.detach().cpu().numpy() @ projector.components.T
    vx = projected_field[:, 0].reshape(xx.shape)
    vy = projected_field[:, 1].reshape(xx.shape)
    curl = planar_curl(vx, vy, x, y)
    curl_scale = max(float(np.quantile(np.abs(curl), .98)), 1e-8)
    near_zero_threshold = .1 * curl_scale
    output.mkdir(parents=True, exist_ok=True)
    projector.save(output / "six_class_train_pca.npz")
    fig, axes = plt.subplots(2, 3, figsize=(17, 7), sharex=True, sharey=True)
    for ax, item in zip(axes.flat, selected):
        ax.contourf(xx, yy, curl, levels=np.linspace(-curl_scale, curl_scale, 21),
                    cmap="coolwarm", vmin=-curl_scale, vmax=curl_scale, extend="both")
        skip = max(1, grid_size // 12)
        ax.quiver(xx[::skip, ::skip], yy[::skip, ::skip],
                  vx[::skip, ::skip], vy[::skip, ::skip],
                  color="#263646", alpha=.7, angles="xy", scale=None)
        path = item["projected"]
        ax.plot(path[:, 0], path[:, 1], color="#f8d158", lw=2.3)
        ax.scatter(path[0, 0], path[0, 1], c="#0d8f7c", s=45, marker="o", zorder=4)
        ax.scatter(path[-1, 0], path[-1, 1], c="#a21d37", s=45, marker="s", zorder=4)
        local = np.interp(path[:, 0], x, np.arange(len(x))).astype(int).clip(0, len(x)-1)
        local_y = np.interp(path[:, 1], y, np.arange(len(y))).astype(int).clip(0, len(y)-1)
        item["mean_abs_projected_curl"] = float(np.mean(np.abs(curl[local_y, local])))
        item["mean_signed_projected_curl"] = float(np.mean(curl[local_y, local]))
        item["projected_rotation_assessment"] = ("local_near_zero_curl" if
            item["mean_abs_projected_curl"] <= near_zero_threshold else "local_nonzero_curl")
        assessment = ("near-zero curl" if item["projected_rotation_assessment"] ==
                      "local_near_zero_curl" else "local rotation")
        ax.set(title="%s | %s | mean |curl| %.3g" %
               (CLASSES[item["class_id"]].upper(), assessment,
                item["mean_abs_projected_curl"]),
               xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
        ax.grid(alpha=.14)
    for ax in axes[-1]:
        ax.set_xlabel("Train PCA 1")
    for ax in axes[:, 0]:
        ax.set_ylabel("Train PCA 2")
    fig.suptitle("One shared projected ODE field; six held-out classes start at different states")
    fig.text(.5, .015, "Blue/Red = signed projected curl; yellow = observed latent path; green circle = start; red square = end."
             " Local 2D model diagnostic only.", ha="center", fontsize=10)
    for extension in ("png", "svg"):
        fig.savefig(output / ("six_class_ode_field." + extension), dpi=180,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    summary = {"schema_version": "odebrain_six_class_projected_field_v1",
               "checkpoint_sha256": sha256(checkpoint), "manifest_sha256": sha256(manifest),
               "split_hash": split_hash, "pca_train_window_ids": train_ids,
               "pca_fit": "classification trajectories from train patients only",
               "field": "same autonomous f(z) on train-PCA 2D slice in all six panels",
               "curl_definition": "d(v_PCA2)/d(PCA1)-d(v_PCA1)/d(PCA2)",
               "near_zero_threshold": near_zero_threshold,
               "near_zero_rule": "mean absolute curl along the projected path <= 10% of the grid's 98th-percentile absolute curl",
               "warning": "Projected slice curl is not physical EEG circulation or full-dimensional curl",
               "cases": [{k: v for k, v in item.items() if k not in ("projected", "projected_derivative")}
                         for item in selected]}
    (output / "six_class_ode_field.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/curated/split_manifest.csv")
    parser.add_argument("--output", default="results/odebrain_github_v2/six_class_field")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-per-class", type=int, default=16)
    args = parser.parse_args()
    plot(args.checkpoint, args.manifest, args.output, args.device, args.train_per_class)
