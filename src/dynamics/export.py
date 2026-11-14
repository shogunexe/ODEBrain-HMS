"""Export evidence from a trusted trained checkpoint and real HMS windows."""
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import HMSWindowDataset
from src.data.hms import CLASSES, ELECTRODES, graph_frames, manifest_hash
from src.models.odebrain_hms import ODEBrainInspiredHMS
from .analysis import (TrainOnlyPCA, choose_cases, entropy, forecast_errors,
                       mean_valid_nodes, sha256_file, trajectory_summary)


def _tensor_batch(sample, device):
    return {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) and k != "future_times_s"
            else v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()}


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def load_trained_model(checkpoint_path, device="cpu", rk4_step_seconds=None):
    """Load only a trusted local project checkpoint, never untrusted pickle."""
    kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    checkpoint = torch.load(str(checkpoint_path), **kwargs)
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise ValueError("Checkpoint needs model or state_dict")
    hidden = int(state["node_encoder.weight_ih"].shape[0] // 3)
    features = int(state["node_encoder.weight_ih"].shape[1])
    config = checkpoint.get("config", {})
    model_config = config.get("model", {}) if isinstance(config, dict) else {}
    nested = model_config.get("kwargs", {}) if isinstance(model_config, dict) else {}
    architecture = nested.get("architecture", "legacy_v1")
    if architecture == "github_trajectory_v2":
        raw_dim = int(state["raw_encoder.fc.weight"].shape[0])
        classes = int(state["classifier.2.weight"].shape[0])
    else:
        raw_dim = int(state["raw_encoder.0.weight"].shape[0])
        classes = int(state["classifier.1.weight"].shape[0])
    if rk4_step_seconds is None:
        rk4_step_seconds = nested.get("rk4_step_seconds", 0.25)
    model = ODEBrainInspiredHMS(node_features=features, num_nodes=len(ELECTRODES),
                                hidden_dim=hidden, raw_dim=raw_dim,
                                num_classes=classes, architecture=architecture,
                                rk4_step_seconds=rk4_step_seconds)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, checkpoint


def _verified_forecast_source(checkpoint_path, snapshot, model, manifest_csv,
                              device, rk4_step_seconds):
    """Resolve and verify the forecast head retained by a classifier."""
    if snapshot.get("schema_version") == "hms_final_refit_v1":
        # The final classifier deliberately drops the initial run's init_checkpoint.
        # Its forecast head must come from the separately refitted pretrain run.
        from scripts.refit_final import _check_final_run, _refit_pretrain

        checked = _check_final_run(checkpoint_path.parent)
        if (checked["checkpoint_path"] != checkpoint_path or
                checked["config"] != snapshot or
                snapshot.get("refit", {}).get("task") != "classification" or
                snapshot.get("model", {}).get("name") != "odebrain_inspired_hms"):
            raise ValueError("Dynamics checkpoint is not the verified final ODE classifier refit")
        init_dir = snapshot["refit"].get("init_refit_run")
        if not init_dir:
            raise ValueError("Final ODE classifier has no forecast pretraining refit")
        init = _refit_pretrain(init_dir, checked["source"])
        source_path = init["checkpoint"]
        source_hash = snapshot["refit"].get("init_refit_checkpoint_sha256")
        if (source_hash != init["checkpoint_sha256"] or
                Path(init["path"]).resolve() != Path(init_dir).resolve()):
            raise ValueError("Final forecast pretrain refit lineage changed")
        protocol = "train_val_fixed_epoch_refit"
    else:
        source_name = snapshot.get("init_checkpoint")
        source_hash = snapshot.get("init_checkpoint_sha256")
        if not source_name or not source_hash:
            raise ValueError("Forecast head needs positive loss weight or a verified pretrained checkpoint lineage")
        source_path = Path(source_name).expanduser()
        if not source_path.is_absolute():
            source_path = Path(__file__).resolve().parents[2] / source_path
        protocol = None
    source_path = Path(source_path).resolve()
    if not source_path.is_file() or sha256_file(source_path) != source_hash:
        raise ValueError("Pretrained forecast checkpoint is missing or its hash changed")
    source_model, source_checkpoint = load_trained_model(source_path, device, rk4_step_seconds)
    source_config = source_checkpoint.get("config", {})
    source_training = source_config.get("training", {})
    source_manifest_digest = source_config.get("manifest_sha256")
    if protocol and (source_config.get("schema_version") != "hms_final_refit_v1" or
                     source_config.get("refit", {}).get("task") != "forecast" or
                     source_config.get("model") != snapshot.get("model") or
                     source_manifest_digest != snapshot.get("manifest_sha256") or
                     source_config.get("smoke_subset") is not False or
                     source_config != init["config"]):
        raise ValueError("Forecast source is not the verified final pretraining refit")
    if source_manifest_digest is not None and sha256_file(manifest_csv) != source_manifest_digest:
        raise ValueError("Forecast pretrain manifest differs from the export manifest")
    if source_manifest_digest is None and source_config.get("smoke_subset") is False:
        raise ValueError("Formal forecast source lacks a bound manifest digest")
    if float(source_training.get("forecast_weight", 0)) <= 0:
        raise ValueError("Source forecast head was not trained with positive forecast loss")
    final_state = model.state_dict()
    source_state = source_model.state_dict()
    for name in ("forecast_head.weight", "forecast_head.bias"):
        if not torch.equal(final_state[name].detach().cpu(), source_state[name].detach().cpu()):
            raise ValueError("Forecast head differs from verified pretrained source")
    lineage = {"forecast_head_source": str(source_path),
               "forecast_head_source_sha256": source_hash}
    if protocol:
        lineage["training_protocol"] = protocol
        lineage["forecast_pretrain_refit_run"] = str(Path(init_dir).resolve())
    return source_checkpoint, lineage, source_config.get("smoke_subset") is not False


def _infer(model, sample, device):
    with torch.no_grad():
        logits, dynamics = model(_tensor_batch(sample, device), True)
    return _numpy(torch.softmax(logits, dim=-1)[0]), dynamics


def _class_infer(model, sample, device):
    with torch.no_grad():
        logits = model(_tensor_batch(sample, device), False)
    return _numpy(torch.softmax(logits, dim=-1)[0])


def _graph_perturbations(sample, model, device, graph_mean=None, graph_std=None,
                         noise_fraction=0.02, feature_fraction=0.01, seed=42):
    eeg_key = "observed_eeg" if "future_graph_nodes" in sample else "eeg"
    eeg = _numpy(sample[eeg_key]).astype(np.float32)
    valid = _numpy(sample["eeg_mask"]).astype(bool)
    rng = np.random.RandomState(seed)
    scale = float(np.sqrt(np.mean(eeg[valid]**2))) if valid.any() else 0.0
    copies = {}
    noisy = eeg + rng.normal(0, noise_fraction * scale, eeg.shape).astype(np.float32)
    noisy[~valid] = 0
    copies["low_noise"] = (noisy, valid)
    dropped = eeg.copy()
    drop_valid = valid.copy()
    selected_channel = int(np.argmax(valid.mean(axis=1)))
    dropped[selected_channel] = 0
    drop_valid[selected_channel] = False
    copies["single_lead_missing"] = (dropped, drop_valid)
    variants = {}
    for name, (values, observed) in copies.items():
        nodes, adj, mask = graph_frames(values, observed)
        if graph_mean is not None:
            nodes = (nodes - graph_mean) / np.maximum(graph_std, 1e-6)
        nodes = nodes.astype(np.float32, copy=False)
        variant = dict(sample)
        variant[eeg_key] = torch.from_numpy(values.copy())
        variant["graph_nodes"] = torch.from_numpy(nodes.copy())
        variant["graph_adj"] = torch.from_numpy(adj.copy())
        variant["graph_mask"] = torch.from_numpy(mask.copy())
        variant["eeg_mask"] = torch.from_numpy(observed.copy())
        variants[name] = variant
    variant = dict(sample)
    variant["graph_nodes"] = sample["graph_nodes"] * (1.0 + feature_fraction)
    variants["small_graph_feature_scale"] = variant
    return variants, selected_channel


def _write_case_figures(case_dir, arrays, metadata):
    """Portable static PNG/SVG evidence; no graphics library needed in UI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    case_dir = Path(case_dir)
    figures = {}

    def save(fig, name):
        if metadata["provenance"] != "real_hms_checkpoint":
            fig.text(.5, .5, "REAL HMS SMOKE - NOT FORMAL EVIDENCE",
                     ha="center", va="center", rotation=25, fontsize=24,
                     color="crimson", alpha=.25, transform=fig.transFigure)
        for suffix in ("png", "svg"):
            fig.savefig(str(case_dir / (name + "." + suffix)), dpi=180,
                        bbox_inches="tight")
        plt.close(fig)
        figures[name] = {"png": name + ".png", "svg": name + ".svg"}

    time = arrays["times_s"]
    eeg = arrays["classification_eeg"]
    sr = int(metadata["sample_rate_hz"])
    eeg_t = np.arange(eeg.shape[1]) / sr
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    trace_offset = max(1.0, float(np.percentile(np.abs(eeg[:5]), 95)) * 2.5)
    for i in range(min(5, len(eeg))):
        axes[0].plot(eeg_t, eeg[i] + trace_offset*i, lw=.45, label=metadata["electrodes"][i])
    axes[0].set_ylabel("EEG (vertically offset)")
    axes[0].legend(fontsize=7, ncol=5)
    freq, sec, power = spectrogram(eeg[0], fs=sr, nperseg=min(256, eeg.shape[1]))
    keep = freq <= 45
    axes[1].pcolormesh(sec, freq[keep], np.log1p(power[keep]), shading="auto")
    axes[1].set_ylabel("Hz")
    axes[2].scatter([25] * len(arrays["probabilities"]),
                    np.arange(len(arrays["probabilities"])),
                    c=arrays["probabilities"], cmap="viridis", vmin=0, vmax=1, s=100)
    axes[2].set_yticks(np.arange(len(CLASSES)))
    axes[2].set_yticklabels(CLASSES)
    axes[2].set_ylabel("P(class)")
    axes[3].plot(time + metadata["forecast_cut_s"], arrays["projected"][:, 0], "o-")
    axes[3].set_ylabel("PCA-1")
    errs = arrays["forecast_mse"]
    axes[4].scatter(arrays["future_times_s"] + metadata["forecast_cut_s"], errs)
    axes[4].set_ylabel("Graph MSE")
    axes[4].set_xlabel("Window time (s)")
    for ax in axes:
        ax.axvspan(20, 30, color="gray", alpha=.12)
        ax.set_xlim(0, 50)
    fig.suptitle("Classification EEG 0-50 s; forecast uses 0-5 s only; label 20-30 s")
    save(fig, "synchronized")

    fig, ax = plt.subplots(figsize=(7, 6))
    p, v = arrays["projected"], arrays["projected_field"]
    ax.plot(p[:, 0], p[:, 1], color="gray", lw=1)
    ax.scatter(p[:, 0], p[:, 1], c=time, cmap="viridis", s=45)
    ax.quiver(p[:, 0], p[:, 1], v[:, 0], v[:, 1],
              angles="xy", scale_units="xy", scale=None, color="tab:red")
    ax.annotate("start", p[0, :2]); ax.annotate("end", p[-1, :2])
    ax.set(xlabel="train-PCA 1", ylabel="train-PCA 2", title="Observed latent trajectory and du/dt")
    save(fig, "trajectory_direction")

    # A vector field is high dimensional. Show its local, linearly projected
    # values only at the states actually visited by this sample, never a
    # fabricated grid of off-trajectory EEG states.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    selected = np.unique(np.linspace(0, len(time) - 1, min(len(time), 16), dtype=int))
    sc = axes[0].scatter(p[:, 0], p[:, 1], c=time, cmap="viridis", s=25, zorder=2)
    axes[0].plot(p[:, 0], p[:, 1], color="#84929a", linewidth=1, zorder=1)
    axes[0].quiver(p[selected, 0], p[selected, 1], v[selected, 0], v[selected, 1],
                   angles="xy", scale_units="xy", scale=None, color="#a21d37",
                   width=.004, zorder=3)
    axes[0].annotate("t=0", p[0, :2], xytext=(7, 7), textcoords="offset points")
    axes[0].annotate("t=%.1f s" % time[-1], p[-1, :2], xytext=(7, -13),
                     textcoords="offset points")
    axes[0].set(xlabel="Train-only PCA 1", ylabel="Train-only PCA 2",
                title="Visited states and projected f(z)")
    fig.colorbar(sc, ax=axes[0], label="ODE time after observation cutoff (s)")
    axes[1].plot(time, arrays["speed_per_s"], color="#a21d37", lw=1.8)
    axes[1].axvspan(0, 1, color="#f1d9de", alpha=.8,
                    label="12-point class-head interval")
    for horizon in arrays["future_times_s"]:
        axes[1].axvline(float(horizon), color="#447c94", linestyle="--", alpha=.65)
    axes[1].set(xlabel="ODE time after observation cutoff (s)",
                ylabel="||f(z)|| over valid nodes", title="Speed along visited path")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Learned latent ODE; arrows are local derivatives on the observed trajectory")
    save(fig, "ode_field_trajectory")

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=False)
    axes[0].plot(time, arrays["speed_per_s"], "o-")
    axes[0].set(ylabel="||dz/dt||", xlabel="Forecast time (s)")
    axes[1].plot(time[1:], arrays["turn_angle_rad"], "o-")
    axes[1].set(ylabel="Turning angle (rad)", xlabel="Forecast time (s)")
    axes[2].bar(arrays["future_times_s"], errs)
    axes[2].set(ylabel="Graph node MSE", xlabel="Forecast horizon (s)")
    fig.suptitle("Model latent dynamics and masked node-attribute error")
    save(fig, "dynamics_quantities")

    pred = arrays["predicted_graph_nodes"]
    target = arrays["future_graph_nodes"]
    fig, axes = plt.subplots(2, len(pred), figsize=(4*len(pred), 6), squeeze=False)
    combined = np.concatenate([pred.reshape(-1), target.reshape(-1)])
    limit = np.percentile(np.abs(combined[np.isfinite(combined)]), 99) if np.isfinite(combined).any() else 1
    limit = max(limit, 1e-6)
    for i in range(len(pred)):
        for row, data, label in ((0, target, "Observed"), (1, pred, "Predicted")):
            axes[row, i].imshow(data[i], aspect="auto", vmin=-limit, vmax=limit, cmap="coolwarm")
            axes[row, i].set(title="%s +%gs" % (label, arrays["future_times_s"][i]),
                             xlabel="Band feature", ylabel="Electrode")
    fig.suptitle("Future node attributes; common color scale; no edge prediction")
    save(fig, "future_graph_nodes")

    observed_adj = arrays["graph_adj"]
    first_last = [0, len(observed_adj)-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, idx in zip(axes, first_last):
        ax.imshow(observed_adj[idx], cmap="coolwarm", vmin=-1, vmax=1)
        ax.set(title="Observed correlation, %d-%d s" % (idx, idx+1),
               xlabel="Electrode", ylabel="Electrode")
    fig.suptitle("Observed input correlation only; not predicted edges or causal links")
    save(fig, "observed_correlation")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = list(metadata["perturbations"])
    axes[0].bar(names, [metadata["perturbations"][n]["probability_l1"] for n in names])
    axes[0].set(ylabel="Probability L1 change", title="Input sensitivity")
    axes[1].bar(names, [metadata["perturbations"][n]["trajectory_l2_mean"] for n in names])
    axes[1].set(ylabel="Mean latent L2 change")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
    save(fig, "perturbation")
    return figures


def export_dynamics(checkpoint_path, manifest_csv, output_dir, device="cpu",
                    max_train_projection=256, max_per_group=1,
                    graph_normalizer=None, noise_fraction=0.02,
                    feature_fraction=0.01, seed=42, rk4_step_seconds=None):
    checkpoint_path = Path(checkpoint_path).resolve()
    manifest_csv = Path(manifest_csv).resolve()
    output_dir = Path(output_dir).resolve()
    if not checkpoint_path.is_file() or not manifest_csv.is_file():
        raise FileNotFoundError("Real checkpoint and patient split manifest are required")
    table = pd.read_csv(str(manifest_csv))
    required = {"split", "audit_reason", "window_id", "patient_id", "eeg_path", "class_id"}
    if not required.issubset(table.columns):
        raise ValueError("Manifest lacks patient split, audit, identity or labels")
    file_coverage = 0.0
    if "file_exists" in table.columns:
        file_coverage = float(table.groupby("eeg_id")["file_exists"].max().mean())
    source_metadata = Path(__file__).resolve().parents[2] / "data" / "raw" / "train.csv"
    full_metadata_rows = None
    if source_metadata.is_file():
        full_metadata_rows = len(pd.read_csv(str(source_metadata), usecols=["eeg_id"]))
    if not set(("train", "val", "test")).issubset(set(table["split"])):
        raise ValueError("Manifest needs train, val and test patient splits")
    valid = table[table.audit_reason == "ok"].copy()
    if valid.empty or any(not Path(p).is_file() for p in valid.eeg_path):
        raise FileNotFoundError("A valid manifest window has no real EEG parquet")
    model, checkpoint = load_trained_model(checkpoint_path, device, rk4_step_seconds)
    snapshot = checkpoint.get("config", {})
    expected_manifest_digest = snapshot.get("manifest_sha256")
    if expected_manifest_digest is not None and sha256_file(manifest_csv) != expected_manifest_digest:
        raise ValueError("Export manifest bytes differ from the classification checkpoint")
    if expected_manifest_digest is None and snapshot.get("smoke_subset") is False:
        raise ValueError("Formal classification checkpoint lacks a bound manifest digest")
    training_config = snapshot.get("training", {})
    if float(training_config.get("classification_weight", 0)) <= 0:
        raise ValueError("Classification head has no positive training loss weight")
    lineage = {"classification_checkpoint": sha256_file(checkpoint_path)}
    source_smoke_subset = False
    if float(training_config.get("forecast_weight", 0)) > 0:
        lineage["forecast_head_source"] = "same_checkpoint"
        if snapshot.get("schema_version") == "hms_final_refit_v1":
            raise ValueError("Final ODE classifier refit must verify its forecast pretraining source")
        forecast_config = snapshot
    else:
        source_checkpoint, source_lineage, source_smoke_subset = _verified_forecast_source(
            checkpoint_path, snapshot, model, manifest_csv, device, rk4_step_seconds)
        lineage.update(source_lineage)
        forecast_config = source_checkpoint.get("config", {})
    formal_ready = (file_coverage == 1.0 and full_metadata_rows == len(table)
                    and snapshot.get("smoke_subset") is False and not source_smoke_subset)
    provenance = "real_hms_checkpoint" if formal_ready else "real_hms_smoke"
    model_id = "odebrain_hms_sha256_" + sha256_file(checkpoint_path)[:16]
    split_hash = manifest_hash(table)
    training_manifest_name = snapshot.get("data", {}).get("manifest")
    if training_manifest_name:
        training_manifest = Path(training_manifest_name).expanduser()
        if not training_manifest.is_absolute():
            training_manifest = Path(__file__).resolve().parents[2] / training_manifest
        if not training_manifest.is_file() or manifest_hash(pd.read_csv(str(training_manifest))) != split_hash:
            raise ValueError("Export manifest differs from checkpoint training patient split")
    expected_normalizer = checkpoint_path.parent / "graph_normalizer.npz"
    if not expected_normalizer.is_file():
        raise FileNotFoundError("Finetune run directory lacks graph_normalizer.npz")
    if graph_normalizer is None:
        graph_normalizer = expected_normalizer
    graph_normalizer = Path(graph_normalizer).resolve()
    if not graph_normalizer.is_file() or sha256_file(graph_normalizer) != sha256_file(expected_normalizer):
        raise ValueError("Specified graph normalizer differs from the finetune run normalizer")
    with np.load(str(graph_normalizer), allow_pickle=False) as z:
        normalizer_mean = z["mean"].astype(np.float32)
        normalizer_std = z["std"].astype(np.float32)
    forecast_cache = forecast_config.get("data", {}).get("graph_cache_dir")
    if formal_ready and not forecast_cache:
        raise ValueError("Formal dynamics export requires the forecast eligibility cache")
    if forecast_cache:
        forecast_cache = Path(forecast_cache)
        if not forecast_cache.is_absolute():
            forecast_cache = Path(__file__).resolve().parents[2] / forecast_cache
        forecast_cache = forecast_cache.resolve()
    datasets = {split: HMSWindowDataset(str(manifest_csv), split, task="forecast",
                                        graph_mean=normalizer_mean, graph_std=normalizer_std,
                                        graph_cache_dir=forecast_cache)
                for split in ("train", "val", "test")}
    classification_cache = snapshot.get("data", {}).get("graph_cache_dir")
    if classification_cache:
        classification_cache = Path(classification_cache)
        if not classification_cache.is_absolute():
            classification_cache = Path(__file__).resolve().parents[2] / classification_cache
        classification_cache = classification_cache.resolve()
    class_datasets = {split: HMSWindowDataset(str(manifest_csv), split, task="classification",
                                             graph_mean=normalizer_mean, graph_std=normalizer_std,
                                             graph_cache_dir=classification_cache)
                      for split in ("val", "test")}
    for split in ("val", "test"):
        eligible_ids = set(datasets[split].rows.window_id.astype(str))
        class_datasets[split].rows = class_datasets[split].rows.loc[
            class_datasets[split].rows.window_id.astype(str).isin(eligible_ids)
        ].reset_index(drop=True)
        if len(class_datasets[split]) != len(eligible_ids):
            raise ValueError("Classification and forecast eligibility sets differ")
    train_ds = datasets["train"]
    if max_train_projection < 1:
        raise ValueError("max_train_projection must be positive")
    train_indices = sorted(range(len(train_ds)), key=lambda i: str(train_ds.rows.iloc[i].window_id))[:max_train_projection]
    train_states, train_ids = [], []
    for i in train_indices:
        sample = train_ds[i]
        _, d = _infer(model, sample, device)
        mask = _numpy(d["observed_node_mask"][0]).astype(bool)
        train_states.append(mean_valid_nodes(_numpy(d["trajectory"][0]), mask))
        train_ids.append(sample["window_id"])
    projector = TrainOnlyPCA.fit(train_states, model_id, split_hash, train_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    projector.save(output_dir / "train_pca.npz")
    records = []
    for split in ("val", "test"):
        ds = class_datasets[split]
        for i in range(len(ds)):
            class_sample = ds[i]
            probability = _class_infer(model, class_sample, device)
            votes = _numpy(class_sample["target"])
            records.append({"split": split, "dataset_index": i,
                            "window_id": class_sample["window_id"],
                            "expert_entropy": entropy(votes),
                            "correct": bool(int(np.argmax(probability)) == int(class_sample["class_id"])),
                            "predicted_class": int(np.argmax(probability)),
                            "true_class": int(class_sample["class_id"])})
    chosen = choose_cases(records, max_per_group)
    forecast_positions = {split: {str(window_id): position for position, window_id in
                                  enumerate(datasets[split].rows.window_id.astype(str))}
                          for split in ("val", "test")}
    cases = []
    for item in chosen:
        record, tags = item["record"], item["tags"]
        forecast_index = forecast_positions[record["split"]][str(record["window_id"])]
        sample = datasets[record["split"]][forecast_index]
        class_sample = class_datasets[record["split"]][record["dataset_index"]]
        if class_sample["window_id"] != sample["window_id"]:
            raise ValueError("Classification and forecast datasets have different window order")
        probability = _class_infer(model, class_sample, device)
        _, d = _infer(model, sample, device)
        mask = _numpy(d["observed_node_mask"][0]).astype(bool)
        trajectory = _numpy(d["trajectory"][0])
        field = _numpy(d["vector_field"][0])
        times_s = _numpy(d["times_s"])
        summary = trajectory_summary(times_s, trajectory, field, mask, projector)
        prediction = _numpy(d["predicted_graph_nodes"][0])
        target = _numpy(sample["future_graph_nodes"])
        future_mask = _numpy(d["future_graph_mask"][0]).astype(bool)
        horizons = _numpy(sample["future_times_s"])
        errors = forecast_errors(prediction, target, future_mask, horizons)
        variants, dropped_channel = _graph_perturbations(sample, model, device,
                                                          normalizer_mean, normalizer_std,
                                                          noise_fraction, feature_fraction,
                                                          seed)
        class_variants, _ = _graph_perturbations(class_sample, model, device,
                                                  normalizer_mean, normalizer_std,
                                                  noise_fraction, feature_fraction,
                                                  seed)
        perturbations = {}
        for name, variant in variants.items():
            p2 = _class_infer(model, class_variants[name], device)
            _, d2 = _infer(model, variant, device)
            z2 = _numpy(d2["trajectory"][0])
            e2 = forecast_errors(_numpy(d2["predicted_graph_nodes"][0]),
                                 target, future_mask, horizons)
            perturbations[name] = {
                "probability_l1": float(np.abs(p2 - probability).sum()),
                "trajectory_l2_mean": float(np.linalg.norm((z2 - trajectory).reshape(len(times_s), -1), axis=1).mean()),
                "forecast_mse_by_horizon": [x["mse"] for x in e2],
            }
        case_id = "%s_%s_%s" % ("formal" if formal_ready else "smoke", record["split"],
                                 str(sample["window_id"]).replace("/", "_"))
        case_dir = output_dir / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "observed_eeg": _numpy(sample["observed_eeg"]),
            "classification_eeg": _numpy(class_sample["eeg"]),
            "graph_nodes": _numpy(sample["graph_nodes"]),
            "graph_adj": _numpy(sample["graph_adj"]),
            "graph_mask": _numpy(sample["graph_mask"]),
            "times_s": times_s, "trajectory": trajectory, "vector_field": field,
            "projected": summary["projected"], "projected_field": summary["projected_field"],
            "speed_per_s": summary["speed_per_s"], "turn_angle_rad": summary["turn_angle_rad"],
            "predicted_graph_nodes": prediction, "future_graph_nodes": target,
            "future_graph_mask": future_mask, "future_times_s": horizons,
            "forecast_mse": np.array([e["mse"] if e["mse"] is not None else np.nan for e in errors]),
            "probabilities": probability, "expert_votes": _numpy(sample["target"]),
        }
        np.savez_compressed(str(case_dir / "case.npz"), **arrays)
        meta = {
            "schema_version": "ode_case_v1", "provenance": provenance,
            "architecture": model.architecture,
            "model_id": model_id, "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_lineage": lineage,
            "manifest_file_coverage": file_coverage,
            "manifest_rows": int(len(table)), "source_metadata_rows": full_metadata_rows,
            "graph_normalizer_sha256": sha256_file(graph_normalizer),
            "split_hash": split_hash, "split": record["split"],
            "window_id": sample["window_id"], "patient_id": str(sample["patient_id"]),
            "eeg_id": str(datasets[record["split"]].rows.iloc[forecast_index].eeg_id),
            "selection_tags": tags, "selection_rule": "validation-prespecified entropy extrema and correct/error, applied to each split",
            "forecast_cut_s": int(datasets[record["split"]].cut),
            "sample_rate_hz": int(sample["sample_rate"]), "electrodes": list(ELECTRODES),
            "class_names": list(CLASSES), "label_region_window_s": [20, 30],
            "label_note": "Expert votes cover 20-30 s only; forecast input ends at 5 s",
            "probability_context": "one offline classification prediction from the full 0-50 s window; not per-second supervision",
            "forecast_context": "separate strict forecast branch sees 0-5 s only",
            "predicted_class": record["predicted_class"], "true_class": record["true_class"],
            "prediction_entropy": entropy(probability), "expert_entropy": record["expert_entropy"],
            "forecast_errors": errors, "path_length_integral": summary["path_length_integral"],
            "sampled_chord_length": summary["sampled_chord_length"],
            "mean_turn_angle_rad": summary["mean_turn_angle_rad"],
            "near_zero_turn_pairs": summary["near_zero_turn_pairs"],
            "nfe_including_export_field_evals": int(d["nfe"]),
            "pca_train_window_count": len(train_ids),
            "pca_explanation": "PCA fitted only on train trajectories of this model; projected arrows are W f(z,t), at observed states",
            "time_unit": "physical seconds", "graph_head": "future node attributes only; no predicted adjacency",
            "perturbation_config": {"noise_rms_fraction": noise_fraction,
                                    "feature_scale_fraction": feature_fraction,
                                    "dropped_electrode": ELECTRODES[dropped_channel], "seed": seed},
            "perturbations": perturbations,
        }
        meta["figures"] = _write_case_figures(case_dir, arrays, meta)
        (case_dir / "case.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        cases.append({"id": case_id, "split": record["split"], "window_id": sample["window_id"],
                      "patient_id": str(sample["patient_id"]), "tags": tags,
                      "metadata": str((case_dir / "case.json").relative_to(output_dir)),
                      "data": str((case_dir / "case.npz").relative_to(output_dir))})
    index = {"schema_version": "ode_results_v1", "provenance": provenance,
             "architecture": model.architecture,
             "model_id": model_id, "split_hash": split_hash,
             "manifest_file_coverage": file_coverage,
             "manifest_rows": int(len(table)), "source_metadata_rows": full_metadata_rows,
             "checkpoint_sha256": sha256_file(checkpoint_path),
             "training_lineage": lineage,
             "forecast_eligibility": (datasets["test"].graph_cache.eligibility_summary
                                      if datasets["test"].graph_cache is not None else None),
             "graph_normalizer_sha256": sha256_file(graph_normalizer),
             "pca_file": "train_pca.npz", "pca_train_window_count": len(train_ids),
             "selection_rule": "For each val/test split: min/max expert entropy, min entropy correct and min entropy error; window ID tie break",
             "cases": cases}
    (output_dir / "result_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return index
