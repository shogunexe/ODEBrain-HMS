"""Project entry points. Run from the project root with Python 3.8."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def read_config(path):
    import yaml
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _print(data):
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def doctor(args):
    import importlib.util
    import platform
    import shutil
    import subprocess
    info = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "cwd": str(Path.cwd()), "python": sys.version.split()[0],
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "packages": {x: importlib.util.find_spec(x) is not None for x in
                         ("numpy", "pandas", "pyarrow", "pyspark", "torch", "scipy", "sklearn", "streamlit", "plotly")},
            "executables": {x: shutil.which(x) for x in ("java", "spark-submit", "hdfs", "nvidia-smi", "libreoffice")},
            "kaggle_credentials_available": bool(os.environ.get("KAGGLE_KEY") or Path.home().joinpath(".kaggle/kaggle.json").exists())}
    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        info["gpu_query"] = result.stdout.strip().splitlines() if result.returncode == 0 else []
    _print(info)
    return 0


def audit(args):
    from src.data.hms import audit_manifest, audit_window_quality, manifest_from_csv, write_json
    config = read_config(args.config)
    table = manifest_from_csv(config["metadata_csv"], config["eeg_dir"])
    if args.quality:
        table = audit_window_quality(table, max_missing_fraction=float(config.get("max_missing_fraction", .05)),
                                     workers=int(args.workers))
    Path(config["manifest_csv"]).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(config["manifest_csv"], index=False)
    result = audit_manifest(table)
    write_json(config["audit_json"], result)
    _print(result)
    return 0


def split(args):
    import pandas as pd
    from src.data.splits import assign_patient_splits, save_split
    config = read_config(args.config)
    table = pd.read_csv(config["manifest_csv"])
    split_table = assign_patient_splits(table, seed=int(config.get("seed", 42)),
                                        test_fraction=float(config.get("test_fraction", .2)),
                                        val_fraction_of_dev=float(config.get("val_fraction_of_dev", .2)),
                                        folds=int(config.get("folds", 3)))
    result = save_split(split_table, config["split_csv"], seed=int(config.get("seed", 42)))
    _print(result)
    return 0


def make_model(config):
    name = config["model"]["name"]
    kwargs = config["model"]["kwargs"]
    if name == "odebrain_inspired_hms":
        from src.models.odebrain_hms import ODEBrainInspiredHMS
        return ODEBrainInspiredHMS(**kwargs)
    from src.models.baselines import build_model
    return build_model(name, **kwargs)


def input_modalities(model_name):
    if model_name == "tcn":
        return False, False
    if model_name in ("spectrogram_cnn", "fusion"):
        return True, False
    if model_name in ("graph_gru", "odebrain_inspired_hms"):
        return False, True
    raise ValueError("Unknown model input requirements: " + model_name)


def require_full_data(manifest_path, metadata_path="data/raw/train.csv"):
    """Refuse to label an incomplete or unaudited subset as a formal run."""
    import pandas as pd
    manifest = pd.read_csv(manifest_path)
    metadata = pd.read_csv(metadata_path, usecols=["eeg_id"])
    if len(manifest) != len(metadata) or not manifest.eeg_id.value_counts().sort_index().equals(metadata.eeg_id.value_counts().sort_index()):
        raise ValueError("Formal run requires a manifest covering every train.csv row and EEG ID")
    required = {"eeg_path", "file_exists", "audit_reason", "missing_fraction", "raw_recording_seconds"}
    if not required.issubset(manifest.columns):
        raise ValueError("Formal run requires a quality-audited manifest")
    if not manifest.file_exists.fillna(False).all():
        raise ValueError("Formal run requires every EEG parquet; rerun audit after download")
    if manifest.audit_reason.isin(("missing_eeg_file", "unreadable_eeg_file")).any():
        raise ValueError("Formal run contains missing or unreadable EEG recordings")
    if not all(Path(path).is_file() for path in manifest.eeg_path.dropna().unique()):
        raise ValueError("Formal run requires EEG files still present at audited paths")
    valid = manifest.audit_reason == "ok"
    if not valid.any() or manifest.loc[valid, ["missing_fraction", "raw_recording_seconds"]].isna().any().any():
        raise ValueError("Formal run requires completed quality checks for valid EEG windows")


def _datasets(config, smoke=False, normalizer=None):
    import numpy as np
    from src.data.dataset import HMSWindowDataset, fit_graph_normalizer
    path = config.get("data", {}).get("manifest", "data/curated/split_manifest.csv")
    task = config.get("data", {}).get("task", "classification")
    include_spectrogram, include_graph = input_modalities(config["model"]["name"])
    cache_dir = config.get("data", {}).get("graph_cache_dir")
    # Forecast smoke must sample eligible histories; otherwise its first
    # patient-stratified rows can fail before the model is exercised.
    smoke_cache = cache_dir if (smoke and task == "forecast" and cache_dir and
                                Path(cache_dir).is_dir()) else None
    kwargs = {"task": task, "max_rows": 12 if smoke else None,
              "forecast_cut_seconds": config.get("data", {}).get("forecast_cut_seconds", 5),
              "forecast_horizons": config.get("data", {}).get("forecast_horizons", (1, 3, 5)),
              "include_spectrogram": include_spectrogram, "include_graph": include_graph,
              "graph_cache_dir": smoke_cache if smoke else cache_dir}
    train = HMSWindowDataset(path, "train", **kwargs)
    val = HMSWindowDataset(path, "val", **kwargs)
    if len(train) == 0 or len(val) == 0:
        raise ValueError("Train or validation split is empty")
    # Fit only on observed train data; small smoke runs use the same small subset.
    if include_graph:
        mean, std = normalizer if normalizer is not None else fit_graph_normalizer(train)
    else:
        mean, std = np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.float32)
    for ds in (train, val):
        ds.graph_mean = mean
        ds.graph_std = std
    return train, val, mean, std


def train(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from src.data.dataset import RecordingBatchSampler, hms_collate
    from src.training.engine import fit, load_checkpoint
    config = read_config(args.config)
    manifest_path = config.get("data", {}).get("manifest", "data/curated/split_manifest.csv")
    if not args.smoke:
        require_full_data(manifest_path)
    config["smoke_subset"] = bool(args.smoke)
    config["source_config_sha256"] = file_sha256(args.config)
    config["manifest_sha256"] = file_sha256(manifest_path)
    if args.init_checkpoint:
        checkpoint_path = Path(args.init_checkpoint).resolve()
        config["init_checkpoint"] = str(checkpoint_path)
        config["init_checkpoint_sha256"] = file_sha256(checkpoint_path)
    if args.graph_normalizer:
        normalizer_source = Path(args.graph_normalizer).resolve()
        source_config_path = normalizer_source.parent / "config.json"
        if not normalizer_source.is_file() or not source_config_path.is_file():
            raise FileNotFoundError("Normalizer source requires graph_normalizer.npz and config.json")
        source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
        source_data = source_config.get("data", {})
        target_data = config.get("data", {})
        same_task = (source_data.get("task", "classification") ==
                     target_data.get("task", "classification") and
                     source_data.get("graph_cache_dir") == target_data.get("graph_cache_dir"))
        forecast_to_class = (source_data.get("task") == "forecast" and
                             target_data.get("task", "classification") == "classification" and
                             source_data.get("graph_cache_dir") == "data/training/graph_forecast_v2" and
                             target_data.get("graph_cache_dir") == "data/training/graph_classification")
        same_window_rules = all(source_data.get(key) == target_data.get(key)
                                for key in ("forecast_cut_seconds", "forecast_horizons"))
        if (source_config.get("manifest_sha256") != config["manifest_sha256"] or
                not same_window_rules or not (same_task or forecast_to_class)):
            raise ValueError("Normalizer source was fitted for another manifest or graph task")
        config["normalizer_source_path"] = str(normalizer_source)
        config["normalizer_source_sha256"] = file_sha256(normalizer_source)
    training = config["training"]
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    init_normalizer = None
    if args.init_checkpoint:
        source = Path(args.init_checkpoint).parent / "graph_normalizer.npz"
        if source.is_file():
            values = np.load(source)
            init_normalizer = values["mean"], values["std"]
    if args.graph_normalizer:
        with np.load(config["normalizer_source_path"], allow_pickle=False) as values:
            init_normalizer = values["mean"].copy(), values["std"].copy()
    run_dir = Path(args.run_dir or ("artifacts/runs/" + config["model"]["name"] + "_" + time.strftime("%Y%m%d_%H%M%S")))
    existing_config = run_dir / "config.json"
    existing_normalizer = run_dir / "graph_normalizer.npz"
    if existing_config.is_file() and existing_normalizer.is_file():
        previous = json.loads(existing_config.read_text(encoding="utf-8"))
        before_snapshot = {k: v for k, v in previous.items()
                           if k != "window_snapshot_sha256"}
        if before_snapshot != config:
            raise ValueError("Existing run directory has a different config or manifest")
        with np.load(existing_normalizer, allow_pickle=False) as values:
            init_normalizer = values["mean"].copy(), values["std"].copy()
    train_data, val_data, mean, std = _datasets(config, args.smoke, normalizer=init_normalizer)
    run_dir.mkdir(parents=True, exist_ok=True)
    window_snapshot = {name: dataset.rows.window_id.astype(str).tolist()
                       for name, dataset in (("train", train_data), ("val", val_data))}
    snapshot_bytes = json.dumps(window_snapshot, sort_keys=True).encode("utf-8")
    config["window_snapshot_sha256"] = hashlib.sha256(snapshot_bytes).hexdigest()
    config_path = run_dir / "config.json"
    snapshot_path = run_dir / "window_snapshot.json"
    if config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing run directory has a different config or manifest")
    if snapshot_path.is_file() and snapshot_path.read_bytes() != snapshot_bytes:
        raise ValueError("Existing run directory has different training windows")
    if not args.resume and ((run_dir / "best.pt").exists() or (run_dir / "latest.pt").exists()):
        raise FileExistsError("Run directory already has checkpoints; use --resume or a new directory")
    if not snapshot_path.exists():
        snapshot_path.write_bytes(snapshot_bytes)
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    normalizer_path = run_dir / "graph_normalizer.npz"
    if normalizer_path.is_file():
        with np.load(normalizer_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["mean"], mean) or not np.array_equal(saved["std"], std):
                raise ValueError("Existing run directory has a different graph normalizer")
    else:
        np.savez(normalizer_path, mean=mean, std=std)
    batch_size = min(int(training.get("batch_size", 8)), 2) if args.smoke else int(training.get("batch_size", 8))
    common = {"num_workers": int(args.workers), "collate_fn": hms_collate}
    train_loader = DataLoader(train_data, batch_sampler=RecordingBatchSampler(train_data, batch_size), **common)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, **common)
    model = make_model(config)
    if args.init_checkpoint:
        source_checkpoint = load_checkpoint(args.init_checkpoint, model, optimizer=None, restore_rng=False)
        source_config = source_checkpoint.get("config", {})
        if source_config.get("manifest_sha256") != config["manifest_sha256"]:
            raise ValueError("Pretrain checkpoint and finetune run use different manifests")
        if not args.smoke and source_config.get("smoke_subset") is not False:
            raise ValueError("Formal finetune cannot initialize from a smoke checkpoint")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.get("lr", .001)), weight_decay=float(training.get("weight_decay", .01)))
    history = fit(model, train_loader, val_loader, optimizer, run_dir, config,
                  max_epochs=2 if args.smoke else int(training.get("epochs", 30)),
                  device=device, classification_weight=float(training.get("classification_weight", 1)),
                  forecast_weight=float(training.get("forecast_weight", 0)),
                  patience=int(training.get("patience", 8)),
                  grad_clip=float(training.get("grad_clip", 1)), resume_from=args.resume)
    completion = {"source_config_sha256": config["source_config_sha256"],
                  "manifest_sha256": config["manifest_sha256"],
                  "window_snapshot_sha256": config["window_snapshot_sha256"],
                  "best_checkpoint_sha256": file_sha256(run_dir / "best.pt"),
                  "max_epochs": 2 if args.smoke else int(training.get("epochs", 30)),
                  "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    completion_path = run_dir / "completed.json"
    temporary = run_dir / "completed.json.tmp"
    temporary.write_text(json.dumps(completion, indent=2), encoding="utf-8")
    temporary.replace(completion_path)
    _print({"run_dir": str(run_dir), "device": device, "epochs_ran": len(history),
            "best_checkpoint": str(run_dir / "best.pt"), "synthetic": False,
            "smoke_subset": bool(args.smoke)})
    return 0


def evaluate(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from src.data.dataset import HMSWindowDataset, hms_collate
    from src.evaluation.metrics import classification_metrics, patient_bootstrap_kl
    from src.training.engine import load_checkpoint
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if not config.get("smoke_subset", False):
        require_full_data(config.get("data", {}).get("manifest", "data/curated/split_manifest.csv"))
    if config.get("data", {}).get("task", "classification") != "classification":
        raise ValueError("Forecast-only run has no valid classification evaluation")
    model = make_model(config)
    checkpoint = load_checkpoint(run_dir / "best.pt", model, optimizer=None, restore_rng=False)
    if checkpoint.get("config") != config:
        raise ValueError("Run config differs from the checkpoint config")
    manifest_path = config.get("data", {}).get("manifest", "data/curated/split_manifest.csv")
    expected_manifest_hash = config.get("manifest_sha256")
    if expected_manifest_hash is None and not config.get("smoke_subset", False):
        raise ValueError("Formal checkpoint lacks a bound manifest hash")
    if expected_manifest_hash is not None and file_sha256(manifest_path) != expected_manifest_hash:
        raise ValueError("Current manifest differs from the training checkpoint")
    snapshot_hash = config.get("window_snapshot_sha256")
    snapshot_path = run_dir / "window_snapshot.json"
    if snapshot_hash is not None and (not snapshot_path.is_file() or file_sha256(snapshot_path) != snapshot_hash):
        raise ValueError("Training window snapshot is missing or changed")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    normalizer = np.load(run_dir / "graph_normalizer.npz")
    data = HMSWindowDataset(config.get("data", {}).get("manifest", "data/curated/split_manifest.csv"),
                            args.split, graph_mean=normalizer["mean"], graph_std=normalizer["std"],
                            graph_cache_dir=None if config.get("smoke_subset", False) else config.get("data", {}).get("graph_cache_dir"),
                            include_spectrogram=input_modalities(config["model"]["name"])[0],
                            include_graph=input_modalities(config["model"]["name"])[1])
    evaluation_scope = "full_split"
    if config.get("smoke_subset", False):
        if args.split != "val":
            raise ValueError("Smoke runs only support their frozen validation subset")
        if snapshot_hash is None:
            evaluation_scope = "legacy_smoke_full_val"
        else:
            selected = json.loads(snapshot_path.read_text(encoding="utf-8"))["val"]
            lookup = data.rows.set_index(data.rows.window_id.astype(str), drop=False)
            if any(window_id not in lookup.index for window_id in selected):
                raise ValueError("Frozen smoke validation window missing from current manifest")
            data.rows = lookup.loc[selected].reset_index(drop=True)
            evaluation_scope = "frozen_smoke_val"
    loader = DataLoader(data, batch_size=int(args.batch_size), num_workers=int(args.workers), collate_fn=hms_collate)
    target, probs, hard, patients, windows = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            device_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            output = model(device_batch)
            logits = output[0] if isinstance(output, tuple) else output
            probs.append(torch.softmax(logits, -1).cpu().numpy())
            target.append(batch["target"].numpy())
            hard.extend(batch["class_id"].tolist())
            patients.extend(batch["patient_id"])
            windows.extend(batch["window_id"])
    if not probs:
        raise ValueError("Evaluation split empty")
    target = np.concatenate(target)
    probs = np.concatenate(probs)
    result = classification_metrics(target, probs, hard, patients)
    result["kl_patient_bootstrap"] = patient_bootstrap_kl(target, probs, patients, n_boot=1000)
    from src.data.hms import manifest_hash
    import pandas as pd
    full_manifest = pd.read_csv(config.get("data", {}).get("manifest", "data/curated/split_manifest.csv"))
    result.update({"split": args.split, "run_dir": str(run_dir), "model": config["model"]["name"],
                   "split_hash": manifest_hash(full_manifest),
                   "smoke_subset": bool(config.get("smoke_subset", False)),
                   "evaluation_scope": evaluation_scope,
                   "metadata_eeg_coverage": float(full_manifest.loc[full_manifest.audit_reason == "ok", "eeg_id"].nunique() / max(1, full_manifest.eeg_id.nunique()))})
    out_dir = Path("results") / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / (args.split + "_metrics.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    frame = pd.DataFrame({"window_id": windows, "patient_id": patients, "class_id": hard})
    for i, name in enumerate(("seizure", "lpd", "gpd", "lrda", "grda", "other")):
        frame["target_" + name] = target[:, i]
        frame["pred_" + name] = probs[:, i]
    frame.to_csv(out_dir / (args.split + "_predictions.csv"), index=False)
    _print(result)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    p = sub.add_parser("audit"); p.add_argument("--config", default="configs/data.yaml"); p.add_argument("--quality", action="store_true"); p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("split"); p.add_argument("--config", default="configs/split.yaml")
    p = sub.add_parser("train")
    p.add_argument("--config", required=True); p.add_argument("--run-dir"); p.add_argument("--device")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--workers", type=int, default=0)
    p.add_argument("--resume"); p.add_argument("--init-checkpoint"); p.add_argument("--smoke", action="store_true")
    p.add_argument("--graph-normalizer", help="Reuse a verified normalizer from a run with the same manifest and graph task")
    p = sub.add_parser("evaluate")
    p.add_argument("--run-dir", required=True); p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--device"); p.add_argument("--batch-size", type=int, default=8); p.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    return globals()[args.command.replace("-", "_")](args)


if __name__ == "__main__":
    sys.exit(main())
