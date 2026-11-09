"""Small, resumable training loop for six-class soft HMS labels.

Only observed graph history is passed into model forward. Future graph nodes
may be present in the batch as loss targets. Checkpoints include optimizer,
scheduler, RNG state, and a configuration snapshot for exact local resumption.
"""

import json
import inspect
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def soft_target_kl(logits: Tensor, target: Tensor) -> Tensor:
    """Mean KL(target || softmax(logits)) over examples."""
    if logits.ndim != 2 or target.shape != logits.shape:
        raise ValueError("logits and target must both have shape [B,num_classes]")
    if not bool(torch.isfinite(target).all()) or not bool((target >= 0).all()):
        raise ValueError("target must contain finite, nonnegative probabilities")
    if not bool(torch.allclose(target.sum(-1), torch.ones_like(target[:, 0]), atol=1e-4)):
        raise ValueError("every target row must sum to one")
    return F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")


def _unpack(output: Any) -> Tuple[Tensor, Mapping[str, Tensor]]:
    if isinstance(output, Tensor):
        return output, {}
    if isinstance(output, tuple) and len(output) == 2:
        logits, extras = output
        if not isinstance(logits, Tensor) or not isinstance(extras, Mapping):
            raise TypeError("model must return logits or (logits, extras)")
        return logits, extras
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"], output
    raise TypeError("model must return logits, (logits, extras), or a logits dict")


def _forecast_loss(prediction: Tensor, batch: Mapping[str, Any]) -> Tuple[Tensor, int]:
    if "future_graph_nodes" not in batch:
        raise KeyError("forecast loss needs future_graph_nodes")
    target = batch["future_graph_nodes"]
    if prediction.shape != target.shape:
        raise ValueError("future_graph_nodes_pred and future_graph_nodes shapes differ")
    if prediction.ndim != 4:
        raise ValueError("graph forecast must have shape [B,H,N,F]")
    mask = batch.get("future_graph_mask")
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    elif mask.shape == target.shape[:-1]:
        mask = mask.unsqueeze(-1).expand_as(target)
    elif mask.shape != target.shape:
        raise ValueError("future_graph_mask must be [B,H,N] or [B,H,N,F]")
    mask = mask.bool()
    count = int(mask.sum().item())
    if count == 0:
        return prediction.sum() * 0, 0
    if not bool(torch.isfinite(target[mask]).all()):
        raise ValueError("valid forecast targets must be finite")
    squared_error = (prediction - torch.where(mask, target, torch.zeros_like(target))).square()
    return squared_error.masked_select(mask).mean(), count


def _to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
            for key, value in batch.items()}


def _run_epoch(model: nn.Module, batches: Iterable[Mapping[str, Any]],
               device: Union[torch.device, str], optimizer: Optional[torch.optim.Optimizer],
               forecast_weight: float, grad_clip: Optional[float],
               classification_weight: float) -> Dict[str, Any]:
    if forecast_weight < 0 or classification_weight < 0 or not (forecast_weight or classification_weight):
        raise ValueError("loss weights must be nonnegative and at least one positive")
    device = torch.device(device)
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "kl": 0.0, "forecast_mse": 0.0}
    examples = 0
    forecast_elements = 0
    with torch.set_grad_enabled(training):
        for raw_batch in batches:
            batch = _to_device(raw_batch, device)
            target = batch.get("target")
            if classification_weight > 0 and not isinstance(target, Tensor):
                raise KeyError("classification training requires tensor target [B,6]")
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(batch, return_dynamics=forecast_weight > 0)
            logits, extras = _unpack(output)
            kl = soft_target_kl(logits, target) if classification_weight > 0 else logits.sum() * 0
            if forecast_weight > 0:
                prediction = extras.get("future_graph_nodes_pred")
                if prediction is None:
                    prediction = extras.get("predicted_graph_nodes")
                if prediction is None:
                    raise KeyError("model extras require predicted graph nodes")
                mse, count = _forecast_loss(prediction, batch)
                if count == 0 and classification_weight == 0:
                    raise ValueError("forecast pretraining batch has no valid targets")
            else:
                mse, count = kl.new_zeros(()), 0
            loss = classification_weight * kl + forecast_weight * mse
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite training loss")
            if training:
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip,
                                             error_if_nonfinite=True)
                optimizer.step()
            size = int(logits.shape[0])
            examples += size
            totals["loss"] += float(loss.detach()) * size
            totals["kl"] += float(kl.detach()) * size
            totals["forecast_mse"] += float(mse.detach()) * count
            forecast_elements += count
    if examples == 0:
        raise ValueError("training/evaluation loader is empty")
    return {"loss": totals["loss"] / examples,
            "kl": totals["kl"] / examples if classification_weight > 0 else None,
            "forecast_mse": totals["forecast_mse"] / forecast_elements
            if forecast_elements else None,
            "examples": examples, "forecast_elements": forecast_elements}


def train_epoch(model: nn.Module, batches: Iterable[Mapping[str, Any]],
                optimizer: torch.optim.Optimizer, device: Union[torch.device, str] = "cpu",
                forecast_weight: float = 0.0,
                grad_clip: Optional[float] = 1.0,
                classification_weight: float = 1.0) -> Dict[str, Any]:
    return _run_epoch(model, batches, device, optimizer, forecast_weight, grad_clip,
                      classification_weight)


def evaluate_epoch(model: nn.Module, batches: Iterable[Mapping[str, Any]],
                   device: Union[torch.device, str] = "cpu",
                   forecast_weight: float = 0.0,
                   classification_weight: float = 1.0) -> Dict[str, Any]:
    return _run_epoch(model, batches, device, None, forecast_weight, None,
                      classification_weight)


def save_checkpoint(path: Union[str, Path], model: nn.Module,
                    optimizer: torch.optim.Optimizer, epoch: int,
                    best_val_score: float, config: Mapping[str, Any],
                    scheduler: Any = None, bad_epochs: int = 0,
                    monitor: str = "kl") -> None:
    """Atomically save a trusted local training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": int(epoch), "best_val_score": float(best_val_score),
        "monitor": monitor,
        "bad_epochs": int(bad_epochs),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": dict(config), "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Union[str, Path], model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Any = None, map_location: Union[str, torch.device] = "cpu",
                    restore_rng: bool = True) -> Dict[str, Any]:
    """Restore a checkpoint produced by this project; only load trusted files."""
    load_kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    checkpoint = torch.load(path, **load_kwargs)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if restore_rng:
        torch.set_rng_state(checkpoint["rng_cpu"].cpu())
        if torch.cuda.is_available() and checkpoint.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
    return checkpoint


def fit(model: nn.Module, train_batches: Iterable[Mapping[str, Any]],
        val_batches: Iterable[Mapping[str, Any]], optimizer: torch.optim.Optimizer,
        run_dir: Union[str, Path], config: Mapping[str, Any], max_epochs: int,
        device: Union[torch.device, str] = "cpu", forecast_weight: float = 0.0,
        patience: int = 8, grad_clip: Optional[float] = 1.0,
        scheduler: Any = None, resume_from: Optional[Union[str, Path]] = None,
        classification_weight: float = 1.0) -> List[Dict[str, Any]]:
    """Train with validation KL or graph MSE early stopping and checkpoints."""
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    model.to(device)
    monitor = "kl" if classification_weight > 0 else "forecast_mse"
    start_epoch, best_score, bad_epochs = 0, float("inf"), 0
    if resume_from is not None:
        if Path(resume_from).resolve().parent != directory.resolve():
            raise ValueError("Resume checkpoint must be in the requested run directory")
        checkpoint = load_checkpoint(resume_from, model, optimizer, scheduler,
                                     map_location=device)
        if checkpoint.get("monitor") != monitor:
            raise ValueError("checkpoint monitor differs from requested training stage")
        if checkpoint.get("config") != dict(config):
            raise ValueError("Resume checkpoint config or data snapshot differs from this run")
        if not (directory / "best.pt").is_file():
            raise FileNotFoundError("Resume run directory lacks its best.pt checkpoint")
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_val_score"])
        bad_epochs = int(checkpoint.get("bad_epochs", 0))
    history = []  # type: List[Dict[str, Any]]
    for epoch in range(start_epoch, max_epochs):
        train = train_epoch(model, train_batches, optimizer, device,
                            forecast_weight, grad_clip, classification_weight)
        val = evaluate_epoch(model, val_batches, device, forecast_weight,
                             classification_weight)
        if scheduler is not None:
            scheduler.step()
        if val[monitor] is None:
            raise ValueError(f"validation {monitor} unavailable")
        improved = val[monitor] < best_score
        if improved:
            best_score, bad_epochs = val[monitor], 0
        else:
            bad_epochs += 1
        row = {"epoch": epoch, "train": train, "val": val,
               "monitor": monitor, "best_val_score": best_score,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        with (directory / "history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, allow_nan=False) + "\n")
        save_checkpoint(directory / "latest.pt", model, optimizer, epoch,
                        best_score, config, scheduler, bad_epochs, monitor)
        if improved:
            save_checkpoint(directory / "best.pt", model, optimizer, epoch,
                            best_score, config, scheduler, bad_epochs, monitor)
        if bad_epochs >= patience:
            break
    return history
