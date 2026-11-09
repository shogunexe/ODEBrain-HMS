"""Offline metrics; all KL aggregation uses float64."""
import numpy as np
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

from src.data.hms import CLASSES


def normalize_probabilities(values, eps=1e-15):
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != len(CLASSES) or not np.isfinite(p).all():
        raise ValueError("Expected finite [N,6] probability matrix")
    p = np.clip(p, eps, None)
    return p / p.sum(axis=1, keepdims=True)


def normalize_targets(values):
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != len(CLASSES) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Expected nonnegative finite [N,6] targets")
    totals = p.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Targets must have positive vote sum")
    return p / totals


def sample_kl(target, predicted):
    p = normalize_targets(target)
    q = normalize_probabilities(predicted)
    return np.sum(p * (np.log(np.maximum(p, 1e-300)) - np.log(q)), axis=1)


def classification_metrics(target, predicted, hard_labels, patient_ids):
    p = normalize_targets(target)
    q = normalize_probabilities(predicted)
    y = np.asarray(hard_labels, dtype=int)
    patients = np.asarray(patient_ids).astype(str)
    if len(p) != len(q) or len(p) != len(y) or len(p) != len(patients):
        raise ValueError("Prediction and identity lengths differ")
    pred = q.argmax(axis=1)
    kl = sample_kl(p, q)
    patient_kl = [float(kl[patients == pid].mean()) for pid in np.unique(patients)]
    prec, recall, f1, support = precision_recall_fscore_support(y, pred, labels=np.arange(6), zero_division=0)
    auc = {}
    for i, name in enumerate(CLASSES):
        truth = (y == i).astype(int)
        auc[name] = None if len(np.unique(truth)) < 2 else float(roc_auc_score(truth, q[:, i]))
    return {
        "n_windows": int(len(y)), "n_patients": int(len(patient_kl)),
        "kl_window_mean": float(kl.mean()), "kl_patient_equal_mean": float(np.mean(patient_kl)),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(6), average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(6)).tolist(),
        "per_class": {name: {"precision": float(prec[i]), "recall": float(recall[i]),
                             "f1": float(f1[i]), "support": int(support[i]), "ovr_auroc": auc[name]}
                      for i, name in enumerate(CLASSES)},
    }


def patient_bootstrap_kl(target, predicted, patient_ids, n_boot=1000, seed=42):
    """Sample patients with replacement, keeping all their windows together."""
    kl = sample_kl(target, predicted)
    patients = np.asarray(patient_ids).astype(str)
    unique = np.unique(patients)
    if len(unique) < 2:
        raise ValueError("At least two patients required for bootstrap")
    per_patient = np.array([kl[patients == pid].mean() for pid in unique])
    rng = np.random.RandomState(seed)
    indices = rng.randint(len(unique), size=(n_boot, len(unique)))
    samples = per_patient[indices].mean(axis=1)
    return {"patient_equal_mean": float(per_patient.mean()), "ci95": np.percentile(samples, [2.5, 97.5]).tolist(),
            "resampling_unit": "patient", "n_boot": n_boot, "seed": seed}


def forecast_metrics(target, predicted, mask, horizons=(1, 3, 5)):
    target = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if target.shape != predicted.shape or mask.shape != target.shape[:-1]:
        raise ValueError("Forecast array/mask dimensions disagree")
    result = {}
    for i, horizon in enumerate(horizons):
        valid = mask[:, i]
        errors = predicted[:, i][valid] - target[:, i][valid]
        result[str(horizon)] = {"mae": float(np.abs(errors).mean()) if errors.size else None,
                                "mse": float((errors**2).mean()) if errors.size else None,
                                "coverage": float(valid.mean()), "valid_nodes": int(valid.sum())}
    return result
