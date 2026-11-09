"""Numerical summaries for real ODEBrainInspiredHMS outputs.

The projection uses mean valid-node latent state; full-state speed retains all
valid nodes. All time derivatives are per physical second (times_s).
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def mean_valid_nodes(values, node_mask):
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(node_mask, dtype=bool)
    if values.ndim != 3 or mask.shape != (values.shape[1],):
        raise ValueError("Expected [T,N,D] values and [N] node mask")
    if not mask.any():
        raise ValueError("No valid nodes for projection")
    return values[:, mask].mean(axis=1)


class TrainOnlyPCA:
    """A shared orthogonal projection for one model and one patient split."""

    def __init__(self, mean, components, model_id, split_hash, train_window_ids):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.components = np.asarray(components, dtype=np.float64)
        self.model_id = str(model_id)
        self.split_hash = str(split_hash)
        self.train_window_ids = tuple(map(str, train_window_ids))
        if self.components.ndim != 2 or self.components.shape[1] != len(self.mean):
            raise ValueError("PCA dimensions do not match")

    @classmethod
    def fit(cls, trajectories, model_id, split_hash, train_window_ids, n_components=3):
        if not trajectories or not train_window_ids or len(trajectories) != len(train_window_ids):
            raise ValueError("PCA requires matched training trajectories and IDs")
        arrays = [np.asarray(x, dtype=np.float64) for x in trajectories]
        if any(a.ndim != 2 or a.shape[1] != arrays[0].shape[1] for a in arrays):
            raise ValueError("PCA inputs must be [T,D] with common D")
        x = np.concatenate(arrays, axis=0)
        if len(x) < 2 or not np.isfinite(x).all():
            raise ValueError("PCA needs at least two finite training states")
        mean = x.mean(axis=0)
        _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
        k = min(int(n_components), len(vt))
        if k < 2:
            raise ValueError("PCA needs at least two components")
        return cls(mean, vt[:k], model_id, split_hash, train_window_ids)

    def transform(self, states):
        states = np.asarray(states, dtype=np.float64)
        return (states - self.mean) @ self.components.T

    def transform_derivative(self, derivatives):
        """du/dt = W dz/dt, with no centering of derivative vectors."""
        return np.asarray(derivatives, dtype=np.float64) @ self.components.T

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), mean=self.mean, components=self.components,
                            model_id=self.model_id, split_hash=self.split_hash,
                            train_window_ids=np.asarray(self.train_window_ids))

    @classmethod
    def load(cls, path):
        with np.load(str(path), allow_pickle=False) as z:
            return cls(z["mean"], z["components"], str(z["model_id"]),
                       str(z["split_hash"]), z["train_window_ids"].tolist())


def entropy(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    p = p / max(float(p.sum()), 1e-15)
    return float(-np.sum(p[p > 0] * np.log(p[p > 0])))


def trajectory_summary(times_s, trajectory, vector_field, node_mask, projector):
    t = np.asarray(times_s, dtype=np.float64)
    z = np.asarray(trajectory, dtype=np.float64)
    f = np.asarray(vector_field, dtype=np.float64)
    mask = np.asarray(node_mask, dtype=bool)
    if t.ndim != 1 or z.shape != f.shape or z.ndim != 3 or z.shape[0] != len(t):
        raise ValueError("Trajectory, field, or times shape mismatch")
    if len(t) < 2 or t[0] != 0 or np.any(np.diff(t) <= 0):
        raise ValueError("times_s must begin at zero and increase")
    if not np.isfinite(z).all() or not np.isfinite(f).all():
        raise ValueError("Nonfinite latent state or vector field")
    reduced = mean_valid_nodes(z, mask)
    derivative = mean_valid_nodes(f, mask)
    projected = projector.transform(reduced)
    projected_field = projector.transform_derivative(derivative)
    valid_field = f[:, mask].reshape(len(t), -1)
    speed = np.linalg.norm(valid_field, axis=1)
    increments = np.linalg.norm(np.diff(z[:, mask], axis=0).reshape(len(t)-1, -1), axis=1)
    segment_length = 0.5 * (speed[1:] + speed[:-1]) * np.diff(t)
    dot = np.sum(valid_field[1:] * valid_field[:-1], axis=1)
    norm_product = speed[1:] * speed[:-1]
    turns = np.full(len(t)-1, np.nan)
    stable = norm_product > 1e-12
    turns[stable] = np.arccos(np.clip(dot[stable] / norm_product[stable], -1, 1))
    return {
        "projected": projected, "projected_field": projected_field,
        "speed_per_s": speed, "path_length_integral": float(segment_length.sum()),
        "sampled_chord_length": float(increments.sum()),
        "turn_angle_rad": turns,
        "mean_turn_angle_rad": float(np.nanmean(turns)) if stable.any() else None,
        "near_zero_turn_pairs": int((~stable).sum()),
    }


def forecast_errors(prediction, target, mask, horizons_s):
    p, y = np.asarray(prediction), np.asarray(target)
    m, h = np.asarray(mask, dtype=bool), np.asarray(horizons_s)
    if p.shape != y.shape or p.ndim != 3 or m.shape != p.shape[:2] or len(h) != len(p):
        raise ValueError("Forecast arrays must be [H,N,F], [H,N], [H]")
    result = []
    for i, sec in enumerate(h):
        error = (p[i] - y[i])[m[i]]
        result.append({"horizon_s": float(sec), "valid_nodes": int(m[i].sum()),
                       "mse": float(np.mean(error**2)) if error.size else None,
                       "mae": float(np.mean(np.abs(error))) if error.size else None})
    return result


def choose_cases(records, max_per_group=1):
    """Deterministic, prespecified rule applied separately to val and test.

    Valid records: window_id, expert_entropy, correct, split. A window may
    appear in multiple categories, and is exported once with all tags.
    """
    chosen = {}
    for split in ("val", "test"):
        pool = [r for r in records if r["split"] == split]
        if not pool:
            continue
        plans = (
            ("low_disagreement", sorted(pool, key=lambda r: (r["expert_entropy"], r["window_id"]))),
            ("high_disagreement", sorted(pool, key=lambda r: (-r["expert_entropy"], r["window_id"]))),
            ("typical_correct", sorted((r for r in pool if r["correct"]), key=lambda r: (r["expert_entropy"], r["window_id"]))),
            ("typical_error", sorted((r for r in pool if not r["correct"]), key=lambda r: (r["expert_entropy"], r["window_id"]))),
        )
        for tag, candidates in plans:
            for item in candidates[:max_per_group]:
                key = (split, item["window_id"])
                chosen.setdefault(key, {"record": item, "tags": []})["tags"].append(tag)
    return list(chosen.values())


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
