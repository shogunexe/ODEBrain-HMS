"""Deterministic patient-disjoint train/validation/test and development folds."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .hms import manifest_hash


def assign_patient_splits(manifest, seed=42, test_fraction=0.20, val_fraction_of_dev=0.20, folds=3):
    table = manifest.copy()
    valid = table.audit_reason == "ok"
    # Freeze identities from all metadata, regardless of which raw files have
    # arrived. A small download must not change the eventual test patients.
    patients = np.array(sorted(table.patient_id.astype(str).unique()))
    if len(patients) < 5:
        raise ValueError("Need at least five patients for patient-disjoint split")
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)
    n_test = max(1, int(round(len(patients) * test_fraction)))
    test_ids = set(patients[:n_test])
    dev = patients[n_test:].copy()
    n_val = max(1, int(round(len(dev) * val_fraction_of_dev)))
    val_ids = set(dev[:n_val])
    train_ids = set(dev[n_val:])
    if not train_ids:
        raise ValueError("Empty training group")
    table["split"] = "excluded"
    ps = table.patient_id.astype(str)
    table.loc[valid & ps.isin(train_ids), "split"] = "train"
    table.loc[valid & ps.isin(val_ids), "split"] = "val"
    table.loc[valid & ps.isin(test_ids), "split"] = "test"
    table["fold"] = -1
    rng2 = np.random.RandomState(seed + 1)
    rng2.shuffle(dev)
    fold_map = {pid: i % folds for i, pid in enumerate(dev)}
    table.loc[valid & ~ps.isin(test_ids), "fold"] = ps[valid & ~ps.isin(test_ids)].map(fold_map).astype(int)
    groups = {name: set(table.loc[table.split == name, "patient_id"].astype(str)) for name in ("train", "val", "test")}
    assert not (groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"])
    return table


def save_split(manifest, path, seed=42):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    summary = {"seed": seed, "sha256": manifest_hash(manifest), "split_patient_counts": {}, "split_window_counts": {}}
    for name in ("train", "val", "test", "excluded"):
        section = manifest[manifest.split == name]
        summary["split_patient_counts"][name] = int(section.patient_id.nunique())
        summary["split_window_counts"][name] = int(len(section))
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
