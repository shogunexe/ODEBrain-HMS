"""Spark prediction postprocessing keeps EEG recordings equally weighted."""

import numpy as np
import pandas as pd
import pytest

from scripts.supplement_spark_record_metrics import calculate
from src.data.hms import CLASSES


def test_record_equal_kl_uses_frozen_predictions_and_manifest_identity():
    rows = []
    for i, (record, probability) in enumerate((("e1", 1.0), ("e1", 1.0),
                                                ("e2", 0.1))):
        row = {"window_id": "w%d" % i, "eeg_id": record, "patient_id": "p1",
               "split": "test", "probabilities": np.array([probability, 1-probability, 0, 0, 0, 0])}
        row.update({"target_" + name: float(j == 0) for j, name in enumerate(CLASSES)})
        rows.append(row)
    predictions = pd.DataFrame(rows)
    manifest = predictions.drop(columns=["probabilities"]).copy()
    reported = {"n": 3, "kl_window_mean": -np.log(0.1) / 3,
                "kl_patient_equal_mean": -np.log(0.1) / 3}
    result = calculate(predictions, manifest, reported)
    assert result["n_windows"] == 3
    assert result["n_eeg_records"] == 2
    assert result["kl_window_mean"] == pytest.approx(-np.log(0.1) / 3)
    assert result["kl_eeg_record_equal_mean"] == pytest.approx(-np.log(0.1) / 2)
    bad = predictions.copy()
    bad.loc[0, "eeg_id"] = "wrong"
    with pytest.raises(ValueError, match="differs from audited manifest"):
        calculate(bad, manifest, reported)
