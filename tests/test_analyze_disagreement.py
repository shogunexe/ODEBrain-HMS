"""Small logic checks for frozen strata and paired patient resampling."""

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_disagreement import (TARGET, PREDICTED, aligned_predictions,
                                          frozen_strata, paired_patient_bootstrap)
from scripts.write_model_cards import CardError


def test_validation_median_is_frozen_for_test():
    val = np.array([[1, 0, 0, 0, 0, 0], [.5, .5, 0, 0, 0, 0]])
    test = np.array([[1, 0, 0, 0, 0, 0], [.5, .5, 0, 0, 0, 0]])
    threshold, strata = frozen_strata(val, test)
    assert 0 < threshold < np.log(2)
    assert strata["low"].tolist() == [True, False]
    assert strata["high"].tolist() == [False, True]


def test_paired_bootstrap_resamples_patients_not_windows():
    a = np.array([0] * 10 + [2], dtype=float)
    b = np.zeros(11)
    patients = np.array(["long"] * 10 + ["short"])
    result = paired_patient_bootstrap(a, b, patients, np.ones(11, bool), 1000, 42)
    assert result["difference_patient_equal_mean"] == 1.0
    assert result["ci95"] == [0.0, 2.0]
    assert result["n_patients"] == 2


def test_paired_bootstrap_applies_subset_mask_once():
    a = np.array([0.0, 10.0, 2.0, 99.0])
    b = np.zeros(4)
    patients = np.array(["p1", "p1", "p2", "p3"])
    result = paired_patient_bootstrap(a, b, patients,
                                      np.array([True, False, True, False]), 1000, 42)
    assert result["difference_patient_equal_mean"] == 1.0
    assert result["n_patients"] == 2


def test_predictions_reject_changed_patient_or_targets(tmp_path):
    expected = pd.DataFrame({"window_id": ["a", "b"], "patient_id": ["p1", "p2"],
                             "class_id": [0, 1],
                             **{name: [int(i == 0), int(i == 1)]
                                for i, name in enumerate(TARGET)}}).set_index("window_id")
    frame = expected.reset_index().copy()
    for i, name in enumerate(PREDICTED):
        frame[name] = [int(i == 0), int(i == 1)]
    path = tmp_path / "predictions.csv"
    frame.to_csv(path, index=False)
    assert aligned_predictions(path, expected).shape == (2, 6)
    frame.loc[0, "patient_id"] = "wrong"
    frame.to_csv(path, index=False)
    with pytest.raises(CardError, match="patient mismatch"):
        aligned_predictions(path, expected)
    frame.loc[0, "patient_id"] = "p1"
    frame.loc[0, TARGET[0]] = 0.5
    frame.to_csv(path, index=False)
    with pytest.raises(CardError, match="expert targets"):
        aligned_predictions(path, expected)
