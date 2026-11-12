import numpy as np
import pandas as pd
import pytest

from scripts.supplement_classification_metrics import (checked_arrays, group_weights,
                                                       validate_kl, weighted_view)
from src.data.hms import CLASSES


def test_group_equal_weights_change_window_metrics_without_group_vote():
    hard = np.array([0, 0, 0, 1])
    scores = np.array([[.8, .2, 0, 0, 0, 0],
                       [.8, .2, 0, 0, 0, 0],
                       [.8, .2, 0, 0, 0, 0],
                       [.8, .2, 0, 0, 0, 0]])
    target = np.eye(6)[hard]
    patients = ["a", "a", "a", "b"]
    window = weighted_view(target, scores, hard, np.ones(4))
    patient = weighted_view(target, scores, hard, group_weights(patients))
    assert window["total_window_weight"] == pytest.approx(4)
    assert patient["total_window_weight"] == pytest.approx(2)
    assert patient["confusion_matrix"][0][0] == pytest.approx(1)
    assert patient["confusion_matrix"][1][0] == pytest.approx(1)
    assert patient["per_class"]["seizure"]["false_positive_rate"] == pytest.approx(1)
    assert patient["per_class"]["seizure"]["ovr_auroc"] == pytest.approx(.5)
    assert patient["kl_mean"] != pytest.approx(window["kl_mean"])


def test_prediction_manifest_identity_and_source_kl_validation():
    names = ["target_" + name for name in CLASSES]
    scores = np.eye(6)[[0, 1]] * .9 + .1 / 6
    manifest = pd.DataFrame({"window_id": ["a", "b"], "patient_id": [1, 2],
                             "eeg_id": [11, 22], "class_id": [0, 1],
                             **{name: np.eye(6)[[0, 1], i] for i, name in enumerate(names)}})
    pred = manifest.drop(columns="eeg_id").copy()
    for i, name in enumerate(CLASSES):
        pred["pred_" + name] = scores[:, i]
    target, actual, hard, patient, eeg = checked_arrays(pred, manifest, "deep")
    assert list(eeg) == ["11", "22"]
    assert list(patient) == ["1", "2"]
    assert np.array_equal(hard, [0, 1])
    assert actual.shape == target.shape == (2, 6)
    bad = pred.copy()
    bad.loc[0, "patient_id"] = 3
    with pytest.raises(ValueError, match="patient_id"):
        checked_arrays(bad, manifest, "deep")
    views = {key: {"kl_mean": 0.25} for key in ("window_equal", "patient_equal", "eeg_record_equal")}
    validate_kl(views, {"kl_window_mean": .25, "kl_patient_equal_mean": .25,
                        "kl_eeg_record_equal_mean": .25})
    with pytest.raises(ValueError, match="formal comparison"):
        validate_kl(views, {"kl_window_mean": .25, "kl_patient_equal_mean": .25,
                            "kl_eeg_record_equal_mean": .26})
