"""Formal dynamics slides use verified source plots and both outcome types."""

from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_formal_slides import (
    DYNAMICS_FIGURES,
    DYNAMICS_LAYOUT,
    EvidenceError,
    forecast_slide_scope,
    render_dynamics_pair,
    select_dynamics_cases,
    validate_dynamics_figures,
)


def _case(tmp_path, case_id, true_class, predicted_class, color):
    folder = tmp_path / case_id
    folder.mkdir()
    case_path = folder / "case.json"
    case_path.write_text("{}", encoding="utf-8")
    figures = {}
    for name in DYNAMICS_FIGURES:
        Image.new("RGB", (320, 220), color).save(folder / (name + ".png"))
        figures[name] = {"png": name + ".png"}
    case = {"true_class": true_class, "predicted_class": predicted_class,
            "selection_tags": ["typical_correct" if true_class == predicted_class
                               else "typical_error"], "figures": figures}
    paths = validate_dynamics_figures(case_path, case, {})
    return {"id": case_id, "metadata": case_path, "case": case, "figures": paths}


def test_all_six_original_pngs_required(tmp_path):
    case = _case(tmp_path, "case_a", 0, 0, "red")
    Path(case["figures"]["perturbation"]).unlink()
    with pytest.raises(EvidenceError, match="Missing source file"):
        validate_dynamics_figures(case["metadata"], case["case"], {})


def test_both_actual_prediction_outcomes_required(tmp_path):
    correct = _case(tmp_path, "correct_a", 1, 1, "red")
    with pytest.raises(EvidenceError, match="incorrectly predicted"):
        select_dynamics_cases([correct])
    error = _case(tmp_path, "error_b", 2, 3, "blue")
    selected = select_dynamics_cases([error, correct])
    assert selected["correct"]["id"] == "correct_a"
    assert selected["error"]["id"] == "error_b"
    assert {figure for pair in DYNAMICS_LAYOUT.values() for _, figure in pair} == set(DYNAMICS_FIGURES)


def test_composed_slide_contains_both_verified_source_images(tmp_path):
    correct = _case(tmp_path, "correct_a", 1, 1, "red")
    error = _case(tmp_path, "error_b", 2, 3, "blue")
    output = tmp_path / "slide_17.png"
    render_dynamics_pair(17, ("correct", "synchronized", correct),
                         ("error", "trajectory_direction", error), output)
    with Image.open(output) as image:
        assert image.size == (2000, 1040)
        rgb = image.convert("RGB")
        assert rgb.getpixel((500, 550)) == (255, 0, 0)
        assert rgb.getpixel((1500, 550)) == (0, 0, 255)


def test_forecast_slide_records_eligible_and_original_scope():
    comparison = {"n_windows": 22606, "n_patients": 1297,
                  "forecast_eligibility": {"eligible_test_windows": 22606,
                                           "excluded_test_windows": 32}}
    scope = forecast_slide_scope(comparison, test_n=22638, test_patients=1300)
    assert scope == {"n_windows": 22606, "n_patients": 1297,
                     "evaluation_scope": "eligible_test_forecast_windows",
                     "original_test_n_windows": 22638,
                     "original_test_n_patients": 1300, "excluded_count": 32}
    with pytest.raises(EvidenceError, match="differ from original test count"):
        forecast_slide_scope(comparison, test_n=22639, test_patients=1300)
