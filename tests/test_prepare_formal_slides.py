"""Fail-closed checks for formal slide packaging."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from PIL import Image

from scripts import prepare_formal_slides as slides
from scripts.prepare_formal_slides import EvidenceError, reject_smoke


@pytest.mark.parametrize("fixture", [
    {"smoke_subset": True},
    {"runs": [{"experiment_status": "real_data_smoke"}]},
    {"cases": [{"provenance": "real_hms_smoke"}]},
    {"synthetic": True},
])
def test_nested_smoke_rejected(fixture):
    with pytest.raises(EvidenceError, match="Smoke/synthetic"):
        reject_smoke(fixture, "synthetic fixture")


def test_cli_rejects_smoke_bundle_without_writing_manifest(tmp_path):
    bundle = tmp_path / "inputs.json"
    bundle.write_text(json.dumps({"smoke_subset": True}), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_formal_slides.py"
    result = subprocess.run([sys.executable, str(script), "--inputs", str(bundle)],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 2
    assert "Smoke/synthetic evidence" in result.stderr


def test_ui_capture_requires_unchanged_rendered_screenshot(tmp_path, monkeypatch):
    monkeypatch.setattr(slides, "ROOT", tmp_path)
    screenshot = tmp_path / "ui.png"
    Image.new("RGB", (900, 600), "white").save(screenshot)
    import hashlib
    checksum = hashlib.sha256(screenshot.read_bytes()).hexdigest()
    metadata = tmp_path / "capture.json"
    metadata.write_text(json.dumps({
        "experiment_status": "formal", "split_hash": "a" * 64,
        "page": "ODE 动力学", "rendered_charts": 4,
        "formal_gate": {"passed": True}, "smoke_watermark_added": False,
        "image": str(screenshot), "image_sha256": checksum,
    }), encoding="utf-8")
    bundle = {"ui_capture": str(metadata)}
    assert slides.validate_ui(bundle, "a" * 64, {})[2] == screenshot
    Image.new("RGB", (900, 600), "black").save(screenshot)
    with pytest.raises(EvidenceError, match="differs from capture metadata"):
        slides.validate_ui(bundle, "a" * 64, {})


def test_kl_chart_sources_keep_fixed_models_and_three_distinct_weightings(tmp_path):
    def metrics(value):
        return {"kl_window_mean": value, "kl_patient_equal_mean": value + .1,
                "kl_eeg_record_equal_mean": value + .2}

    deep = {name: (None, metrics(index + 1))
            for index, name in enumerate(slides.DEEP_MODELS)}
    spark = {"models": {"lr": metrics(6), "rf": metrics(7)}}
    labels, values = slides.classification_kl_chart_data(deep, spark)
    assert labels == ("TCN", "CNN", "Fusion", "GraphGRU", "ODE", "Spark LR", "Spark RF")
    assert values == [(1, 1.1, 1.2), (2, 2.1, 2.2), (3, 3.1, 3.2),
                      (4, 4.1, 4.2), (5, 5.1, 5.2), (6, 6.1, 6.2), (7, 7.1, 7.2)]

    def arm(name, value):
        return {"name": name, "aggregations": {
            "window_equal": {"kl_mean": value},
            "patient_equal": {"kl_mean": value + .1},
            "eeg_record_equal": {"kl_mean": value + .2}}}

    ablation = {"rows": [arm("pretrained_rk4_025", 3),
                         arm("scratch_rk4_025", 1),
                         arm("pretrained_rk4_050", 2)]}
    arm_labels, arm_values = slides.ablation_kl_chart_data(ablation)
    assert arm_labels == ["Pretrain · RK4 .25", "Scratch · RK4 .25", "Pretrain · RK4 .50"]
    assert arm_values == [(3, 3.1, 3.2), (1, 1.1, 1.2), (2, 2.1, 2.2)]
    output = tmp_path / "kl_chart.png"
    slides.render_kl_chart("Measured KL", arm_labels, arm_values, output, "Frozen setup")
    with Image.open(output) as chart:
        assert chart.format == "PNG" and chart.width >= 1000 and chart.height >= 500
