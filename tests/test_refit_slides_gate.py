"""Final-slide headline must remain bound to verified train+val refit files."""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
from pptx import Presentation

from scripts import build_slides, prepare_formal_slides as slides
from scripts import refit_final


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_prepare_rechecks_refit_inputs_and_rejects_tampered_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(slides, "ROOT", tmp_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("window_id\nw1\n", encoding="utf-8")
    summary_path = tmp_path / "results/refit/final_comparison.json"
    rows = []
    for name in slides.DEEP_MODELS:
        run = tmp_path / "artifacts/refit" / name
        result = summary_path.parent / name
        checkpoint = run / "final.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes((name + " checkpoint").encode())
        _write(run / "config.json", {"refit": {"training_split": "train+val",
                                                "test_used_for_selection": False}})
        _write(run / "completed.json", {})
        _write(run / "validation_record.json", {})
        metric = _write(result / "test_metrics.json", {"model": name})
        predictions = result / "test_predictions.csv"
        predictions.write_text("window_id\nw1\n", encoding="utf-8")
        rows.append({"name": name, "run_dir": str(run), "kl_window_mean": 1.0,
                     "checkpoint_sha256": _hash(checkpoint),
                     "test_metrics_sha256": _hash(metric),
                     "test_predictions_sha256": _hash(predictions)})
    summary = {"schema_version": "hms_final_refit_comparison_v1",
               "experiment_status": "formal", "training_protocol": "train_val_fixed_epoch_refit",
               "split": "test", "split_hash": "a" * 64,
               "manifest_sha256": _hash(manifest), "n_windows": 10, "n_patients": 2,
               "smoke_subset": False, "synthetic": False,
               "test_used_for_selection": False, "rows": rows}
    _write(summary_path, summary)

    def regenerate(runs, output, results_root):
        assert set(runs) == set(slides.DEEP_MODELS)
        assert Path(results_root) == summary_path.parent
        _write(output, summary)

    monkeypatch.setattr(refit_final, "summarize_refits", regenerate)
    bundle = {"refit_comparison": str(summary_path)}
    result = slides.validate_deep(bundle, manifest, "a" * 64,
                                  {"test": (10, 2)}, {})
    assert result["odebrain_inspired_hms"][2] == rows[-1]["checkpoint_sha256"]
    predictions = summary_path.parent / "tcn/test_predictions.csv"
    predictions.write_text("window_id\nchanged\n", encoding="utf-8")
    with pytest.raises(slides.EvidenceError, match="source file hash differs"):
        slides.validate_deep(bundle, manifest, "a" * 64, {"test": (10, 2)}, {})
    with pytest.raises(slides.EvidenceError, match="not initial-run"):
        slides.validate_deep({**bundle, "deep_metrics": {}}, manifest, "a" * 64,
                             {"test": (10, 2)}, {})


def test_builder_rejects_initial_or_unbound_headline(tmp_path, monkeypatch):
    monkeypatch.setattr(build_slides, "ROOT", tmp_path)
    monkeypatch.setattr(build_slides, "EVIDENCE", tmp_path / "slides_evidence.json")
    monkeypatch.setattr(build_slides, "EXPERIMENTAL", {9})
    summary_path = _write(tmp_path / "final_comparison.json", {
        "schema_version": "hms_final_refit_comparison_v1", "experiment_status": "formal",
        "training_protocol": "train_val_fixed_epoch_refit", "split": "test",
        "split_hash": "a" * 64, "test_used_for_selection": False,
        "rows": [{"name": name} for name in slides.DEEP_MODELS]})
    image = tmp_path / "panel.png"
    image.write_bytes(b"figure")
    row = {"image": image.name, "image_sha256": _hash(image),
           "source_file": summary_path.name, "source_sha256": _hash(summary_path),
           "split_hash": "a" * 64, "n_patients": 2, "n_windows": 10,
           "model_version": "refit-final", "experiment_status": "formal",
           "smoke_subset": False,
           "headline_training_protocol": "train_val_fixed_epoch_refit"}
    evidence = {"experiment_status": "formal", "split_hash": "a" * 64,
                "dependencies_sha256": {summary_path.name: _hash(summary_path)},
                "headline_training_protocol": "train_val_fixed_epoch_refit",
                "headline_refit_comparison": summary_path.name,
                "headline_refit_comparison_sha256": _hash(summary_path),
                "slides": {"9": row}}
    _write(build_slides.EVIDENCE, evidence)
    assert 9 in build_slides.load_evidence(True)
    evidence["slides"]["9"]["source_file"] = "initial_best.json"
    _write(build_slides.EVIDENCE, evidence)
    with pytest.raises(SystemExit, match="headline provenance mismatch"):
        build_slides.load_evidence(True)


def test_publication_pending_preview_is_watermarked_and_cannot_be_final(tmp_path, monkeypatch):
    monkeypatch.setattr(build_slides, "ROOT", tmp_path)
    monkeypatch.setattr(build_slides, "SLIDES", tmp_path / "slides")
    monkeypatch.setattr(build_slides, "EXPERIMENTAL", {23})
    monkeypatch.setattr(build_slides, "PREVIEW_EVIDENCE", tmp_path / "preview.json")
    monkeypatch.setattr(build_slides, "EVIDENCE", tmp_path / "preview.json")
    image = tmp_path / "publication_pending.png"
    Image.new("RGB", (200, 120), "white").save(str(image))
    source = _write(tmp_path / "inputs.json", {"resources": "unpublished"})
    row = {"image": image.name, "image_sha256": _hash(image),
           "source_file": source.name, "source_sha256": _hash(source),
           "split_hash": "a" * 64, "n_patients": 2, "n_windows": 10,
           "model_version": "refit-final", "experiment_status": "local_preview",
           "publication_status": "pending_review", "smoke_subset": False}
    _write(build_slides.PREVIEW_EVIDENCE, {
        "experiment_status": "local_preview", "formal": False,
        "publication_status": "pending_review", "split_hash": "a" * 64,
        "dependencies_sha256": {source.name: _hash(source)}, "slides": {"23": row}})
    deck = Presentation(str(build_slides.build(preview=True)))
    assert len(deck.slides) == 24
    for slide in deck.slides:
        content = " ".join(shape.text for shape in slide.shapes if shape.has_text_frame)
        assert "LOCAL PREVIEW · 待发布审核 · 非正式交付" in content
    assert not any("项目代码仓库" in shape.text for shape in deck.slides[22].shapes
                   if shape.has_text_frame)
    with pytest.raises(SystemExit, match="状态"):
        build_slides.build(final=True)
