import hashlib
import json

import pytest
from PIL import Image
from pptx import Presentation

from scripts import build_slides


def test_final_slides_reject_smoke_source(tmp_path, monkeypatch):
    monkeypatch.setattr(build_slides, "ROOT", tmp_path)
    monkeypatch.setattr(build_slides, "EVIDENCE", tmp_path / "slides_evidence.json")
    monkeypatch.setattr(build_slides, "EXPERIMENTAL", {23})
    image = tmp_path / "figure.png"
    source = tmp_path / "metrics.json"
    image.write_bytes(b"static figure")
    source.write_text(json.dumps({"smoke_subset": False}), encoding="utf-8")
    row = {"image": "figure.png", "source_file": "metrics.json",
           "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
           "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
           "split_hash": "frozen-split", "n_patients": 2, "n_windows": 10,
           "model_version": "trained-v1", "experiment_status": "formal", "smoke_subset": False}
    evidence = {"experiment_status": "formal", "split_hash": "frozen-split",
                "dependencies_sha256": {source.name: row["source_sha256"]},
                "slides": {str(i): row for i in build_slides.EXPERIMENTAL}}
    build_slides.EVIDENCE.write_text(json.dumps(evidence), encoding="utf-8")
    assert len(build_slides.load_evidence(True)) == len(build_slides.EXPERIMENTAL)
    source.write_text(json.dumps({"smoke_subset": True}), encoding="utf-8")
    row["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    evidence["dependencies_sha256"][source.name] = row["source_sha256"]
    build_slides.EVIDENCE.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(SystemExit, match="smoke evidence"):
        build_slides.load_evidence(True)


def test_formal_resource_links_are_clickable_and_require_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(build_slides, "ROOT", tmp_path)
    monkeypatch.setattr(build_slides, "SLIDES", tmp_path / "slides")
    monkeypatch.setattr(build_slides, "EVIDENCE", tmp_path / "slides_evidence.json")
    monkeypatch.setattr(build_slides, "EXPERIMENTAL", {23})
    image = tmp_path / "resources.png"
    Image.new("RGB", (16, 16), "white").save(str(image))
    source = tmp_path / "formal_resources.json"
    resources = {"experiment_status": "formal",
                 "repository": {"url": "https://example.org/project", "verified_http_status": 200,
                                "verified_at": "2026-11-08T00:00:00Z"},
                 "weights": {"url": "https://example.org/weights", "verified_http_status": 200,
                             "verified_at": "2026-11-08T00:00:00Z"}}
    source.write_text(json.dumps(resources), encoding="utf-8")
    row = {"image": image.name, "source_file": source.name,
           "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
           "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
           "split_hash": "frozen-split", "n_patients": 2, "n_windows": 10,
           "model_version": "trained-v1", "experiment_status": "formal", "smoke_subset": False}
    evidence = {"experiment_status": "formal", "split_hash": "frozen-split",
                "dependencies_sha256": {source.name: row["source_sha256"]},
                "slides": {"23": row}}
    build_slides.EVIDENCE.write_text(json.dumps(evidence), encoding="utf-8")

    deck = Presentation(str(build_slides.build(final=True)))
    for page in (23, 24):
        links = {run.hyperlink.address for shape in deck.slides[page - 1].shapes
                 if shape.has_text_frame for paragraph in shape.text_frame.paragraphs
                 for run in paragraph.runs if run.hyperlink.address}
        assert resources["repository"]["url"] in links
        assert resources["weights"]["url"] in links
        assert build_slides.SOURCES["hms"] in links

    draft = Presentation(str(build_slides.build(final=False)))
    draft_links = {run.hyperlink.address for page in (23, 24)
                   for shape in draft.slides[page - 1].shapes if shape.has_text_frame
                   for paragraph in shape.text_frame.paragraphs
                   for run in paragraph.runs if run.hyperlink.address}
    assert resources["repository"]["url"] not in draft_links
    assert resources["weights"]["url"] not in draft_links
    assert build_slides.SOURCES["hms"] in draft_links

    resources["weights"]["verified_at"] = ""
    source.write_text(json.dumps(resources), encoding="utf-8")
    row["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    evidence["dependencies_sha256"][source.name] = row["source_sha256"]
    build_slides.EVIDENCE.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(SystemExit, match="weights"):
        build_slides.build(final=True)
