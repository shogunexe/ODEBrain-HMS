"""Check final ODEBrain result provenance and the revised report."""

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from pptx import Presentation


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/odebrain_github_v2"
PPTX = ROOT / "FINAL_REVIEW/01_delivery/项目汇报修改版.pptx"
BACKUP = ROOT / "FINAL_REVIEW/01_delivery/项目汇报修改版_ODE更新前.pptx"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slide_text(slide):
    return "\n".join(shape.text for shape in slide.shapes if shape.has_text_frame)


def main():
    completed = load(RESULTS / "pipeline_completed.json")
    assert completed["status"] == "complete" and completed["pptx_sha256"] == sha(PPTX)
    comparison = load(RESULTS / "final_comparison.json")
    forecast = load(RESULTS / "forecast_comparison.json")
    weighted = load(RESULTS / "classification_weighted_metrics.json")
    field = load(RESULTS / "six_class_field/six_class_ode_field.json")
    cases = load(RESULTS / "cases/case_pair.json")
    for result in (comparison, forecast, weighted):
        assert result["experiment_status"] == "formal"
    assert len(comparison["rows"]) == 5 and len(field["cases"]) == 6 and len(cases["cases"]) == 2
    assert comparison["split_hash"] == forecast["split_hash"] == weighted["split_hash"]
    ode = next(row for row in comparison["rows"] if row["name"] == "odebrain_inspired_hms")
    checkpoint = ROOT / "artifacts/refit/odebrain_github_v2_hms/final.pt"
    assert ode["checkpoint_sha256"] == field["checkpoint_sha256"] == cases["checkpoint_sha256"] == sha(checkpoint)
    assert weighted["source_sha256"]["deep_comparison"] == sha(RESULTS / "final_comparison.json")
    assert sha(RESULTS / "cases/case_pair.json") and sha(RESULTS / "six_class_field/six_class_ode_field.png")

    with ZipFile(PPTX) as after, ZipFile(BACKUP) as before:
        for page in range(1, 6):
            key = "ppt/slides/slide%d.xml" % page
            assert after.read(key) == before.read(key), "Original slide %d changed" % page
    prs = Presentation(str(PPTX))
    assert len(prs.slides) == 27
    assert "12 个" in slide_text(prs.slides[11])
    assert "统一使用2×V100S联合训练" in slide_text(prs.slides[12])
    assert "完整指标表" in slide_text(prs.slides[13])
    assert any(shape.has_table and len(shape.table.rows) == 8 and len(shape.table.columns) == 10
               for shape in prs.slides[13].shapes)
    assert "六类 EEG" in slide_text(prs.slides[22])
    later_text = "\n".join(slide_text(slide) for slide in list(prs.slides)[9:])
    for stale in ("1D CNN", "raw 32", "t=1s 分类六类"):
        assert stale not in later_text, "Stale model description: %s" % stale
    output = {"status": "passed", "slides": len(prs.slides),
              "first_five_slide_xml_unchanged": True,
              "classification_models": len(comparison["rows"]),
              "field_classes": len(field["cases"]),
              "checkpoint_sha256": sha(checkpoint), "pptx_sha256": sha(PPTX)}
    path = RESULTS / "delivery_audit.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
