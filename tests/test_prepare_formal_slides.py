"""Fail-closed checks for formal slide packaging."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

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
