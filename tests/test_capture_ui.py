"""The capture tool must reject blank screens before writing slide evidence."""
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from scripts.capture_ui import CaptureError, _formal_readiness, _smoke_watermark, _validate_png


def _png(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_blank_browser_capture_is_rejected():
    with pytest.raises(CaptureError, match="blank/loading skeleton"):
        _validate_png(_png(Image.new("RGB", (1600, 1000), "white")))


def test_rendered_content_passes_png_gate():
    rng = np.random.RandomState(9)
    pixels = rng.randint(0, 30, (700, 1000, 3), dtype=np.uint8)
    pixels[:, 500:] += 180
    image = Image.fromarray(pixels)
    info = _validate_png(_png(image))
    assert info["width"] == 1000 and info["height"] == 700
    assert info["bytes"] > 20000


def test_smoke_watermark_is_visible_and_keeps_png_valid():
    pixels = np.full((700, 1000, 3), 230, dtype=np.uint8)
    pixels[200:500, 150:850] = (10, 80, 160)
    original = _png(Image.fromarray(pixels))
    marked = _smoke_watermark(original)
    assert marked != original
    assert Image.open(BytesIO(marked)).size == (1000, 700)


def test_missing_formal_input_bundle_stays_smoke(tmp_path):
    split_hash, reasons = _formal_readiness(tmp_path / "missing_inputs.json")
    assert split_hash is None and reasons
