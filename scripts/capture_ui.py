#!/usr/bin/env python3
"""Capture a rendered Streamlit page through Chrome DevTools Protocol.

Chrome 78 on the project host lacks ChromeDriver. This command waits for
Streamlit headings and rendered Plotly SVG traces, then uses Page.captureScreenshot.
It does not save a loading skeleton or promote partial HMS results to formal.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import ProxyHandler, build_opener

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAGES = ("数据概览", "样本分析", "模型对比", "ODE 动力学", "大数据性能")
REQUIRED_CHARTS = {"数据概览": 3, "样本分析": 3, "模型对比": 2,
                   "ODE 动力学": 3, "大数据性能": 1}
HEADINGS = {"数据概览": "1 · 数据概览", "样本分析": "2 · 样本分析",
            "模型对比": "3 · 模型对比", "ODE 动力学": "4 · ODE 动力学",
            "大数据性能": "5 · 大数据性能"}
OPEN = build_opener(ProxyHandler({}))  # Ignore the host's proxy for localhost.


class CaptureError(RuntimeError):
    pass


def _path(value):
    value = Path(value).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


def _relative(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_json(url, timeout=3):
    with OPEN.open(url, timeout=timeout) as response:
        return json.load(response)


def _health(url):
    try:
        with OPEN.open(url, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def _cjk_font():
    """A screenshot with tofu squares is not usable Chinese UI evidence."""
    try:
        result = subprocess.run(["fc-list", ":lang=zh", "family"],
                                capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CaptureError("Cannot verify a Chinese font: {}".format(exc)) from exc
    families = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not families:
        raise CaptureError("No Chinese font installed; install Noto Sans CJK SC before capture")
    return families[0]


def _smoke_watermark(blob):
    """Make provenance visible even when the JSON is separated from the PNG."""
    image = Image.open(BytesIO(blob)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font_path = subprocess.check_output(["fc-list", "-f", "%{file}\n", ":lang=zh"],
                                            text=True, timeout=8).splitlines()[0].strip()
        font = ImageFont.truetype(font_path, size=32)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        font = ImageFont.load_default()
    label = "REAL HMS SMOKE · 非正式证据"
    bounds = draw.textbbox((0, 0), label, font=font)
    width = bounds[2] - bounds[0]
    x = max(10, image.width - width - 40)
    draw.rectangle((x - 14, 18, image.width - 14, 66), fill=(255, 241, 238, 240))
    draw.text((x, 21), label, fill=(170, 25, 25, 255), font=font)
    image = Image.alpha_composite(image, overlay).convert("RGB")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _split_information(results_root):
    """Read the actual patient split and independently check its saved hash."""
    from src.data.hms import manifest_hash
    import pandas as pd

    manifest = ROOT / "data/curated/split_manifest.csv"
    summary = ROOT / "data/curated/split_manifest.json"
    if not manifest.is_file():
        return None, ["患者级 split manifest 缺失"]
    try:
        table = pd.read_csv(str(manifest))
        actual_hash = manifest_hash(table)
    except Exception as exc:
        return None, ["患者级 split manifest 不可校验：{}".format(exc)]
    reasons = []
    if not summary.is_file():
        reasons.append("split summary JSON 缺失")
    else:
        try:
            saved = json.loads(summary.read_text(encoding="utf-8"))
            if saved.get("sha256") != actual_hash:
                reasons.append("split summary 哈希与当前 manifest 不一致")
        except (OSError, ValueError) as exc:
            reasons.append("split summary 不可读取：{}".format(exc))
    dynamics = results_root / "dynamics_refit/result_index.json"
    if not dynamics.is_file():
        dynamics = results_root / "dynamics/result_index.json"
    if dynamics.is_file():
        try:
            index = json.loads(dynamics.read_text(encoding="utf-8"))
            if index.get("split_hash") != actual_hash:
                reasons.append("动力学索引 split hash 与当前 manifest 不一致")
            if index.get("provenance") != "real_hms_checkpoint":
                reasons.append("动力学案例仍为 smoke 或未证明正式来源")
        except (OSError, ValueError) as exc:
            reasons.append("动力学索引不可读取：{}".format(exc))
    else:
        reasons.append("动力学结果索引缺失")
    return actual_hash, reasons


def _formal_readiness(inputs_path):
    """Run all existing formal evidence gates except UI/resources (circular)."""
    from scripts import prepare_formal_slides as gate

    if not inputs_path.is_file():
        return None, ["正式交付输入清单缺失：{}".format(_relative(inputs_path))]
    try:
        bundle = gate.read_json(inputs_path)
        if not isinstance(bundle, dict):
            raise gate.EvidenceError("Formal input bundle must be a JSON object")
        gate.reject_smoke(bundle, "formal input bundle")
        dependencies = {}
        manifest, _, _, _, split_hash, counts = gate.validate_manifest(bundle, dependencies)
        gate.validate_features(bundle, manifest, counts, dependencies)
        gate.validate_spark_models(bundle, counts, dependencies)
        gate.validate_spark_record_metrics(bundle, manifest, split_hash, counts, dependencies)
        gate.validate_weighted_classification(bundle, manifest, split_hash, counts, dependencies)
        deep = gate.validate_deep(bundle, manifest, split_hash, counts, dependencies)
        gate.validate_paired_refit(bundle, manifest, split_hash, counts, dependencies)
        gate.validate_benchmark(bundle, counts, dependencies)
        ablation_path, ablation = gate.validate_comparison(
            bundle, "ablation", split_hash, counts, dependencies)
        gate.validate_ablation_weighted(bundle, ablation_path, ablation,
                                        split_hash, counts, dependencies)
        gate.validate_disagreement(bundle, manifest, split_hash, counts, dependencies)
        gate.validate_comparison(bundle, "forecast_comparison", split_hash, counts, dependencies)
        gate.validate_dynamics(bundle, manifest, split_hash, counts,
                               deep["odebrain_inspired_hms"][2], dependencies)
        return split_hash, []
    except (Exception) as exc:
        return None, ["正式结果门禁未通过：{}".format(exc)]


class CDP:
    def __init__(self, websocket_url):
        try:
            import websocket
        except ImportError as exc:
            raise CaptureError("Install websocket-client==1.6.4 in the project venv") from exc
        self.ws = websocket.create_connection(websocket_url, timeout=10,
                                              http_proxy_host=None, suppress_origin=True)
        self.serial = 0

    def close(self):
        self.ws.close()

    def call(self, method, params=None, timeout=15):
        self.serial += 1
        wanted = self.serial
        self.ws.send(json.dumps({"id": wanted, "method": method,
                                 "params": params or {}}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.ws.settimeout(max(.1, deadline - time.monotonic()))
            try:
                message = json.loads(self.ws.recv())
            except Exception as exc:
                raise CaptureError("Chrome DevTools disconnected during {}: {}".format(method, exc)) from exc
            if message.get("id") != wanted:
                continue
            if "error" in message:
                raise CaptureError("Chrome DevTools {} failed: {}".format(method, message["error"]))
            return message.get("result", {})
        raise CaptureError("Chrome DevTools {} timed out".format(method))

    def evaluate(self, expression):
        response = self.call("Runtime.evaluate", {"expression": expression,
                                                   "returnByValue": True,
                                                   "awaitPromise": False})
        if "exceptionDetails" in response:
            raise CaptureError("Browser JavaScript failed: {}".format(response["exceptionDetails"]))
        return response.get("result", {}).get("value")


PROBE = r"""(() => {
  const headings = [...document.querySelectorAll('h1,h2,h3')].map(x => x.innerText.trim());
  const charts = [...document.querySelectorAll('.js-plotly-plot')];
  const drawn = charts.filter(x => {
    const svg = x.querySelector('svg.main-svg');
    if (!svg) return false;
    const box = x.getBoundingClientRect();
    return box.width > 200 && box.height > 100 &&
           svg.querySelectorAll('path,rect,circle,text').length > 10;
  });
  const skeletons = [...document.querySelectorAll('[data-testid="stSkeleton"],.stSkeleton,[data-testid="stSpinner"]')]
    .filter(x => { const b=x.getBoundingClientRect(); return b.width>0 && b.height>0; });
  return {readyState: document.readyState, headings: headings, charts: charts.length,
          drawnCharts: drawn.length, skeletons: skeletons.length,
          mainText: ((document.querySelector('[data-testid="stMain"]') || document.body) || {innerText:''}).innerText.slice(0,5000),
          visibleText: document.body ? document.body.innerText.slice(0,8000) : '',
          bodyWidth: document.body ? document.body.scrollWidth : 0,
          bodyHeight: document.body ? document.body.scrollHeight : 0,
          fontStatus: document.fonts ? document.fonts.status : 'unknown'};
})()"""


CHROME78_POLYFILLS = r"""(() => {
  if (!String.prototype.replaceAll) {
    Object.defineProperty(String.prototype, 'replaceAll', {
      value: function(search, replacement) {
        if (search instanceof RegExp) {
          if (!search.global) throw new TypeError('replaceAll needs a global RegExp');
          return String(this).replace(search, replacement);
        }
        return String(this).split(String(search)).join(String(replacement));
      }, configurable: true, writable: true
    });
  }
})()"""


def _wait_rendered(cdp, page, results_root, min_charts, timeout):
    deadline = time.monotonic() + timeout
    stable = 0
    last_signature = None
    last = None
    while time.monotonic() < deadline:
        state = cdp.evaluate(PROBE)
        last = state
        if isinstance(state, dict):
            text = state.get("mainText", "")
            title_ok = any(HEADINGS[page] in h for h in state.get("headings", []))
            results_ok = str(results_root) in state.get("visibleText", "")
            valid = (state.get("readyState") == "complete" and title_ok and results_ok
                     and state.get("drawnCharts", 0) >= min_charts
                     and state.get("skeletons", 0) == 0
                     and state.get("fontStatus") in ("loaded", "unknown"))
            signature = (state.get("drawnCharts"), state.get("charts"),
                         state.get("bodyWidth"), state.get("bodyHeight"))
            stable = stable + 1 if valid and signature == last_signature else (1 if valid else 0)
            last_signature = signature
            if stable >= 3:
                # A short extra frame ensures Plotly's final layout is painted.
                time.sleep(.7)
                return state
        time.sleep(.5)
    raise CaptureError("Page never reached a rendered, stable state within {} s; last probe: {}".format(
        timeout, json.dumps(last, ensure_ascii=False)[:900]))


def _click_page(cdp, page, timeout):
    expression = """(() => {
      const wanted = %s;
      const sidebar = document.querySelector('[data-testid="stSidebar"]') || document;
      const labels = [...sidebar.querySelectorAll('label,[role="radio"]')];
      const match = labels.find(x => (x.innerText || x.textContent || '').trim() === wanted);
      if (!match) return false;
      match.click();
      return true;
    })()""" % json.dumps(page, ensure_ascii=False)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cdp.evaluate(expression):
            return
        time.sleep(.4)
    sidebar = cdp.evaluate("""(() => {
      const x = document.querySelector('[data-testid="stSidebar"]');
      return x ? x.innerText.slice(0,1200) : 'sidebar missing';
    })()""")
    raise CaptureError("Cannot find Streamlit sidebar page option {}: {}".format(page, sidebar))


def _validate_png(blob, min_width=900, min_height=600):
    if not blob.startswith(b"\x89PNG\r\n\x1a\n"):
        raise CaptureError("Chrome returned non-PNG data")
    try:
        image = Image.open(BytesIO(blob)).convert("RGB")
    except Exception as exc:
        raise CaptureError("Chrome screenshot cannot be decoded: {}".format(exc)) from exc
    width, height = image.size
    if width < min_width or height < min_height:
        raise CaptureError("Screenshot is too small: {}x{}".format(width, height))
    gray = image.resize((160, 90)).convert("L")
    contrast = float(ImageStat.Stat(gray).stddev[0])
    nonwhite = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
    if len(blob) < 20000 or contrast < 7 or nonwhite is None:
        raise CaptureError("Screenshot resembles a blank/loading skeleton ({} bytes, contrast {:.1f})".format(
            len(blob), contrast))
    return {"width": width, "height": height, "bytes": len(blob),
            "grayscale_stddev": round(contrast, 2)}


def _chrome_target(profile, chrome, timeout):
    log_path = profile / "chrome.log"
    log = log_path.open("wb")
    command = [str(chrome), "--headless", "--disable-gpu", "--no-sandbox",
               "--disable-dev-shm-usage", "--hide-scrollbars", "--no-proxy-server",
               "--remote-debugging-port=0", "--window-size=1600,1000",
               "--user-data-dir=" + str(profile), "about:blank"]
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               cwd=str(ROOT), start_new_session=True)
    active = profile / "DevToolsActivePort"
    deadline = time.monotonic() + min(timeout, 25)
    port = None
    while time.monotonic() < deadline:
        if active.is_file():
            try:
                port = int(active.read_text().splitlines()[0])
                break
            except (ValueError, IndexError, OSError):
                pass
        if process.poll() is not None:
            break
        time.sleep(.2)
    log.close()
    if port is None:
        tail = log_path.read_text(errors="replace")[-1000:] if log_path.is_file() else ""
        process.terminate()
        raise CaptureError("Chrome DevTools did not start: {}".format(tail))
    targets = _get_json("http://127.0.0.1:{}/json/list".format(port))
    page = next((x for x in targets if x.get("type") == "page"), None)
    if not page:
        process.terminate()
        raise CaptureError("Chrome did not expose a page target")
    return process, page["webSocketDebuggerUrl"]


def _launch_streamlit(url, results_root, timeout):
    parsed = urlparse(url)
    if parsed.hostname not in ("127.0.0.1", "localhost"):
        raise CaptureError("--launch-app supports localhost URLs only")
    python = ROOT / ".venv/bin/python"
    if not python.is_file():
        raise CaptureError("Project .venv/bin/python is missing")
    lib = "/export/usr/python/3.8.9/lib"
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = lib + ":" + env.get("LD_LIBRARY_PATH", "")
    port = parsed.port or 8501
    command = [str(python), "-m", "streamlit", "run", str(ROOT / "app/main.py"),
               "--server.headless", "true", "--server.address", "127.0.0.1",
               "--server.port", str(port), "--server.fileWatcherType", "none",
               "--", "--results", str(results_root)]
    log = tempfile.TemporaryFile(mode="w+b")
    process = subprocess.Popen(command, cwd=str(ROOT), env=env,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + min(timeout, 35)
    while time.monotonic() < deadline:
        if _health(url):
            return process, log
        if process.poll() is not None:
            log.seek(0)
            raise CaptureError("Streamlit exited: {}".format(log.read()[-1200:].decode(errors="replace")))
        time.sleep(.4)
    process.terminate()
    raise CaptureError("Streamlit did not become reachable within {} s".format(timeout))


def capture(args):
    cjk_font = _cjk_font()
    results_root = _path(args.results)
    formal_inputs = _path(args.formal_inputs)
    split_hash, split_reasons = _split_information(results_root)
    formal_hash, formal_reasons = _formal_readiness(formal_inputs)
    reasons = split_reasons + formal_reasons
    if formal_hash and split_hash and formal_hash != split_hash:
        reasons.append("正式交付结果 split hash 与当前 manifest 不一致")
    formal = bool(split_hash and formal_hash == split_hash and not reasons)
    status = "formal" if formal else "smoke"
    output = _path(args.output) if args.output else ROOT / ("results/formal_ui.png" if formal else "artifacts/smoke/ui_capture.png")
    metadata = _path(args.metadata) if args.metadata else ROOT / ("results/formal_ui_capture.json" if formal else "artifacts/smoke/ui_capture.json")
    if not formal and (output.name.startswith("formal_") or metadata.name.startswith("formal_")):
        raise CaptureError("Smoke capture cannot be written with a formal_* filename")
    if output.suffix.lower() != ".png" or metadata.suffix.lower() != ".json":
        raise CaptureError("Output must be PNG and metadata must be JSON")
    if output.parent != metadata.parent and args.metadata is None:
        raise CaptureError("Default metadata must share the screenshot directory")
    url = args.url.rstrip("/")
    server_process = server_log = None
    chrome_process = None
    cdp = None
    try:
        if not _health(url):
            if not args.launch_app:
                raise CaptureError("Streamlit is unreachable at {}; use --launch-app".format(url))
            server_process, server_log = _launch_streamlit(url, results_root, args.timeout)
        with tempfile.TemporaryDirectory(prefix="odebrain_chrome_") as temporary:
            chrome_process, debugger = _chrome_target(Path(temporary), _path(args.chrome), args.timeout)
            cdp = CDP(debugger)
            cdp.call("Page.enable")
            cdp.call("Runtime.enable")
            # Streamlit 1.31 uses replaceAll, added after Chrome 78. Install the
            # compatibility shim before any app JavaScript executes.
            cdp.call("Page.addScriptToEvaluateOnNewDocument", {"source": CHROME78_POLYFILLS})
            cdp.call("Emulation.setDeviceMetricsOverride", {"width": args.width,
                                                             "height": args.height,
                                                             "deviceScaleFactor": 1,
                                                             "mobile": False})
            cdp.call("Page.navigate", {"url": url})
            if args.page != "数据概览":
                # Wait for the Streamlit shell before clicking the sidebar radio.
                deadline = time.monotonic() + args.timeout
                while time.monotonic() < deadline:
                    state = cdp.evaluate(PROBE)
                    if state and any("HMS × ODEBrain" in h for h in state.get("headings", [])):
                        break
                    time.sleep(.4)
                else:
                    raise CaptureError("Streamlit shell never rendered")
                _click_page(cdp, args.page, args.timeout)
            min_charts = max(REQUIRED_CHARTS[args.page], args.min_charts or 0)
            probe = _wait_rendered(cdp, args.page, results_root, min_charts, args.timeout)
            if formal and re.search(r"real_hms_smoke|非正式证据|真实 HMS 子集 smoke", probe["visibleText"], re.I):
                raise CaptureError("Browser still displays smoke data after formal evidence validation")
            response = cdp.call("Page.captureScreenshot", {"format": "png", "fromSurface": True,
                                                           "captureBeyondViewport": False}, timeout=20)
            if not isinstance(response.get("data"), str):
                raise CaptureError("Chrome did not provide screenshot data")
            image_bytes = base64.b64decode(response["data"], validate=True)
            image_info = _validate_png(image_bytes)
            if not formal:
                image_bytes = _smoke_watermark(image_bytes)
                image_info = _validate_png(image_bytes)
    finally:
        if cdp is not None:
            cdp.close()
        if chrome_process is not None and chrome_process.poll() is None:
            chrome_process.terminate()
            try:
                chrome_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                chrome_process.kill()
        if server_process is not None and server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
        if server_log is not None:
            server_log.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    temporary_image = output.with_suffix(".png.tmp")
    temporary_image.write_bytes(image_bytes)
    os.replace(str(temporary_image), str(output))
    image_digest = _sha256(output)
    details = {
        "schema_version": "ui_capture_v1", "experiment_status": status,
        "split_hash": split_hash, "image": _relative(output),
        "image_sha256": image_digest, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "page": args.page, "url": url, "results_root": _relative(results_root),
        "chrome_version": subprocess.check_output([str(_path(args.chrome)), "--version"],
                                                  text=True).strip(),
        "cjk_font": cjk_font,
        "rendered_charts": probe["drawnCharts"], "render_probe": {
            "headings": probe["headings"], "charts": probe["charts"],
            "drawn_charts": probe["drawnCharts"], "skeletons": probe["skeletons"],
            "body_width": probe["bodyWidth"], "body_height": probe["bodyHeight"]},
        "image_geometry": image_info, "formal_gate": {
            "passed": formal, "input_bundle": _relative(formal_inputs), "reasons": reasons},
        "smoke_watermark_added": not formal,
    }
    temporary_json = metadata.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(temporary_json), str(metadata))
    return details, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--results", default="results")
    parser.add_argument("--formal-inputs", default="results/formal_delivery_inputs.json")
    parser.add_argument("--page", choices=PAGES, default="数据概览")
    parser.add_argument("--output", help="PNG path; defaults to formal or smoke directory after validation")
    parser.add_argument("--metadata", help="JSON path; defaults to formal or smoke directory after validation")
    parser.add_argument("--chrome", default="/usr/bin/google-chrome")
    parser.add_argument("--launch-app", action="store_true", help="Start app with file watcher disabled if URL is down")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--min-charts", type=int, help="Extra minimum; never lower than the selected page's required chart count")
    parser.add_argument("--timeout", type=float, default=75)
    args = parser.parse_args()
    if args.width < 900 or args.height < 600 or (args.min_charts is not None and args.min_charts < 1) or args.timeout < 5:
        parser.error("Viewport must be at least 900x600, min-charts >= 1 and timeout >= 5 s")
    try:
        details, path = capture(args)
    except CaptureError as exc:
        parser.exit(1, "UI capture failed: {}\n".format(exc))
    print(json.dumps({"status": details["experiment_status"], "image": details["image"],
                      "metadata": _relative(path), "charts": details["rendered_charts"],
                      "split_hash": details["split_hash"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
