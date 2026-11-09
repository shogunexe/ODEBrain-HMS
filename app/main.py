"""Five-page, result-file-driven HMS/ODEBrain dashboard.

Run from repository root: streamlit run app/main.py -- --results results
No model training or implicit inference occurs during UI interaction.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.hms import CLASSES, ELECTRODES, manifest_hash


def _args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results", default=str(ROOT / "results"))
    known, _ = parser.parse_known_args()
    return Path(known.results).resolve()


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained(root, relative):
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError("Result index contains a path outside the result root")
    return path


def _empty(detail):
    st.info("未生成：" + detail)


def _plot(fig, key, container=None, meta=None):
    """Show a Plotly view and offer on-demand PNG/SVG for this exact view."""
    if meta and meta.get("provenance") != "real_hms_checkpoint":
        label = ("REAL HMS SMOKE · 非正式证据" if meta.get("provenance") == "real_hms_smoke"
                 else "未验证 · 非正式证据")
        fig.add_annotation(text=label, x=.5, y=.5,
                           xref="paper", yref="paper", showarrow=False,
                           opacity=.35, textangle=-20,
                           font=dict(color="crimson", size=30))
    ui = container if container is not None else st
    ui.plotly_chart(fig, use_container_width=True)
    figure_key = key + "_" + hashlib.sha1(fig.to_json().encode("utf-8")).hexdigest()[:12]
    cache_key = "static_" + figure_key
    if ui.button("生成此图 PNG / SVG", key="prepare_" + figure_key):
        try:
            st.session_state[cache_key] = (fig.to_image(format="png", width=1200, height=700, scale=2),
                                           fig.to_image(format="svg", width=1200, height=700))
        except Exception as exc:
            ui.warning("静态导出未生成：%s" % exc)
    if cache_key in st.session_state:
        png, svg = st.session_state[cache_key]
        ui.download_button("下载 PNG", png, file_name=figure_key + ".png", mime="image/png",
                           key="download_png_" + figure_key)
        ui.download_button("下载 SVG", svg, file_name=figure_key + ".svg", mime="image/svg+xml",
                           key="download_svg_" + figure_key)


def _context(meta, time_unit="秒"):
    st.caption("模型版本：%s ｜ 样本：%s / %s / %s ｜ 时间单位：%s ｜ 来源：%s" % (
        meta.get("model_id", "未记录"), meta.get("split", "未记录"),
        meta.get("patient_id", "未记录"), meta.get("window_id", "未记录"),
        time_unit, meta.get("provenance", "未记录")))


def _download_figures(case_dir, meta):
    figures = meta.get("figures", {})
    if not figures:
        _empty("当前案例没有静态图文件")
        return
    with st.expander("下载静态证据图（PNG / SVG）"):
        for title, variants in figures.items():
            cols = st.columns(2)
            for j, ext in enumerate(("png", "svg")):
                file = case_dir / variants.get(ext, "")
                if file.is_file():
                    cols[j].download_button("%s · %s" % (title, ext.upper()),
                                            file.read_bytes(), file_name=file.name,
                                            mime="image/png" if ext == "png" else "image/svg+xml",
                                            key="%s_%s_%s" % (case_dir.name, title, ext))


def _case_picker(result_root):
    final_root = result_root / "dynamics_refit"
    dynamics_root = (final_root if (final_root / "result_index.json").is_file()
                     else result_root / "dynamics")
    index = _json(dynamics_root / "result_index.json")
    if not index or not index.get("cases"):
        _empty("缺少正式或初训动力学案例。请先运行 scripts/export_dynamics.py。")
        return None
    if index.get("provenance") != "real_hms_checkpoint":
        st.warning("真实 HMS 子集 smoke：案例使用真实数据，但训练/文件覆盖尚未达到正式实验条件。覆盖率：%.2f%%。图仅供流程验证，不能作为正式证据。" % (
            100 * float(index.get("manifest_file_coverage", 0))))
    choices = index["cases"]
    case_labels = ["%d · %s · %s · %s" % (i, item.get("split", "?"), item.get("window_id", "?"),
                                             ", ".join(item.get("tags", []))) for i, item in enumerate(choices)]
    selected = choices[case_labels.index(st.selectbox("案例", case_labels))]
    try:
        meta_path = _contained(dynamics_root, selected["metadata"])
        data_path = _contained(dynamics_root, selected["data"])
    except (KeyError, ValueError) as exc:
        st.error(str(exc))
        return None
    meta = _json(meta_path)
    if not meta or not data_path.is_file():
        _empty("案例 JSON 或 NPZ 数据文件缺失")
        return None
    try:
        with np.load(str(data_path), allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
    except (OSError, ValueError) as exc:
        st.error("案例数组不可读取：%s" % exc)
        return None
    if meta.get("model_id") != index.get("model_id") or meta.get("split_hash") != index.get("split_hash"):
        st.error("案例与结果索引的模型版本或患者划分不一致")
        return None
    if meta.get("provenance") != "real_hms_checkpoint":
        st.warning("当前案例为 real_hms_smoke；导出的静态图带非正式证据水印。")
    return arrays, meta, meta_path.parent


def page_data(result_root):
    st.header("1 · 数据概览")
    candidates = sorted(result_root.rglob("*manifest*.csv")) if result_root.exists() else []
    candidates += sorted((ROOT / "artifacts").rglob("*manifest*.csv")) if (ROOT / "artifacts").exists() else []
    curated = ROOT / "data" / "curated" / "split_manifest.csv"
    if curated.is_file():
        candidates.insert(0, curated)
    if not candidates:
        _empty("未发现患者级审计 manifest CSV；数据规模、类别、缺失和分歧图尚无依据。")
        return
    path_labels = [str(p.relative_to(ROOT)) if ROOT in p.parents else str(p) for p in candidates]
    path = candidates[path_labels.index(st.selectbox("患者级 manifest", path_labels))]
    try:
        df = pd.read_csv(str(path))
    except Exception as exc:
        st.error("manifest 不可读取：%s" % exc)
        return
    needed = {"patient_id", "split", "audit_reason", "class_id"}
    if not needed.issubset(df.columns):
        _empty("manifest 缺少 patient_id / split / audit_reason / class_id")
        return
    st.metric("标注窗口", format(len(df), ",d"))
    c1, c2 = st.columns(2)
    split = df.groupby("split")["patient_id"].agg(["size", "nunique"])
    _plot(go.Figure([go.Bar(x=split.index.tolist(), y=split["size"].tolist(), name="窗口"),
                     go.Bar(x=split.index.tolist(), y=split["nunique"].tolist(), name="患者")]).update_layout(
                         barmode="group", title="患者级划分"), "data_split", c1)
    audit = df.audit_reason.value_counts()
    _plot(go.Figure(go.Bar(x=audit.index.tolist(), y=audit.tolist())).update_layout(title="审计 / 缺失原因"), "data_audit", c2)
    valid = df[df.audit_reason == "ok"]
    counts = valid.class_id.value_counts().reindex(range(len(CLASSES)), fill_value=0)
    _plot(go.Figure(go.Bar(x=list(CLASSES), y=counts.tolist())).update_layout(title="有效窗口类别数"), "data_class_counts")
    vote_cols = [c + "_vote" for c in CLASSES]
    if set(vote_cols).issubset(valid.columns) and len(valid):
        votes = valid[vote_cols].to_numpy(dtype=float)
        denom = votes.sum(axis=1, keepdims=True)
        probabilities = np.divide(votes, denom, out=np.zeros_like(votes), where=denom > 0)
        ent = -np.sum(np.where(probabilities > 0, probabilities * np.log(np.maximum(probabilities, 1e-15)), 0), axis=1)
        _plot(go.Figure(go.Histogram(x=ent, nbinsx=30)).update_layout(
            title="专家投票熵分布", xaxis_title="nats"), "data_vote_entropy")
    else:
        _empty("专家投票列缺失，分歧图未生成")
    st.caption("来源：%s。计数基于 manifest；仅 audit_reason=ok 的窗口进入类别及分歧图。" % path)


def page_sample(result_root):
    st.header("2 · 样本分析")
    picked = _case_picker(result_root)
    if picked is None:
        return
    a, meta, case_dir = picked
    _context(meta)
    eeg = a["classification_eeg"]
    sr = meta["sample_rate_hz"]
    t = np.arange(eeg.shape[1]) / sr
    selected = st.multiselect("导联", list(ELECTRODES), default=list(ELECTRODES[:5]))
    if selected:
        fig = go.Figure()
        for lead in selected:
            i = ELECTRODES.index(lead)
            fig.add_trace(go.Scatter(x=t, y=eeg[i], mode="lines", name=lead))
        fig.add_vrect(x0=20, x1=30, fillcolor="gray", opacity=.15, line_width=0)
        fig.update_layout(title="分类分支观测 EEG：完整 0–50 秒窗口",
                          xaxis_title="窗口时间（秒）", yaxis_title="模型输入幅度")
        _plot(fig, "sample_eeg_" + case_dir.name, meta=meta)
    st.caption("灰色中央标注区是 20–30 秒；分类分支使用完整 50 秒，独立的严格预测分支只读取前 %s 秒。未标注时间没有逐秒标签。" % meta["forecast_cut_s"])
    from scipy.signal import spectrogram
    freq, sec, power = spectrogram(eeg[0], fs=sr, nperseg=min(256, eeg.shape[1]))
    keep = freq <= 45
    fig = go.Figure(go.Heatmap(x=sec, y=freq[keep], z=np.log1p(power[keep]), colorscale="Viridis"))
    fig.update_layout(title="%s 的观测频谱" % ELECTRODES[0], xaxis_title="秒", yaxis_title="Hz")
    _plot(fig, "sample_spectrogram_" + case_dir.name, meta=meta)
    fig = go.Figure([go.Bar(name="专家票数比例", x=list(CLASSES), y=a["expert_votes"]),
                     go.Bar(name="模型概率", x=list(CLASSES), y=a["probabilities"])])
    fig.update_layout(barmode="group", title="专家投票与单次模型分类概率", yaxis_title="比例")
    _plot(fig, "sample_votes_" + case_dir.name, meta=meta)
    st.caption("专家票数对应 20–30 秒中央标注区；模型分类概率由完整 0–50 秒窗口给出一组值，不是逐秒预测。来源：%s。" % meta["model_id"])
    _download_figures(case_dir, meta)


def _metric_rows(result_root):
    rows = []
    if not result_root.exists():
        return rows
    for path in sorted(result_root.rglob("*.json")):
        if path.name in ("case.json", "result_index.json"):
            continue
        obj = _json(path)
        if not isinstance(obj, dict):
            continue
        if "models" in obj and "counts" in obj:
            rows.extend(_spark_metric_rows(path, obj))
        elif "kl_window_mean" in obj or "macro_f1" in obj:
            split_hash = obj.get("split_hash")
            file_coverage = 0.0
            full_row_coverage = False
            split_coverage = False
            run_dir = Path(obj.get("run_dir", ""))
            if not run_dir.is_absolute():
                run_dir = ROOT / run_dir
            config = _json(run_dir / "config.json")
            phase = ("train_val_refit" if obj.get("training_protocol") ==
                     "train_val_fixed_epoch_refit" else "initial_train_only")
            seed = (config.get("refit", {}).get("seed", 42) if config else 42)
            if config:
                manifest = Path(config.get("data", {}).get("manifest", "data/curated/split_manifest.csv"))
                if not manifest.is_absolute():
                    manifest = ROOT / manifest
                if manifest.is_file():
                    try:
                        manifest_table = pd.read_csv(str(manifest))
                        actual_hash = manifest_hash(manifest_table)
                        if split_hash and split_hash != actual_hash:
                            raise ValueError("metric split hash differs from manifest")
                        split_hash = actual_hash
                        if "file_exists" in manifest_table:
                            file_coverage = float(manifest_table.groupby("eeg_id")["file_exists"].max().mean())
                        source_metadata = ROOT / "data" / "raw" / "train.csv"
                        if source_metadata.is_file():
                            full_row_coverage = len(manifest_table) == len(pd.read_csv(str(source_metadata), usecols=["eeg_id"]))
                        selected = manifest_table[(manifest_table.split == obj.get("split")) &
                                                  (manifest_table.audit_reason == "ok")]
                        split_coverage = (len(selected) > 0 and obj.get("n_windows") == len(selected)
                                          and obj.get("n_patients") == selected.patient_id.nunique())
                    except (OSError, ValueError, KeyError):
                        pass
            if obj.get("smoke_subset") is True or "smoke" in str(obj.get("experiment_status", "")).lower():
                provenance = "real_hms_smoke"
            elif (obj.get("smoke_subset") is False and file_coverage == 1.0
                  and full_row_coverage and split_coverage
                  and obj.get("experiment_status") == "formal"
                  and obj.get("synthetic") is False
                  and phase == "train_val_refit"
                  and obj.get("split") == "test"
                  and obj.get("test_used_for_selection") is False
                  and config is not None
                  and config.get("schema_version") == "hms_final_refit_v1"
                  and (run_dir / "final.pt").is_file()
                  and (completed := _json(run_dir / "completed.json")) is not None
                  and completed.get("final_checkpoint_sha256") ==
                      obj.get("checkpoint_sha256") == _sha256(run_dir / "final.pt")):
                provenance = "formal"
            elif (obj.get("smoke_subset") is False and file_coverage == 1.0
                  and full_row_coverage and split_coverage
                  and obj.get("evaluation_scope") == "full_split"):
                provenance = "formal"
            else:
                provenance = "unverified"
            run_name = Path(obj.get("run_dir", path.parent)).name
            model_name = obj.get("model_id") or "%s:%s" % (obj.get("model", path.parent.name), run_name)
            if phase == "train_val_refit":
                model_name += " · train+val 重训 · seed %s" % seed
            rows.append({"model": model_name,
                         "split": obj.get("split", path.name.split("_")[0]),
                         "phase": phase,
                         "seed": seed,
                         "split_hash": split_hash or "未记录",
                         "provenance": provenance, "file_coverage": file_coverage,
                         "KL": obj.get("kl_window_mean"), "Macro-F1": obj.get("macro_f1"),
                         "windows": obj.get("n_windows"), "patients": obj.get("n_patients"),
                         "path": path, "confusion_matrix": obj.get("confusion_matrix")})
    return rows


def _spark_metric_rows(path, obj):
    """Flatten Spark LR/RF metrics without promoting incomplete runs to formal."""
    models = obj.get("models", {})
    features = Path(str(obj.get("features_path", "")))
    if not isinstance(models, dict) or not obj.get("features_path"):
        return []
    if not features.is_absolute():
        features = ROOT / features
    summary = _json(features.with_name(features.name + "_summary.json"))
    if not isinstance(summary, dict) or not summary.get("manifest"):
        return []
    manifest = Path(summary["manifest"])
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    try:
        table = pd.read_csv(str(manifest))
        split_hash = manifest_hash(table)
        file_coverage = float(table.groupby("eeg_id")["file_exists"].max().mean())
        metadata = pd.read_csv(str(ROOT / "data/raw/train.csv"), usecols=["eeg_id"])
        row_coverage = (len(table) == len(metadata) and
                        table.eeg_id.value_counts().sort_index().equals(
                            metadata.eeg_id.value_counts().sort_index()))
        manifest_relative = str(manifest.resolve().relative_to(ROOT))
    except (OSError, ValueError, KeyError):
        return []
    result = []
    for name in ("lr", "rf"):
        model = models.get(name, {})
        if not isinstance(model, dict):
            continue
        model_dir = Path(str(model.get("model_path", "")))
        if not model_dir.is_absolute():
            model_dir = ROOT / model_dir
        for split, metric in model.get("splits", {}).items():
            if split not in ("train", "val", "test") or not isinstance(metric, dict):
                continue
            subset = table[(table.split == split) & (table.audit_reason == "ok")]
            windows, patients = len(subset), int(subset.patient_id.nunique())
            aligned = (metric.get("n") == windows and
                       obj.get("counts", {}).get(split) == windows and
                       summary.get("by_split", {}).get(split, {}).get("valid") == windows and
                       summary.get("manifest") in (manifest_relative, str(manifest)) and
                       summary.get("output") == obj.get("features_path"))
            confusion = metric.get("confusion")
            scores_valid = (all(isinstance(metric.get(key), (int, float)) and
                                math.isfinite(metric[key]) for key in ("kl_window_mean", "macro_f1"))
                            and isinstance(confusion, list) and len(confusion) == len(CLASSES)
                            and all(isinstance(row, list) and len(row) == len(CLASSES)
                                    for row in confusion)
                            and sum(sum(row) for row in confusion) == windows)
            if obj.get("experiment_status") == "real_data_smoke" or "smoke" in str(obj.get("experiment_status", "")).lower():
                provenance = "real_hms_smoke"
            elif (obj.get("experiment_status") == "formal" and
                  summary.get("experiment_status") == "formal" and
                  obj.get("target_source") == "expert_votes" and
                  obj.get("label_order") == list(CLASSES) and model_dir.is_dir() and
                  file_coverage == 1.0 and row_coverage and aligned and scores_valid):
                provenance = "formal"
            else:
                provenance = "unverified"
            result.append({"model": "Spark {}".format(name.upper()), "split": split,
                           "phase": "spark_train_only",
                           "seed": 42,
                           "split_hash": split_hash, "provenance": provenance,
                           "file_coverage": file_coverage,
                           "KL": metric.get("kl_window_mean"),
                           "Macro-F1": metric.get("macro_f1"),
                           "windows": metric.get("n"), "patients": patients,
                           "path": path, "confusion_matrix": metric.get("confusion")})
    return result


def _ablation_rows(path, split_hash, split, windows, patients):
    """Read the formal JSON produced by run_ablations, scoped to one split."""
    obj = _json(path)
    if not isinstance(obj, dict) or not isinstance(obj.get("rows"), list):
        return None
    if obj.get("split_hash") != split_hash or obj.get("split") != split:
        return None
    rows = obj["rows"]
    if not rows or any(not isinstance(row, dict) or not row.get("name") or
                       not isinstance(row.get("kl_window_mean"), (int, float)) or
                       not math.isfinite(row["kl_window_mean"])
                       for row in rows):
        return None
    if obj.get("smoke_subset") is True or "smoke" in str(obj.get("experiment_status", "")).lower():
        provenance = "real_hms_smoke"
    elif (obj.get("schema_version") == "ode_ablation_formal_v1" and
          obj.get("experiment_status") == "formal" and obj.get("synthetic") is False
          and obj.get("smoke_subset") is False and obj.get("n_windows") == windows
          and obj.get("n_patients") == patients):
        provenance = "formal"
    else:
        provenance = "unverified"
    return pd.DataFrame([{"实验": row["name"], "窗口 KL": row["kl_window_mean"],
                          "患者等权 KL": row.get("kl_patient_equal_mean"),
                          "Macro-F1": row.get("macro_f1"), "证据状态": provenance}
                         for row in rows])


def page_models(result_root):
    st.header("3 · 模型对比")
    rows = _metric_rows(result_root)
    if not rows:
        _empty("缺少正式 metrics JSON；KL、Macro-F1、混淆矩阵和消融结果未生成。")
        return
    table = pd.DataFrame([{k: r[k] for k in ("model", "split", "phase", "seed", "split_hash", "provenance", "file_coverage",
                                                   "KL", "Macro-F1", "windows", "patients")} for r in rows])
    groups = sorted(set((r["split_hash"], r["split"], r["phase"], r["seed"]) for r in rows))
    group_labels = ["%s · %s · seed %s · %s" % (item[1], item[2], item[3], item[0]) for item in groups]
    formal_test = next((i for i, (hash_value, split_name, phase, seed) in enumerate(groups)
                        if split_name == "test" and phase == "train_val_refit" and seed == 42 and all(
                            row["provenance"] == "formal" for row in rows
                            if row["split_hash"] == hash_value and row["split"] == split_name
                            and row["phase"] == phase and row["seed"] == seed)), 0)
    split_hash, split, phase, seed = groups[group_labels.index(st.selectbox(
        "仅比较相同患者划分", group_labels, index=formal_test))]
    section = table[(table["split"] == split) & (table["split_hash"] == split_hash)
                    & (table["phase"] == phase) & (table["seed"] == seed)]
    st.dataframe(section, use_container_width=True)
    if split_hash == "未记录":
        st.warning("这些指标文件没有 split_hash，无法验证它们使用同一患者划分。")
    if (section["provenance"] != "formal").any():
        st.warning("此图包含 smoke 或未验证指标；非正式模型比较。请查看表中的证据状态。")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("KL ↓", "Macro-F1 ↑"))
    fig.add_trace(go.Bar(x=section.model, y=section["KL"], name="KL"), row=1, col=1)
    fig.add_trace(go.Bar(x=section.model, y=section["Macro-F1"], name="Macro-F1"), row=1, col=2)
    chart_meta = ({"provenance": "real_hms_smoke" if
                   (section["provenance"] == "real_hms_smoke").any() else "unverified"}
                  if (section["provenance"] != "formal").any() else None)
    _plot(fig, "model_metrics_" + split + "_" + split_hash[:12], meta=chart_meta)
    eligible = [r for r in rows if r["split"] == split and r["split_hash"] == split_hash
                and r["phase"] == phase and r["seed"] == seed
                and r["confusion_matrix"] is not None]
    if eligible:
        model_labels = ["%d · %s" % (i, item["model"]) for i, item in enumerate(eligible)]
        row = eligible[model_labels.index(st.selectbox("混淆矩阵模型", model_labels))]
        _plot(go.Figure(go.Heatmap(z=row["confusion_matrix"], x=list(CLASSES), y=list(CLASSES),
                                   colorscale="Blues")).update_layout(xaxis_title="预测", yaxis_title="真实"),
              "model_confusion_" + row["model"].replace(":", "_") + "_" + split,
              meta={"provenance": row["provenance"]} if row["provenance"] != "formal" else None)
        st.caption("模型版本：%s；划分：%s；有效窗口：%s；患者：%s；来源：%s。" % (
            row["model"], split, row["windows"], row["patients"], row["path"]))
    else:
        _empty("该划分没有混淆矩阵")
    reference = section[section.provenance == "formal"]
    if reference.empty:
        reference = section
    expected_windows = int(reference.iloc[0]["windows"]) if pd.notna(reference.iloc[0]["windows"]) else -1
    expected_patients = int(reference.iloc[0]["patients"]) if pd.notna(reference.iloc[0]["patients"]) else -1
    ablations = [(path, frame) for path in sorted(result_root.rglob("*ablation*.json"))
                 if (frame := _ablation_rows(path, split_hash, split,
                                             expected_windows, expected_patients)) is not None]
    if ablations:
        labels = [str(path) for path, _ in ablations]
        path, frame = ablations[labels.index(st.selectbox("消融结果", labels))]
        st.dataframe(frame, use_container_width=True)
        if (frame["证据状态"] != "formal").any():
            st.warning("所选消融记录为 smoke 或未验证结果，不作正式结论。")
        st.caption("来源：%s；划分：%s；患者级 split：%s。" % (path, split, split_hash))
    else:
        _empty("当前划分没有可核对的消融 JSON；正式消融结果尚未生成")
    errors = sorted(result_root.rglob("*error*case*.csv"))
    if errors:
        path = Path(st.selectbox("错误案例表", [str(p) for p in errors]))
        st.dataframe(pd.read_csv(str(path)).head(100), use_container_width=True)
    else:
        prediction_files = sorted({path for row in rows
                                   if row["split"] == split and row["split_hash"] == split_hash
                                   and row["phase"] == phase and row["seed"] == seed
                                   for path in [row["path"].parent / (split + "_predictions.csv")]
                                   if path.is_file()})
        if prediction_files:
            path = Path(st.selectbox("从正式预测文件查看错误案例", [str(p) for p in prediction_files]))
            frame = pd.read_csv(str(path))
            prediction_cols = ["pred_" + name for name in CLASSES]
            if set(prediction_cols + ["class_id", "window_id", "patient_id"]).issubset(frame.columns):
                predicted = frame[prediction_cols].to_numpy().argmax(axis=1)
                mistakes = frame[predicted != frame["class_id"].to_numpy()]
                st.dataframe(mistakes.head(100), use_container_width=True)
                st.caption("来源：%s。仅列真实文件中的误分类窗口；最多显示 100 行。" % path)
            else:
                _empty("预测文件缺少 class_id、身份或六类概率")
        else:
            _empty("错误案例或预测 CSV 未生成")


ELECTRODE_XY = {
    "Fp1": (-2, 4), "Fp2": (2, 4), "F7": (-4, 2.8), "F3": (-2, 2.5),
    "Fz": (0, 2.5), "F4": (2, 2.5), "F8": (4, 2.8), "T3": (-5, 0),
    "C3": (-2, 0), "Cz": (0, 0), "C4": (2, 0), "T4": (5, 0),
    "T5": (-4, -2.8), "P3": (-2, -2.5), "Pz": (0, -2.5),
    "P4": (2, -2.5), "T6": (4, -2.8), "O1": (-2, -4), "O2": (2, -4),
}


def page_dynamics(result_root):
    st.header("4 · ODE 动力学")
    picked = _case_picker(result_root)
    if picked is None:
        return
    a, meta, case_dir = picked
    _context(meta)
    st.caption("本页轨迹和未来图误差来自只读取前 5 秒的预测分支；分类概率来自完整 50 秒的分类分支。潜在状态不是直接测量的神经元状态。轨迹插值点不等于新增 EEG 观测；不同模型的 PCA 坐标不可直接比较。")
    p = a["projected"]
    v = a["projected_field"]
    t = a["times_s"]
    frame = st.slider("轨迹播放位置", 0, len(t)-1, 0,
                      format="%d")
    is_3d = st.checkbox("三维轨迹", value=False, disabled=p.shape[1] < 3)
    if is_3d:
        fig = go.Figure(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines+markers",
                                     marker=dict(color=t, colorscale="Viridis", size=5, colorbar=dict(title="秒"))))
        fig.add_trace(go.Scatter3d(x=[p[frame, 0]], y=[p[frame, 1]], z=[p[frame, 2]],
                                   mode="markers", marker=dict(color="red", size=9), name="当前时点"))
        fig.update_layout(scene=dict(xaxis_title="PCA 1", yaxis_title="PCA 2", zaxis_title="PCA 3"))
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines+markers",
                                 marker=dict(color=t, colorscale="Viridis", size=9, colorbar=dict(title="秒")),
                                 text=["%.2f s" % value for value in t], name="轨迹"))
        for i in range(len(t)):
            fig.add_annotation(x=p[i, 0] + v[i, 0]*.15, y=p[i, 1] + v[i, 1]*.15,
                               ax=p[i, 0], ay=p[i, 1], axref="x", ayref="y", showarrow=True,
                               arrowhead=2, arrowcolor="#d62728")
        fig.add_trace(go.Scatter(x=[p[frame, 0]], y=[p[frame, 1]], mode="markers",
                                 marker=dict(color="red", size=16), name="当前时点"))
        fig.update_layout(xaxis_title="训练集 PCA 1", yaxis_title="训练集 PCA 2")
    fig.update_layout(title="潜在轨迹：起点 %.1f s → 终点 %.1f s" % (t[0], t[-1]))
    _plot(fig, "ode_trajectory_%s_%d_%d" % (case_dir.name, frame, int(is_3d)), meta=meta)
    st.write("当前轨迹时间：%.2f 秒；投影坐标：%s；投影导数 du/dt：%s" % (
        t[frame], np.round(p[frame], 4), np.round(v[frame], 4)))
    st.caption("箭头在真实高维轨迹处计算 f(z,t)，再用训练集 PCA 的同一线性变换投影；仅表示观测轨迹上的模型方向。")

    fig = make_subplots(rows=2, cols=2, subplot_titles=("状态速度（每秒）", "相邻方向变化（弧度）",
                                                      "未来节点属性误差", "不确定性（nats）"))
    fig.add_trace(go.Scatter(x=t, y=a["speed_per_s"], mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t[1:], y=a["turn_angle_rad"], mode="lines+markers"), row=1, col=2)
    fig.add_trace(go.Bar(x=a["future_times_s"], y=a["forecast_mse"]), row=2, col=1)
    fig.add_trace(go.Bar(x=["模型预测熵", "专家票数熵"],
                         y=[meta["prediction_entropy"], meta["expert_entropy"]]), row=2, col=2)
    fig.update_layout(height=700, showlegend=False)
    _plot(fig, "ode_quantities_" + case_dir.name, meta=meta)
    st.caption("全节点状态空间速度与轨迹长度依赖该模型隐空间尺度；图误差仅按有效未来节点计算。积分轨迹长度：%.4g，近零导数方向对：%d。" % (
        meta["path_length_integral"], meta["near_zero_turn_pairs"]))

    horizon_labels = ["+%.0f 秒" % sec for sec in a["future_times_s"]]
    horizon = horizon_labels.index(st.selectbox("未来图属性时点（秒）", horizon_labels))
    band = st.selectbox("节点频带属性", ["delta", "theta", "alpha", "beta", "gamma"])
    b = ["delta", "theta", "alpha", "beta", "gamma"].index(band)
    values = np.concatenate([a["future_graph_nodes"][:, :, b].ravel(),
                             a["predicted_graph_nodes"][:, :, b].ravel()])
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if lo == hi:
        hi = lo + 1e-6
    graph = make_subplots(rows=1, cols=2, subplot_titles=("观测未来节点属性", "模型预测节点属性"))
    xy = np.asarray([ELECTRODE_XY[e] for e in ELECTRODES])
    for col, key in enumerate(("future_graph_nodes", "predicted_graph_nodes"), start=1):
        graph.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text",
                                   text=list(ELECTRODES), textposition="top center",
                                   marker=dict(size=20, color=a[key][horizon, :, b], colorscale="RdBu",
                                               cmin=lo, cmax=hi, showscale=col == 2),
                                   showlegend=False), row=1, col=col)
    graph.update_layout(height=450)
    _plot(graph, "ode_graph_%s_%d_%d" % (case_dir.name, horizon, b), meta=meta)
    st.caption("电极位置是 10–20 布局示意；两图共用色标。模型仅预测节点属性，没有预测连接。观测相关矩阵、GNN 权重不代表因果或解剖连接。")
    with st.expander("观测相关性图（输入时点）"):
        observed_labels = ["%d–%d 秒" % (i, i+1) for i in range(len(a["graph_adj"]))]
        observed_frame = observed_labels.index(st.selectbox("已观测图时点（秒）", observed_labels))
        _plot(go.Figure(go.Heatmap(z=a["graph_adj"][observed_frame], x=list(ELECTRODES),
                                   y=list(ELECTRODES), zmin=-1, zmax=1,
                                   colorscale="RdBu")).update_layout(title="观测 EEG 相关矩阵；固定色标 −1 至 1"),
              "ode_correlation_%s_%d" % (case_dir.name, observed_frame), meta=meta)
    perturb = meta.get("perturbations", {})
    if perturb:
        names = list(perturb)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("分类概率 L1 变化", "平均潜在轨迹 L2 变化"))
        fig.add_trace(go.Bar(x=names, y=[perturb[n]["probability_l1"] for n in names]), row=1, col=1)
        fig.add_trace(go.Bar(x=names, y=[perturb[n]["trajectory_l2_mean"] for n in names]), row=1, col=2)
        _plot(fig, "ode_perturbation_" + case_dir.name, meta=meta)
        st.dataframe(pd.DataFrame([{"扰动": n, "概率 L1": perturb[n]["probability_l1"],
                                   "轨迹 L2": perturb[n]["trajectory_l2_mean"],
                                   "未来 MSE": perturb[n]["forecast_mse_by_horizon"]}
                                  for n in names]), use_container_width=True)
        st.caption("扰动幅度来自导出配置：%s。此处是输入敏感性分析，不是脑刺激或治疗效果证据。" % meta.get("perturbation_config"))
    else:
        _empty("扰动对照未生成")
    _download_figures(case_dir, meta)


def page_performance(result_root):
    st.header("5 · 大数据性能")
    paths = sorted(result_root.rglob("*benchmark*.json")) if result_root.exists() else []
    if result_root.exists():
        paths += sorted(result_root.rglob("runs.json"))
        paths += sorted(result_root.rglob("summary.json"))
        paths += sorted(result_root.rglob("*performance*.csv"))
    if not paths:
        _empty("缺少实际 Spark/数据管线基准结果；耗时、吞吐、资源和扩展曲线未生成。")
        return
    path = Path(st.selectbox("实际基准文件", [str(p) for p in paths]))
    try:
        obj = pd.read_csv(str(path)) if path.suffix == ".csv" else _json(path)
    except Exception as exc:
        st.error("基准文件不可读取：%s" % exc)
        return
    if isinstance(obj, list):
        df = pd.DataFrame(obj)
    elif isinstance(obj, dict):
        if isinstance(obj.get("runs"), list):
            df = pd.DataFrame(obj["runs"])
        elif isinstance(obj.get("benchmarks"), list):
            df = pd.DataFrame(obj["benchmarks"])
        else:
            st.json(obj)
            st.caption("运行配置与资源字段见原始 JSON；没有可用于扩展曲线的多次运行表。")
            return
    else:
        df = obj
    st.dataframe(df, use_container_width=True)
    if "synthetic" in df and df["synthetic"].astype(bool).any():
        st.warning("包含 synthetic=True 的运行；这些数据仅用于 smoke 验证，不能作为正式性能结论。")
    x_candidates = [x for x in ("workers", "partitions", "windows", "input_rows", "input_gib", "dataset_gib") if x in df]
    y_candidates = [x for x in ("median_end_to_end_seconds", "end_to_end_seconds", "compute_and_write_seconds",
                                 "windows_per_second", "wall_seconds", "duration_seconds", "elapsed_s",
                                 "throughput_rows_s", "rows_per_second") if x in df]
    if x_candidates and y_candidates:
        x = st.selectbox("扩展变量", x_candidates)
        y = st.selectbox("性能指标", y_candidates)
        _plot(go.Figure(go.Scatter(x=df[x], y=df[y], mode="lines+markers")).update_layout(
            xaxis_title=x, yaxis_title=y, title="实际运行扩展曲线"),
              "performance_%s_%s_%s" % (path.stem, x, y))
    else:
        _empty("运行文件没有可识别的规模与耗时/吞吐字段，扩展曲线未生成")
    st.caption("来源：%s。时间单位按原始字段；只展示实际记录的运行配置，不外推未运行的集群性能。" % path)


def main():
    st.set_page_config(page_title="HMS × ODEBrain 结果浏览", layout="wide")
    root = _args()
    st.title("HMS × ODEBrain：真实实验结果浏览")
    st.sidebar.caption("结果目录：%s" % root)
    pages = {"数据概览": page_data, "样本分析": page_sample, "模型对比": page_models,
             "ODE 动力学": page_dynamics, "大数据性能": page_performance}
    page = st.sidebar.radio("页面", list(pages))
    pages[page](root)


if __name__ == "__main__":
    main()
