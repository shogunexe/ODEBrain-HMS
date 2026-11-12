#!/usr/bin/env python3
"""Build an editable Chinese ODEBrain presentation.

Default output is an honestly labelled DRAFT. --final is intentionally strict:
it requires a results/slides_evidence.json manifest and source images for every
experimental slide. A final PDF is never produced by this script.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / "slides"
EVIDENCE = ROOT / "results" / "slides_evidence.json"
PREVIEW_EVIDENCE = ROOT / "results" / "slides_evidence_preview.json"

# The whole-deck source manifest is deliberately required for formal output.
# Static diagrams and explanatory text are editable PowerPoint shapes.
EXPERIMENTAL = {4, 5, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
HEADLINE = {9, 13, 14, 22}
SOURCES = {
    "hms": "https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data",
    "ode_code": "https://github.com/HHJIAnmo/ODEBRAIN",
    "ode_paper": "https://arxiv.org/abs/2602.23285",
    "spark": "https://spark.apache.org/docs/latest/ml-classification-regression.html",
    "ode_solver": "https://github.com/rtqichen/torchdiffeq",
}

# Title, one-sentence claim, talking points, expected figure/evidence.
PAGES = [
 ("项目与贡献", "Spark 批处理与连续图动力学构成可复查的 EEG 分析流程。", ["HMS 六类异常脑电活动分类；短时图属性预测", "患者级划分、基线对照、动力学解释与系统展示", "课程项目：离线研究系统，不作临床诊断或发作预警"], "项目主线与交付状态"),
 ("背景与问题定义", "同一脑电片段可能存在专家分歧，分类应保留软标签信息。", ["输入：50 秒 EEG 上下文和中央 10 秒标签区", "输出：六类概率与可核对的短时图特征预测", "关键困难：患者差异、噪声、重叠窗口与泄漏风险"], "任务示意：上下文—标注区—概率"),
 ("研究目标与功能闭环", "四个研究问题连接处理效率、分类效果、连续建模和解释。", ["RQ1 Spark 吞吐；RQ2 时频融合；RQ3 ODE 与离散对照", "RQ4 轨迹、速度、图预测误差是否帮助复查", "数据审计 → 建模 → 严格评估 → 静态图与交互系统"], "研究问题与模块映射"),
 ("HMS 数据与六分类", "数据规模、标签分布和专家分歧须由原始文件审计得出。", ["类别：seizure / lpd / gpd / lrda / grda / other", "按实际偏移提取窗口；保留专家票数构成的软标签", "报告患者、文件、标注行、有效窗口与唯一时长"], "真实数据统计与类别分布"),
 ("质量、时间对齐与划分", "按患者分组是所有对照共享的评估前提。", ["固定患者级测试集；训练、验证、测试患者不相交", "初训只用 train 拟合标准化；最终重训在 train+val 重新拟合，均排除测试", "检查缺失、坏导联、长度、偏移和窗口重叠"], "审计表、划分统计、质量案例"),
 ("系统总体架构", "Spark、训练器、结果仓库和展示层各承担一段可追踪工作。", ["Kaggle 原始文件 → Spark QC/窗口索引/特征", "Parquet/训练分片 → Spark ML 与 PyTorch 模型", "预测与动力学导出 → 静态图和交互页面"], "可编辑架构图"),
 ("实际部署与资源", "部署叙述必须与实际运行记录一致。", ["初始训练节点为 2×V100S；后续接入共享目录的 A800 节点", "Spark 性能实验固定在单一物理机；GPU 训练按节点分别记录", "报告机器配置与数据规模，不冒充多机集群"], "环境快照与运行日志"),
 ("Spark 数据处理", "Spark 实际执行质量检查、窗口索引和特征批处理。", ["原始文件 → 分区任务 → QC/滤波/特征 → Parquet", "报告处理字节、有效窗口、失败率、耗时和吞吐", "保存 event log 与配置以复核任务是否真实运行"], "Spark DAG、event log、产物统计"),
 ("基线与时频融合", "所有模型使用相同患者划分和统一测试协议。", ["逻辑回归与随机森林：由 Spark MLlib 在人工特征上训练", "TCN、频谱 CNN、融合模型：同源 EEG 对照", "图 GRU：匹配输入与目标的离散动力学对照"], "模型结构及同划分对照"),
 ("ODEBrain 来源与改造", "HMS 适配属于本项目实现，不能当成论文已验证结论。", ["官方方法：图编码、ODE 演化、未来图节点预测与下游分类", "本项目：HMS 输入、六分类头、患者级协议与可视化", "若实现简化，明确标为 ODEBrain-inspired 并记录偏离"], "官方模块与本项目改造对应"),
 ("图构建与训练流程", "图预测目标必须是可观测的未来属性，而非单靠分类损失。", ["固定导联顺序；频谱作节点属性，边定义需说明", "只用过去 5 秒预测未来 1/3/5 秒图属性", "训练患者上预训练，再六分类微调；严格防未来泄漏"], "输入切点、预测 horizon 和损失示意"),
 ("ODE 动力学直观解释", "连续状态通过导数和积分演化，再解码为可核对的量。", ["编码器：可用 EEG 与图 → 潜在初态 z₀", "本项目向量场：dz/dt = fθ(z)；RK4 积分得到 z(t)", "图解码器预测节点属性；分类头输出六类概率"], "可编辑状态—导数—积分—双输出图"),
 ("训练与评价协议", "固定患者划分、各模型训练协议和指标计算口径。", ["主指标：KL(target || prediction)，float64 离线聚合", "补充：窗口、患者和 EEG 记录等权口径；配对差异区间仅描述结果", "图预测按 horizon 报误差；训练耗时单独记录"], "配置表、样本数、模型版本"),
 ("分类主结果", "主结果页只接受真实测试预测文件及同划分模型版本。", ["列出 MLlib 逻辑回归/随机森林、深度基线、图 GRU 与 ODE 模型", "表头写明患者数、窗口数、split hash 和指标定义", "不以论文指标代替本项目实验，也不预设提升"], "真实主结果表与指标口径"),
 ("消融与专家分歧", "机制贡献和困难样本表现需单独量化。", ["消融：有/无图预测预训练与 RK4 步长，均按实际完成臂报告", "专家分歧阈值只在验证集确定；分歧分析来自初训协议", "消融与最终主表训练协议不同，不直接相减"], "真实消融表与分歧分组结果"),
 ("未来图属性预测", "未来图属性预测须与 persistence 和图 GRU 公平对照。", ["分别报告 1/3/5 秒 horizon 的 MAE 或预设误差", "过去观测与未来目标严格按切点分开", "只预测节点属性时，不把真实邻接矩阵称作预测"], "真实预测与目标叠图、误差表"),
 ("潜在轨迹与方向", "轨迹和箭头用于复查模型行为，不能当生理状态。", ["投影器仅在训练集拟合；标注样本与模型版本", "方向箭头优先沿实际观测轨迹绘制", "二维网格截面若使用，须固定上下文并声明构造"], "真实案例轨迹与投影方向"),
 ("速度、误差与扰动", "速度和残差可定位模型发生变化的时间与样本。", ["绘制 ||dz/dt||、图预测残差和分类概率", "轻噪声、单导联缺失、小扰动分别与原样本对照", "展示误判案例并解释条件、时间单位和局限"], "真实速度、误差、扰动案例"),
 ("动态脑电图解释边界", "脑电图边与注意力是统计关联代理，不能推断因果。", ["同一时间轴比较观测与预测节点属性", "标明相关边、参考电极、体积传导的混杂", "轨迹平滑不保证生理真实性；避免发作预警表述"], "真实图快照与图属性对照"),
 ("Spark 性能与资源", "性能结论必须标注数据规模与资源配置。", ["对比串行、多进程与 Spark 的吞吐、端到端耗时和开销", "保存 event log、任务数、输入字节和失败重试", "结果来自单物理机共享磁盘，不推断多机加速"], "真实吞吐曲线和资源表"),
 ("系统界面", "系统页面可交互，但 PDF 证据应为静态导出。", ["数据审计：类别、质量、划分与 provenance", "样本分析：波形、时频图、票数与预测", "动力学：轨迹、速度、未来属性和扰动对照"], "真实界面截图与静态导出"),
 ("完成情况与局限", "按已实现、已运行、已验证和受阻汇报项目状态。", ["区分功能代码、真实实验、正式结果和公开资源", "报告算力、数据可用性、迁移偏离与误判风险", "归纳研究问题的证据与尚未完成项"], "结果摘要与状态清单"),
 ("资源与复现", "每条资源链接都应指向真实可访问对象。", ["代码仓库、固定环境、训练命令与复现 smoke", "HMS 数据获取方法；权重链接、SHA256、model card", "公开 URL 未验证前只在草稿中写‘待发布’，正式稿不得保留"], "可访问链接与验证记录"),
 ("参考与补充", "区分官方 ODEBrain 论文、代码、HMS 任务与本项目实证。", ["HMS Harmful Brain Activity Classification，Kaggle", "ODEBrain，论文及官方代码；Spark ML 分类器文档", "torchdiffeq 数值求解库；本项目方法与结果另行标注"], "来源链接与许可证说明"),
]

NAVY = RGBColor(11, 29, 46)
NAVY2 = RGBColor(18, 44, 65)
TEAL = RGBColor(33, 197, 174)
WHITE = RGBColor(248, 251, 252)
MUTED = RGBColor(184, 203, 215)
AMBER = RGBColor(255, 194, 90)
LINK_BG = RGBColor(239, 247, 252)
LINK_INK = RGBColor(10, 60, 105)
FONT = "Microsoft YaHei"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def box(slide, x, y, w, h, fill, radius=True):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    return sh


def txt(slide, value, x, y, w, h, size=18, color=WHITE, bold=False,
        align=PP_ALIGN.LEFT, link=None):
    sh = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = sh.text_frame; tf.clear(); tf.word_wrap = True
    tf.margin_left = Inches(0.02); tf.margin_right = Inches(0.02)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = value; r.font.name = FONT
    r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    if link:
        r.hyperlink.address = link
        r.font.color.rgb = color
        r.font.underline = True
    return sh


def load_evidence(final, preview=False):
    if not final and not preview:
        return {}
    if final and preview:
        raise SystemExit("正式稿和本地预览不能同时选择。")
    evidence_path = PREVIEW_EVIDENCE if preview else EVIDENCE
    if not evidence_path.is_file():
        raise SystemExit("缺少结果证据清单：{}".format(evidence_path))
    raw = json.loads(evidence_path.read_text(encoding="utf-8"))
    expected_status = "local_preview" if preview else "formal"
    if raw.get("experiment_status") != expected_status or not raw.get("split_hash"):
        raise SystemExit("结果证据清单状态或统一 split_hash 不匹配。")
    if preview and (raw.get("formal") is not False or
                    raw.get("publication_status") != "pending_review"):
        raise SystemExit("本地预览须明确标记待发布审核且非正式交付。")
    evidence = {int(k): v for k, v in raw.get("slides", {}).items()}
    failures = []
    dependencies = raw.get("dependencies_sha256")
    if not isinstance(dependencies, dict) or not dependencies:
        failures.append("evidence:missing source dependency hashes")
    else:
        for source_name, expected_hash in dependencies.items():
            if not isinstance(source_name, str) or not isinstance(expected_hash, str):
                failures.append("evidence:invalid dependency entry")
                continue
            source_path = (ROOT / source_name).resolve()
            try:
                source_path.relative_to(ROOT.resolve())
            except ValueError:
                failures.append("evidence:dependency outside project")
                continue
            if not source_path.is_file() or sha256_file(source_path) != expected_hash:
                failures.append("evidence:dependency changed {}".format(source_name))
    if HEADLINE & EXPERIMENTAL:
        summary_name = raw.get("headline_refit_comparison")
        if raw.get("headline_training_protocol") != "train_val_fixed_epoch_refit" or not summary_name:
            failures.append("headline:missing train+val refit comparison")
        else:
            summary_path = (ROOT / summary_name).resolve()
            try:
                summary_path.relative_to(ROOT.resolve())
            except ValueError:
                failures.append("headline:comparison outside project")
            else:
                if not summary_path.is_file() or sha256_file(summary_path) != raw.get("headline_refit_comparison_sha256"):
                    failures.append("headline:comparison hash mismatch")
                else:
                    comparison = json.loads(summary_path.read_text(encoding="utf-8"))
                    if (comparison.get("schema_version") != "hms_final_refit_comparison_v1" or
                        comparison.get("experiment_status") != "formal" or
                        comparison.get("training_protocol") != "train_val_fixed_epoch_refit" or
                        comparison.get("split") != "test" or
                        comparison.get("split_hash") != raw["split_hash"] or
                        comparison.get("test_used_for_selection") is not False or
                        len(comparison.get("rows", [])) != 5):
                        failures.append("headline:invalid refit comparison")
                    for page in HEADLINE & EXPERIMENTAL:
                        row = evidence.get(page, {})
                        if (row.get("source_file") != summary_name or
                            row.get("source_sha256") != raw.get("headline_refit_comparison_sha256") or
                            row.get("headline_training_protocol") != "train_val_fixed_epoch_refit"):
                            failures.append("{}:headline provenance mismatch".format(page))
    for i in sorted(EXPERIMENTAL):
        row = evidence.get(i, {})
        for key in ("image", "split_hash", "n_patients", "n_windows", "model_version", "source_file",
                    "image_sha256", "source_sha256"):
            if not row.get(key): failures.append("{}:{}".format(i, key))
        if row.get("split_hash") != raw["split_hash"]:
            failures.append("{}:split_hash mismatch".format(i))
        if row.get("experiment_status") != expected_status or row.get("smoke_subset") is not False:
            failures.append("{}:evidence status differs".format(i))
        if preview and i == 23 and row.get("publication_status") != "pending_review":
            failures.append("23:publication status differs")
        if any(token in str(row.get("model_version", "")).lower() for token in ("draft", "smoke", "pending")):
            failures.append("{}:model_version is provisional".format(i))
        for count_key in ("n_patients", "n_windows"):
            try:
                if int(row.get(count_key, 0)) < 1:
                    failures.append("{}:{} must be positive".format(i, count_key))
            except (TypeError, ValueError):
                failures.append("{}:{} must be an integer".format(i, count_key))
        for key in ("image", "source_file"):
            path = row.get(key)
            if not path:
                continue
            source = (ROOT / path).resolve()
            try:
                source.relative_to(ROOT.resolve())
            except ValueError:
                failures.append("{}:{} outside project".format(i, key))
                continue
            if not source.is_file():
                failures.append("{}:{} missing {}".format(i, key, path))
                continue
            digest = sha256_file(source)
            digest_key = "image_sha256" if key == "image" else "source_sha256"
            if digest != row.get(digest_key):
                failures.append("{}:{} hash mismatch".format(i, key))
            if key == "source_file" and source.suffix.lower() == ".json":
                source_data = json.loads(source.read_text(encoding="utf-8"))
                if isinstance(source_data, dict) and (source_data.get("smoke_subset") is True or
                    "smoke" in str(source_data.get("experiment_status") or source_data.get("provenance") or "").lower()):
                    failures.append("{}:source is smoke evidence".format(i))
    if failures:
        raise SystemExit("正式稿证据不完整：" + ", ".join(failures))
    return evidence


def load_verified_resources(evidence):
    """Read the hashed page-23 source, never URLs printed inside a figure."""
    resource_evidence = evidence.get(23, {})
    source_name = resource_evidence.get("source_file")
    if not source_name:
        raise SystemExit("正式稿缺少第 23 页资源证据文件。")
    source = (ROOT / source_name).resolve()
    if not source.is_file() or sha256_file(source) != resource_evidence.get("source_sha256"):
        raise SystemExit("正式稿第 23 页资源证据哈希不匹配。")
    resources = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(resources, dict) or resources.get("experiment_status") != "formal":
        raise SystemExit("正式稿资源证据未标记 formal。")
    links = {}
    for key in ("repository", "weights"):
        row = resources.get(key, {})
        if not isinstance(row, dict):
            raise SystemExit("正式稿 {} 资源证据格式错误。".format(key))
        url = row.get("url")
        try:
            parsed = urlparse(url) if isinstance(url, str) else None
        except ValueError:
            parsed = None
        if (parsed is None or parsed.scheme != "https" or not parsed.hostname or
                row.get("verified_http_status") != 200 or not row.get("verified_at")):
            raise SystemExit("正式稿 {} 链接缺少有效 HTTPS 地址或 HTTP 200 验证记录。".format(key))
        links[key] = url
    return links


def add_flow(slide, index):
    labels = {
        3: ["数据审计", "Spark 特征", "模型训练", "严格评估", "静态展示"],
        6: ["Kaggle HMS", "Spark QC/特征", "Parquet", "Spark ML/PyTorch", "结果与界面"],
        11: ["过去 5 秒", "图初态", "ODE / GRU", "未来 1/3/5 秒", "图属性损失"],
        12: ["EEG / 图输入", "编码 z₀", "RK4 积分", "图预测 / 分类"],
    }.get(index)
    if not labels: return False
    for j, label in enumerate(labels):
        x = 6.65 + (j % 2) * 2.85
        y = 2.25 + (j // 2) * 1.02
        box(slide, x, y, 2.56, .68, NAVY)
        txt(slide, label, x+.12, y+.11, 2.32, .45, 16, WHITE, True, PP_ALIGN.CENTER)
        if j < len(labels)-1:
            txt(slide, "→", x+2.55, y+.13, .38, .4, 17, TEAL, True)
    return True


def build(final=False, preview=False):
    evidence = load_evidence(final, preview=preview)
    project_links = load_verified_resources(evidence) if final else {}
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    for i, (title, claim, bullets, expected) in enumerate(PAGES, 1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.background.fill.solid(); slide.background.fill.fore_color.rgb = NAVY
        box(slide, 0, 0, 13.333, .13, TEAL, False)
        txt(slide, "ODEBrain × Spark  |  HMS EEG", .55, .23, 7.6, .35, 12, MUTED, True)
        txt(slide, title, .55, .74, 12.2, .57, 29, WHITE, True)
        txt(slide, claim, .57, 1.39, 12.0, .59, 18, TEAL)
        box(slide, .55, 2.17, 5.65, 4.53, NAVY2)
        for j, bullet in enumerate(bullets):
            yy = 2.51 + j*1.31
            box(slide, .83, yy+.11, .09, .09, TEAL, False)
            txt(slide, bullet, 1.08, yy-.02, 4.82, 1.03, 17, WHITE)
        box(slide, 6.48, 2.17, 6.28, 4.53, NAVY2)
        txt(slide, "图证 / 页面要点", 6.76, 2.39, 5.70, .39, 14, MUTED, True)
        ev = evidence.get(i)
        if ev:
            img = ROOT / ev["image"]
            # Reserve room for the verified resource links on page 23.
            max_width, max_height = 5.75, (2.02 if i == 23 else 2.97)
            with Image.open(img) as source_image:
                image_width, image_height = source_image.size
            scale = min(max_width / image_width, max_height / image_height)
            picture_width, picture_height = image_width * scale, image_height * scale
            slide.shapes.add_picture(str(img), Inches(6.74 + (max_width - picture_width) / 2),
                                     Inches(2.95 + (max_height - picture_height) / 2),
                                     width=Inches(picture_width), height=Inches(picture_height))
            meta = "测试划分 {}  |  {} 患者 / {} 窗口  |  {}".format(
                ev["split_hash"], ev["n_patients"], ev["n_windows"], ev["model_version"])
            txt(slide, meta, 6.75, 6.03, 5.74, .45, 11, MUTED)
        elif not add_flow(slide, i):
            txt(slide, expected, 6.85, 3.30, 5.48, .75, 20, WHITE, True, PP_ALIGN.CENTER)
            if not final and not preview:
                txt(slide, "待真实结果 / 运行记录", 6.87, 4.17, 5.46, .48, 16, AMBER, False, PP_ALIGN.CENTER)
        if i == 23:
            box(slide, 6.76, 5.10 if final else 5.16,
                5.72, .89 if final else .62, LINK_BG)
            link_y = 5.17 if final else 5.28
            txt(slide, "HMS 数据页", 6.90, link_y, 2.25, .36,
                12, LINK_INK, True, link=SOURCES["hms"])
            txt(slide, "ODEBrain 官方代码", 9.27, link_y, 2.88, .36,
                12, LINK_INK, True, link=SOURCES["ode_code"])
            if final:
                txt(slide, "项目代码仓库", 6.90, 5.57, 2.25, .30,
                    12, LINK_INK, True, link=project_links["repository"])
                txt(slide, "模型权重下载", 9.27, 5.57, 2.88, .30,
                    12, LINK_INK, True, link=project_links["weights"])
        if i == 24:
            box(slide, 6.76, 4.74, 5.72, 1.77 if final else 1.28, LINK_BG)
            for j, (label, url) in enumerate([
                ("HMS 数据", SOURCES["hms"]), ("ODEBrain 论文", SOURCES["ode_paper"]),
                ("ODEBrain 代码", SOURCES["ode_code"]), ("Spark ML 文档", SOURCES["spark"]),
                ("torchdiffeq", SOURCES["ode_solver"])]):
                txt(slide, label, 6.90+(j%2)*2.77, 4.84+(j//2)*.37,
                    2.7, .32, 11, LINK_INK, True, link=url)
            if final:
                txt(slide, "项目代码仓库", 6.90, 6.08, 2.70, .32,
                    11, LINK_INK, True, link=project_links["repository"])
                txt(slide, "模型权重下载", 9.67, 6.08, 2.70, .32,
                    11, LINK_INK, True, link=project_links["weights"])
        if preview:
            txt(slide, "LOCAL PREVIEW · 待发布审核 · 非正式交付", 7.30, .28, 5.48, .43,
                12, AMBER, True, PP_ALIGN.RIGHT)
        elif not final:
            txt(slide, "DRAFT · 待实验 · 不作正式结果", 8.00, .28, 4.78, .43, 13, AMBER, True, PP_ALIGN.RIGHT)
        txt(slide, "{} / 24".format(i), 11.97, 7.04, .78, .25, 10, MUTED, align=PP_ALIGN.RIGHT)
    SLIDES.mkdir(exist_ok=True)
    # ASCII filenames work under this server's Python 3.6 ASCII filesystem locale.
    out = SLIDES / ("ODEBrain_final.pptx" if final else
                    "ODEBrain_LOCAL_PREVIEW_pending_publication.pptx" if preview else
                    "ODEBrain_DRAFT_pending_experiments.pptx")
    with out.open("wb") as stream:
        prs.save(stream)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--final", action="store_true", help="Require verified result evidence and public resources")
    mode.add_argument("--preview", action="store_true", help="Use verified results with publication-pending watermark")
    args = parser.parse_args()
    print(build(args.final, preview=args.preview))
