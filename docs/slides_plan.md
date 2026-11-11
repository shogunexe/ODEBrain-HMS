# ODEBrain 演示稿与 PDF 交付说明

## 当前产物

- `slides/ODEBrain_DRAFT_pending_experiments.pptx`：24 页、16:9、可编辑中文文本和原生图形。每页均有醒目的 `DRAFT · 待实验 · 不作正式结果`。真实结果页仅有文字框位，**没有虚构指标、曲线、截图或模型输出**。
- `scripts/build_slides.py`：从固定的页面论点生成 PPTX；第 3、6、11、12 页的流程框图为原生 PowerPoint 形状，均可修改。
- `slides/ODEBrain_DRAFT_pending_experiments.pdf`：已通过项目内 LibreOffice 从上述草稿转换，PPTX/PDF/逐页 PNG 均核对为 24 页；该文件仍是 DRAFT，不能作为最终提交。正式实验结果包尚未齐备，因此未生成正式 PPTX/PDF。详见 [本地导出验证](pdf_export.md)。

运行：

```bash
python3 -m pip install -r requirements-slides.txt
python3 scripts/build_slides.py
```

## 24 页叙事与应补证据

| 页 | 内容 | 正式版须补的证据 |
|---|---|---|
| 1–3 | 题目、背景、研究闭环 | 完成状态核实；流程图可编辑 |
| 4–5 | HMS 数据、质量与划分 | 实际审计表、患者级划分清单及哈希 |
| 6–8 | 架构、部署、Spark | 运行配置、Spark 日志和处理统计 |
| 9–13 | 对照模型、ODEBrain 改造、训练协议 | 固定代码版本、配置和训练记录 |
| 14–16 | 分类、消融、图预测主结果 | 同一测试划分的指标与真实图 |
| 17–19 | 轨迹、速度、误差、图演化 | 真实模型导出的案例图，含时间单位 |
| 20–21 | Spark 性能、系统界面 | 真实吞吐数据、event log 和界面截图 |
| 22–23 | 成果、局限、资源访问 | 状态证据和可访问的真实仓库/权重链接 |
| 24 | 引用 | 官方资源链接与许可再次核对 |

## 正式稿门禁

### 证据准备器

正式实验完成后，先把真实产物的项目内路径写入 `results/formal_delivery_inputs.json`，再运行：

```bash
LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-} \
  .venv/bin/python scripts/prepare_formal_slides.py \
  --inputs results/formal_delivery_inputs.json
.venv/bin/python scripts/build_slides.py --final
```

输入文件结构：

```json
{
  "manifest": "data/curated/split_manifest.csv",
  "metadata": "data/raw/train.csv",
  "audit": "data/curated/audit.json",
  "split_summary": "data/curated/split_manifest.json",
  "feature_summary": "data/features/eeg_features_v1_summary.json",
  "spark_metrics": "results/spark_baselines/metrics.json",
  "refit_comparison": "results/refit/final_comparison.json",
  "benchmark_runs": "results/spark_benchmark/runs.json",
  "benchmark_summary": "results/spark_benchmark/summary.json",
  "ablation": "results/ablation.json",
  "forecast_comparison": "results/refit/forecast_comparison.json",
  "dynamics_index": "results/dynamics_refit/result_index.json",
  "ui_capture": "results/formal_ui_capture.json",
  "resources": "results/formal_resources.json"
}
```

公开仓库或权重链接尚待发布审核时，可用同一完整实验输入（省略 `resources`）生成**本地预览**。预览仍要求全部数据、模型、测试预测、基准、消融、未来图、最终 ODE 动力学和正式界面截图通过上述实验门禁；仅第 23 页改为“待发布审核”，并在 24 页逐页显示“LOCAL PREVIEW · 待发布审核 · 非正式交付”。产物文件名与正式稿分开，不能作为正式提交：

```bash
LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-} \
  .venv/bin/python scripts/prepare_formal_slides.py \
  --inputs results/formal_delivery_inputs.json --preview
.venv/bin/python scripts/build_slides.py --preview
scripts/export_slides_pdf.sh "$PWD/slides/ODEBrain_LOCAL_PREVIEW_pending_publication.pptx" "$PWD/slides"
```

`--final` 仍必须读取经 HTTP 200 验证的公开仓库和权重链接；预览证据清单不能用于 `--final`。

实际审计与特征汇总路径以运行命令输出为准。预测图缓存使用 `data/training/graph_forecast_v2` 的逐窗口 eligibility 与按 split 汇总；旧 v1 缓存不作正式证据。分类主表只读取 `results/refit/final_comparison.json`：五个深度模型均在 train+val 上按初训验证选出的固定轮数重训，随后对同一完整测试集评估。准备器调用 refit 汇总器重新核对源初训、最终 `final.pt`、逐窗口预测、指标、患者与窗口全集，并要求正式动力学索引的 checkpoint 哈希等于 refit ODE 的 `final.pt`。初训 `best.pt` 的测试指标不得混入分类主表。Spark LR/RF 仍为训练集基线；消融与图预测若来自初训，会在第 15–16 页明确标注。图预测比较须核对原分类测试窗口数、预测 eligible/excluded 数、原因计数、eligible ID 哈希和 v2 缓存索引哈希；`n_windows` 是预测 eligible 子集数量。任何嵌套的 smoke 或 synthetic 标记均使生成失败。第 4、5、7、8、9、13–23 页的 PNG 在 `results/figures/formal_slides/` 生成，最后才写 `results/slides_evidence.json`，记录原始输入与 PNG 的 SHA256。正式图全部基于实测输入；动力学图和界面图复制经过校验的 PNG。

以下四个补充输入尚需由正式实验与实际发布取得，不能由 smoke 案例拼凑：

```text
ablation.json:
  experiment_status="formal", split="test", split_hash=<manifest hash>,
  n_windows=<完整测试窗口>, n_patients=<测试患者>,
  rows=[{"name": <真实模型运行>, "kl_window_mean": <实测值>}, ...]
  至少两项，每项均须从独立运行的同一测试划分预测计算。

forecast_comparison.json:
  experiment_status="formal", split="test", split_hash=<manifest hash>,
  evaluation_scope="forecast_eligible_held_out_test",
  n_windows=<预测 eligible 测试窗口>, n_patients=<其中的测试患者>,
  forecast_eligibility={original_test_windows: <分类测试窗口>,
    eligible_test_windows: <预测 eligible 测试窗口>,
    excluded_test_windows: <预测排除窗口>, reason_counts: <排除原因统计>,
    eligible_window_ids_sha256: <eligible ID 哈希>,
    cache_index_sha256: <graph_forecast_v2/index.json 哈希>},
  rows=[{"name": <真实预测器>, "mae_by_horizon_s":
         {"1": <实测值>, "3": <实测值>, "5": <实测值>}}, ...]
  至少两项，包括真实 persistence/图 GRU 等对照；不得用案例误差冒充完整 eligible 测试集合，也不得把 eligible 子集写成全部分类测试窗口。

formal_ui_capture.json:
  experiment_status="formal", split_hash=<manifest hash>,
  image=<正式系统页面 PNG 的项目内路径>

formal_resources.json:
  experiment_status="formal",
  repository={"url": <公开 HTTPS 地址>, "verified_http_status": 200,
              "verified_at": <实际核对时间>},
  weights={"url": <公开 HTTPS 地址>, "verified_http_status": 200,
           "verified_at": <实际核对时间>}
```

准备器只校验这些补充 JSON 的结构、有限数值与哈希。消融和未来图预测指标仍须人工与原始预测逐项核对；资源的 HTTP 状态须由实际访问核实。准备器不运行实验、不下载数据、不导出 PDF。

正式稿生成命令为 `python3 scripts/build_slides.py --final`。该命令读取 `results/slides_evidence.json`：顶层须有 `experiment_status: "formal"`、统一的 `split_hash` 和经哈希绑定的 refit 主表；第 4、5、7、8、9、13–23 页逐页须有 `image`、`source_file`、对应的 `image_sha256` 与 `source_sha256`、同一个 `split_hash`、正整数 `n_patients`/`n_windows`、`model_version`、`experiment_status: "formal"` 和 `smoke_subset: false`。第 9、13、14、22 页还须绑定同一个 refit 主表及 `train_val_fixed_epoch_refit` 协议。文件须位于项目内且哈希匹配；JSON 来源文件若自报 smoke 状态会被拒绝。缺项时命令报错，不生成正式稿。

图应由正式评估文件导出，且图中必须显式标识单位、分组、模型、样本来源；单靠 JSON 中写路径不能证明图内容正确。正式版制作时还需人工核对结果与图源、划分一致性及公开链接。封面及方法页需按实际实现状态改写。任何未验证的数值和占位语不得进入正式 PDF。

## 导出与视觉检查

系统 PATH 中没有 LibreOffice，但已在 `artifacts/pdf-tools/` 安装本地 LibreOffice 7.6.7.2 和中文字体。DRAFT 已完成本地转换与 24 页渲染检查。正式实验与证据门禁全部通过后，在本机执行：

```bash
python3 scripts/build_slides.py --final
scripts/export_slides_pdf.sh "$PWD/slides/ODEBrain_final.pptx" "$PWD/slides"
```

也可以在 Windows PowerPoint 中打开正式 PPTX，用“导出 → PDF”。导出后逐页检查：24 页完整、中文字体和公式正常、图像不裁切、指标与图注可读、页码正确、URL 可点击；把 PDF 再与 PPTX 对照。若未通过检查，不应称作正式交付。

## 外部来源

2026-11-05 对以下链接执行 HTTP GET，均返回 200；这只验证链接可访问，不代表数据下载许可或论文结论已经由本项目实证：

- [Kaggle HMS 数据](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data)
- [ODEBrain 官方代码](https://github.com/HHJIAnmo/ODEBRAIN)
- [ODEBrain 论文](https://arxiv.org/abs/2602.23285)
- [Spark ML 分类文档](https://spark.apache.org/docs/latest/ml-classification-regression.html)
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq)

当前公开仓库和权重的地址尚未给定，草稿不展示虚构 URL。
