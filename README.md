# Spark 与 ODEBrain-inspired HMS EEG 分析系统

课程项目：《基于 Spark 与 ODEBrain 连续动力学建模的 EEG 异常脑电活动分析与可视化系统》。当前目录是实施中的研究代码。**没有真实 HMS 实验结果时，不应把合成测试当成分类表现。**

## 当前状态（2026-11-12 13:35 Asia/Shanghai）

- 106,800 条 HMS 元数据已全量审计，保留 106,587 个有效分类窗口；患者互斥 train/val/test 为 67,114/16,835/22,638 个窗口。官方训练数据 ZIP 已完整下载、校验和解压。全量 Spark 特征、MLlib LR/RF、36 次性能基准、五模型最终 train+val 重训、42/44/45 三种子、预测任务、六个动力学案例及三条消融测试均已完成。
- 最终分类 seed 42 窗口等权 KL：时频融合 1.0530、TCN 1.2000、频谱 CNN 1.3228、GraphGRU 1.3611、ODEBrain-inspired HMS 1.6276。ODE 未在分类上领先；患者、EEG 记录等权及逐类指标见[正式结果](docs/formal_results.md)。消融是 train-only 加验证早停，与最终重训主表不能直接相减；测试集结果不用于配置选择。
- 24 页本地预览 [PPTX](slides/ODEBrain_LOCAL_PREVIEW_pending_publication.pptx) 与 [PDF](slides/ODEBrain_LOCAL_PREVIEW_pending_publication.pdf) 已生成并逐页渲染；正式浏览器截图已采集。公开仓库、Release 链接验证和最终版 PDF 正在收尾，完成前以预览文件的“待公开发布”标记为准。资源 ZIP 已通过全包与逐文件校验，并安装到干净克隆。全套代码测试 87 项通过。
- 模型名 **ODEBrainInspiredHMS**；这是独立重写的 HMS 改造版，不是完整官方代码移植。[审计与偏离](docs/odebrain_adaptation.md)。

## 数据与研究边界

Spark 是本项目的分布式计算与数据处理框架，并非分类模型或存储系统。特征表保存为 Parquet；逻辑回归（LR）和随机森林（RF）是使用 Spark MLlib 在这些特征上训练的传统分类模型。

使用 [Kaggle HMS 数据](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data)。训练元数据 `train.csv`、EEG `train_eegs/*.parquet` 和频谱 `train_spectrograms/*.parquet` 放在 `data/raw/`；原始数据不进入 Git。当前公平对照的频谱输入由同一 50 秒 EEG 派生，不使用更长的 Kaggle 频谱做主表比较。类序固定为 `seizure, lpd, gpd, lrda, grda, other`。每行偏移指定 50 秒上下文，只有中央 20–30 秒受该行标注。患者级 test 留出约 20%，开发集再分 train/val，固定 seed 42。

官方整包下载曾中断，已通过 `scripts/resume_hms_archive.py` 续传完成。完整性校验命令为 `unzip -t data/raw/hms-harmful-brain-activity-classification.zip`。`scripts/prepare_hms_archive.py` 仅解压训练文件并支持跳过已存在文件。原始数据、归一化参数和图缓存均不进入 Git。

本项目做离线分类和短时图节点属性预测。短时预测不代表发作前预警，也不代表临床诊断。图相关性只是功能关联代理；当前模型不预测边。

## 环境

目标 Python 3.8、Java 8、PySpark 3.4、PyTorch 2.0。服务器的 Python 3.8 需显式载入其共享库：

```bash
export LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-}
/export/usr/python/3.8.9/bin/python3.8 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Linux x86_64 / Python 3.8 环境可用 [已验证的精确版本清单](requirements-lock-py38-linux.txt) 替代上面的 `requirements.txt` 安装；2026-11-12 的干净克隆与新 venv 已通过 `pip check` 和 `scripts/reproduce.sh smoke`（86 passed、1 skipped）。`requirements.txt` 也显式限定了 Streamlit 的三个传递依赖版本，以兼容服务器自带的旧版 pip。

可用 GPU 通过 `CUDA_VISIBLE_DEVICES` 选择；训练先单卡验证，再按实际占用和任务预算扩展。对于其他服务器，使用自身 Python 3.8+ 创建 venv 即可。

## 可复现命令

以下命令从项目根目录执行。运行 `audit --quality` 会逐文件检查窗口边界、EEG 通道和缺失；此操作需要完整原始 EEG 文件。

```bash
export LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-}
export PYSPARK_PYTHON="$PWD/.venv/bin/python"
.venv/bin/python -m src.cli doctor
.venv/bin/python -m src.cli audit --config configs/data.yaml --quality --workers 8
.venv/bin/python -m src.cli split --config configs/split.yaml
.venv/bin/python -m src.data.graph_cache build --manifest data/curated/split_manifest.csv --output data/training/graph_classification --task classification --workers 8
.venv/bin/python -m src.data.graph_cache build --manifest data/curated/split_manifest.csv --output data/training/graph_forecast_v2 --task forecast --cut 5 --horizons 1,3,5 --workers 4
.venv/bin/spark-submit scripts/build_features.py --config configs/spark.yaml --overwrite
.venv/bin/spark-submit scripts/train_spark_baselines.py --features data/features/eeg_features_v1 --output results/spark_baselines
.venv/bin/python -m src.cli train --config configs/models/odebrain_pretrain.yaml --run-dir artifacts/runs/ode_pretrain
.venv/bin/python -m src.cli train --config configs/models/odebrain_inspired_hms.yaml --init-checkpoint artifacts/runs/ode_pretrain/best.pt --run-dir artifacts/runs/ode_finetune
.venv/bin/python -m src.cli evaluate --run-dir artifacts/runs/ode_finetune --split val
.venv/bin/python -m src.cli evaluate --run-dir artifacts/runs/ode_finetune --split test
.venv/bin/python scripts/export_dynamics.py --help
.venv/bin/streamlit run app/main.py
```

`test` 只在配置冻结后运行一次；不得据此反选超参。其他对照使用 `configs/models/` 的相应 YAML。图预测阶段只输入切点前的 EEG/图，未来 1/3/5 秒节点频带属性仅作监督；分类阶段可以读取完整 50 秒，属于离线上下文分类。训练缓存、checkpoint、原始数据在 `.gitignore` 内。

图预测使用独立的 [窗口资格协议](docs/forecast_eligibility.md)：`graph_forecast_v2/index.json` 对每个分类有效窗口记录 `eligibility_reason`，并按 split 汇总 eligible/excluded 数量及 eligible ID 哈希。历史前 5 秒或未来目标因常量通道、缺失超限、无有效图节点而不合格的窗口仅从预测任务排除；分类 manifest 与患者划分不变。正式预测结果只能称为“测试集 eligible 子集”，须同时报告原分类测试窗口数与预测 eligible 数。

未指定 `--smoke` 的训练与评估会检查完整元数据覆盖、每个 EEG 文件已到位和质量审计字段；只有完整患者划分与训练产物通过校验才会生成正式结果。

`scripts/reproduce.sh smoke` 执行合成接口测试并在真实数据已到位时跑小样本闭环；`full` 从原始数据运行初训、Spark 和最终本地交付链，种子重复默认使用 GPU 0，可用 `FULL_SEED_GPU` 指定。合成 smoke 不输出正式研究指标。

初训、Spark 基线与性能实验完成后，运行 `scripts/reproduce_final_local.sh` 可按依赖顺序复核或恢复七个 train+val 重训、三条消融、42/44/45 种子汇总、全部补充指标、模型卡、动力学案例，并生成正式界面截图与带“待发布”标识的本地预览 PPTX/PDF。缺少 seed 44/45 的机器可运行 `scripts/reproduce_final_local.sh --run-seeds 0`，其中 `0` 是可用 GPU 编号；也可先分别运行 `scripts/run_seed_stability.sh 44 0` 和 `scripts/run_seed_stability.sh 45 0`。用 `FINAL_DEVICE=cpu`、`FINAL_WORKERS=0` 可在无 GPU 环境执行其余步骤，但训练时间会显著增加。该入口要求完整 Kaggle 数据与初训结果，不发布仓库或权重；`scripts/export_slides_pdf.sh` 还需按 [导出验证](docs/pdf_export.md) 安装本地 LibreOffice 与中文字体。

## 结果与资源

正式结果写入 `results/`，初训模型及归一化参数写入 `artifacts/runs/`，完整开发集重训写入 `artifacts/refit/`。静态图写入 `results/figures/`。每份权重须与配置、通道顺序、划分哈希和预处理版本一起使用。最终五模型主表是 [完整开发集重训结果](results/refit/final_comparison.json)；初训测试结果仅作探索性对照。最终模型卡见 `docs/model_cards_refit/`，初训模型卡见 `docs/model_cards/`。Spark 细节见 [流水线说明](docs/SPARK_PIPELINE.md)，执行进度见 [状态](docs/status.md)。

分类按冻结测试预测分别报告窗口平均、患者等权和 EEG 记录 `eeg_id` 等权 KL；七模型的三种加权混淆矩阵、逐类 precision/recall/F1、OVR AUC 与误报率在 [辅助指标](results/classification_weighted_metrics.json)，定义和复算命令见 [口径说明](docs/classification_weighted_metrics.md)。[最终模型配对区间](results/refit/paired_kl_intervals.json)按患者成组重采样，只描述预先固定配置的表现，不用于从测试集选择配置。图预测 MAE 是独立辅助任务；标准化 MAE/MSE 和逐频带误差不是 Kaggle 六分类评分要求。初训消融、初训分歧分析与最终重训主表的测试数值不能直接相减。

24 页[本地预览 PPTX](slides/ODEBrain_LOCAL_PREVIEW_pending_publication.pptx) 与[本地预览 PDF](slides/ODEBrain_LOCAL_PREVIEW_pending_publication.pdf) 已由真实实验结果生成，正式截图、三臂消融和三种等权指标均通过证据门禁。预览版明确标有“待公开发布”，不能称为最终 PDF；公开仓库和 Release 的匿名 URL 验证后再生成最终版。旧 DRAFT 仅保留作为过程记录，见 [导出验证](docs/pdf_export.md)。HMS 数据由 Kaggle 标注 CC BY-NC 4.0；来源与再分发边界见 [公开资源说明](docs/public_resources.md)。
