# Spark 与 ODEBrain-inspired HMS EEG 分析系统

课程项目：《基于 Spark 与 ODEBrain 连续动力学建模的 EEG 异常脑电活动分析与可视化系统》。当前目录是实施中的研究代码。**没有真实 HMS 实验结果时，不应把合成测试当成分类表现。**

## 当前状态（2026-11-08）

- 已实现 HMS 元数据审计、患者级划分、EEG 窗口与图特征、Spark 特征和 LR/RF 基线、四种深度对照、ODEBrain-inspired HMS、评估指标与可视化代码。
- 本机实测 2×V100S 32GB、40 CPU、251 GiB RAM；详见 [环境盘点](docs/environment.md)。不以方案中的 8×A100 预算制定实验结论。
- Kaggle 账号已接受[竞赛规则](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/rules)。19,762,787,900 字节的官方训练数据 ZIP 已下载完整并通过 `unzip -t` 全包校验；17,300 个 EEG 和 11,138 个频谱文件均已解压。全量质量审计覆盖 106,800 条标注、17,089 个被引用 EEG 文件，保留 106,587 个有效分类窗口；患者级 train/val/test 分别为 67,114/16,835/22,638 个窗口，split SHA256 为 `f334b0e31f5faf520f85a2d4c79631caf7edef2102460a8bcd629fc2a699a82f`。正式 Spark 特征与 TCN 训练正在运行；正式模型评估、图预测对照和最终交付结果尚未完成。预测任务专用 `graph_forecast_v2` 缓存正在构建，其 eligible 窗口数不能从分类有效窗口数直接推断；冻结的 36 文件、364 窗口子集仅用于 smoke。
- 模型名 **ODEBrainInspiredHMS**；这是独立重写的 HMS 改造版，不是完整官方代码移植。[审计与偏离](docs/odebrain_adaptation.md)。

## 数据与研究边界

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

`scripts/reproduce.sh smoke` 执行合成接口测试并在真实数据已到位时跑小样本闭环；`full` 运行正式阶段入口。合成 smoke 不输出正式研究指标。

## 结果与资源

正式结果写入 `results/`，模型及归一化参数写入 `artifacts/runs/`，静态图写入 `results/figures/`。每份权重须与配置、通道顺序、划分哈希和预处理版本一起使用。Spark 细节见 [流水线说明](docs/SPARK_PIPELINE.md)，执行进度见 [状态](docs/status.md)。

24 页[可编辑 DRAFT PPTX](slides/ODEBrain_DRAFT_pending_experiments.pptx) 已用项目内本地 LibreOffice 转成 [DRAFT PDF](slides/ODEBrain_DRAFT_pending_experiments.pdf)，两者均为 24 页；见 [导出验证](docs/pdf_export.md)。这不是正式实验结果或最终提交 PDF。GitHub 仓库、权重访问地址与最终 PDF 均待正式实验产物和发布授权完成后补充；不会保留示例 URL 作为正式资源。HMS 数据由 Kaggle 标注 CC BY-NC 4.0，下载和再发布应遵守其当时条款；ODEBrain 官方代码为 MIT，来源与项目改造分别记录。
