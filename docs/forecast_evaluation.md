# 图属性预测评估协议

本评估比较 **ODEBrainInspiredHMS、GraphGRU 和 persistence** 对相同未来电极节点属性的预测。属性是每个 1 秒图帧的五个 `log1p` 频带功率，不包括邻接矩阵。三个方法使用同一患者划分、同一过去 5 秒图输入、同一开发数据拟合的图属性均值与标准差，以及切点后第 1、3、5 秒的同一目标。初训协议的开发数据仅为 `train`；固定轮数 refit 协议为 `train+val`。

## 正式运行

先构建 `data/training/graph_forecast_v2`，等待 `index.json` 含完整的 `forecast_eligibility` 汇总；不能使用旧 `graph_forecast` v1 缓存。分类审计覆盖完整 50 秒，预测只使用前 5 秒历史及未来 1/3/5 秒目标，因此某些分类有效窗口会因历史或目标常量通道、缺失超限或无有效图节点而被排除。v2 缓存逐窗口记录 `eligibility_reason`，并按 split 保存 eligible/excluded 数量与 eligible 窗口 ID 哈希；详见[资格协议](forecast_eligibility.md)。

再用 `configs/models/odebrain_pretrain.yaml` 和 `configs/models/graph_gru_pretrain.yaml` 分别训练预测 eligible 集合上的完整正式 checkpoint；训练命令不加 `--smoke`。ODE 也可以使用从完整预测预训练 checkpoint 初始化的正式微调 checkpoint，评估脚本会核对其来源。两个 run 必须使用同一 split manifest 和同一 eligible 训练窗口集合。

```bash
LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib .venv/bin/python -m src.cli train \
  --config configs/models/odebrain_pretrain.yaml \
  --run-dir artifacts/runs/ode_pretrain_full

LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib .venv/bin/python -m src.cli train \
  --config configs/models/graph_gru_pretrain.yaml \
  --run-dir artifacts/runs/graph_gru_pretrain_full

LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib .venv/bin/python scripts/evaluate_forecast.py \
  --ode-run-dir artifacts/runs/ode_pretrain_full \
  --gru-run-dir artifacts/runs/graph_gru_pretrain_full \
  --manifest data/curated/split_manifest.csv \
  --graph-cache-dir data/training/graph_forecast_v2 \
  --output results/forecast_comparison.json
```

正式模式只评估完整、已质量审核 manifest 的 `test` 患者中**预测 eligible** 的窗口。脚本核对原始数据覆盖、患者互斥、manifest SHA256、checkpoint 内部配置、训练窗口快照、归一化参数、预测时间，以及 v2 缓存的原测试窗口数、eligible/excluded 数、原因计数和 eligible ID 哈希。任一前提不满足即报错，不写正式 JSON。`--graph-cache-dir` 可省略，此时读取 ODE run 配置中的缓存路径；正式模式若两处都没有 v2 资格缓存则报错，不能直接退回逐窗口原始 EEG 计算。

## 固定轮数 train+val refit 对照

两个预测预训练 run 按[最终 refit 协议](refit_final.md)完成后，可比较其冻结的 `final.pt`。该模式不重新选择轮数或 checkpoint，也不读取分类 refit 权重；分类任务权重会被拒绝。评估器逐个调用 refit 完成校验，核对验证集选定轮数、原始 `best.pt`、最终 `final.pt`、完成记录、训练快照和开发集拟合的归一化。ODE 与 GraphGRU 必须使用完全相同的预测合格 `train+val` 窗口、标准化参数和 v2 缓存。输出与初训结果分开保存；两者测试范围和 MAE 定义相同，但训练协议不同。

```bash
LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib .venv/bin/python scripts/evaluate_forecast.py \
  --protocol refit \
  --ode-run-dir artifacts/refit/ode_pretrain \
  --gru-run-dir artifacts/refit/graph_gru_pretrain \
  --manifest data/curated/split_manifest.csv \
  --graph-cache-dir data/training/graph_forecast_v2 \
  --output results/refit/forecast_comparison.json
```

省略 `--output` 时，refit 模式默认使用 `results/refit/forecast_comparison.json`。JSON 的 `training_protocol=train_val_fixed_epoch_refit`；每个模型的 provenance 记录固定轮数、原始验证集最佳 checkpoint 的 SHA256、refit 快照/完成记录及最终 checkpoint 的 SHA256。该模式仅支持正式 `test` 预测合格子集，不能与 `--smoke` 合用。

## 小样本接口检查

小样本训练需给两种模型都加 `--smoke`。评估时显式加 `--smoke`，只读取两个 checkpoint 冻结且完全相同的验证窗口，输出 `forecast_comparison_smoke.json`。这个文件是接口检查产物，不能作为正式测试结果。

```bash
LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib .venv/bin/python scripts/evaluate_forecast.py \
  --ode-run-dir artifacts/runs/ode_pretrain_strict_smoke \
  --gru-run-dir artifacts/runs/graph_gru_pretrain_strict_smoke \
  --smoke --output results/forecast_comparison_smoke.json
```

## 因果与统计口径

数据集分别清洗切点前历史与切点后目标；插值不会跨切点。评估器只把 `graph_nodes`、`graph_adj`、`graph_mask`、**过去 5 秒 EEG** 和预测时间交给模型。未来图属性、mask、标签和患者 ID 均留在模型调用之外。历史 EEG 长度与图帧数必须匹配切点；评估器再检查输出的预测时间。

Persistence 对每个节点取切点前最后一个有效图属性，并复制到三个时间距。三种方法仅在“未来节点有效且该节点在过去至少有效一次”的位置计算误差。先使用同一训练集标准化输入，计算误差时乘回相同标准差，因此 JSON 中的 `mae` 单位为原始 `log1p` 频带功率。每个时间距汇报全体有效节点特征值的 MAE、按患者先平均再对患者等权平均的 `patient_macro_mae`、有效值个数和有效患者数。重叠窗口仍有依赖性；这些 MAE 不附带把窗口当独立样本的置信区间。

`forecast_comparison.json` 保存 `experiment_status=formal`、`smoke_subset=false`、`evaluation_scope=forecast_eligible_held_out_test`、eligible 子集的 `n_windows`/`n_patients`、正式幻灯片证据门槛所需的 `rows[].mae_by_horizon_s`，以及更详细的逐方法结果。`forecast_eligibility` 同时保存原分类测试窗口数、eligible/excluded 数、原因计数、eligible ID 哈希及缓存索引哈希，避免把预测 eligible 子集误写成完整分类测试集。文件还保存 split hash、manifest SHA256、测试患者与窗口 ID 集合哈希、模型 checkpoint 与配置哈希、训练来源和归一化文件哈希，不公开原始患者 ID。初训正式对照已生成于 `results/forecast_comparison.json`；最终 train+val 重训权重对照已生成于 `results/refit/forecast_comparison.json`，两者均覆盖 22,606 个预测合格测试窗口。现有输出只有反标准化后的总体 MAE；原方案要求的标准化 MAE/MSE 和逐频带误差尚待补充，不得称预测指标协议全部完成。
