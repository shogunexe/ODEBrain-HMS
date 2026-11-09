# ODEBrainInspiredHMS 正式消融协议

入口：`scripts/run_ablations.py`；预注册配置：`configs/ablations.yaml`。当前计划固定 seed 42、同一患者划分及同一 HMS 分类输入，比较三条训练链：

| 名称 | 图预测预训练 | 分类微调 RK4 步长 | 回答的问题 |
|---|---:|---:|---|
| `pretrained_rk4_025` | 是 | 0.25 秒 | 参考配置 |
| `scratch_rk4_025` | 否 | 0.25 秒 | 预测预训练是否有帮助 |
| `pretrained_rk4_050` | 是，**同一预训练 checkpoint** | 0.50 秒 | 求解步长对结果与成本的影响 |

三条分类训练配置除初始化及声明的 RK4 步长外相同，均使用 `configs/models/odebrain_inspired_hms.yaml` 的 soft-label KL 训练目标、batch、早停和图缓存。预训练来自 `configs/models/odebrain_pretrain.yaml`，仅在训练患者做未来节点属性预测。不同步长可能改变训练轨迹，因此 `0.25` 与 `0.50` 的测试 KL 差异是**训练设置整体差异**，不能单独解释成求解器数值误差；数值收敛还需在相同权重上另做缩步验证。

## 执行

先完成全量 HMS 解压、质量审计、冻结 `data/curated/split_manifest.csv`，并构建正式分类及预测图缓存。项目 Python 3.8 环境需设置库路径：

```bash
cd /export/home/limianchong/CSproject
export LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-}
.venv/bin/python scripts/run_ablations.py train --device cuda --workers 4
.venv/bin/python scripts/run_ablations.py test --device cuda --workers 4
```

`train` 先训练一次图预测预训练模型，随后依次训练三条分类臂并只评估验证集。已有有效 `completed.json` 时验证配置、数据和 checkpoint 后跳过该训练；未完成但有 `latest.pt` 时通过现有 `src.cli train --resume` 恢复；已有 `best.pt` 但缺失 `latest.pt` 会拒绝继续。`test` 在配置和 checkpoint 全部核验后评估统一测试集，然后生成 `results/ablation.json`。若测试评估已完成，也可只运行 `collect` 对现有文件重新核验并汇总。

产生的模型与标准化参数位于 `artifacts/runs/ode_ablations_v1/`；每臂的验证/测试指标和预测位于 `results/<臂目录名>/`；自动生成的运行配置位于 `artifacts/ablations_v1/configs/`。训练计划或关键代码改变后，应新建 `run_root` 与 `generated_config_dir`，避免把不同实验混在同一目录。用 CPU 调试时可以传 `--device cpu`，但不使用 `--smoke`；完整数据门禁依然生效。

## 证据门禁

汇总器会核对：全量 `train.csv` 与已质量审计 manifest 的 EEG 覆盖；有效窗口的 train/val/test 划分；manifest 内容哈希；各 checkpoint 的配置、初始化 checkpoint SHA256、训练完成标记、训练历史和最佳轮次；分类臂完整测试集评估；每一条预测 CSV 的窗口 ID、患者 ID、硬标签及专家软标签与统一测试 manifest 完全一致；从 CSV 重新计算的窗口平均 KL 和患者等权 KL 与指标 JSON 一致；测试指标文件时间不早于最佳 checkpoint。配置记录训练 seed、计划 SHA256 以及模型、训练、数据、评估关键源码 SHA256。输出每行保留 checkpoint、配置、指标、预测文件和其 SHA256，便于 PPT 证据链追溯。

`results/ablation.json` 顶层含 `experiment_status=formal`、`split=test`、`split_hash`、`n_windows`、`n_patients` 和三行真实 `kl_window_mean`。它满足 `scripts/prepare_formal_slides.py` 对消融表的结构要求。脚本绝不会从 smoke 或不完整数据推算正式值；缺失文件时退出并保留原因。

## 当前状态

2026-11-07：脚本及四项聚合测试已通过项目 Python 3.8 环境；当前旧 manifest 仍记录大量 `missing_eeg_file`，实跑 `collect` 正确拒绝，提示 `Formal run requires every EEG parquet`。因此尚无正式 `results/ablation.json` 和可报告的消融指标。全量解压、重新审计及正式训练完成后按上述顺序运行。
