# 最终模型在完整开发集重训

`scripts/refit_final.py` 把初训验证集选出的最佳轮次和配置冻结，然后在原 `train+val` 患者上重新训练同样轮数。重训期间不运行验证或测试评估；测试集仅在 `evaluate` 子命令中读取。该脚本是独立入口，`python -m src.cli train/evaluate` 的默认行为不变。

## 前提与顺序

必须先有完整 HMS 原始 EEG、质量审核和患者划分，并完成七个非 smoke 初训 run：ODE 和 GraphGRU 的预测预训练、五个分类模型的初训。每个初训 run 需要 `config.json`、`best.pt`、`window_snapshot.json`、`history.jsonl` 和 `completed.json`。refit 检查文件哈希、患者互斥、初训 train/val 窗口覆盖完整开发集，以及最佳 checkpoint 的验证数值。缺任一文件或全量数据不齐时停止。

按下列顺序运行，目录名应替换成实际已完成的初训 run。两个图预测 refit 先从随机初始化开始，并在 `train+val` 拟合图属性标准化。ODE/GraphGRU 分类 refit 再从对应的开发集 refit 预测 checkpoint 初始化，沿用其标准化。TCN、频谱 CNN、融合模型从随机初始化开始；不会直接继续优化初训最佳分类 checkpoint。

```bash
# 先查看冻结轮数、窗口数、参数量与预计更新次数；不启动训练。
env LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-} \
  .venv/bin/python scripts/refit_final.py plan --source-run artifacts/runs/ode_pretrain

# 1. 两个图预测预训练 refit。
env LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-} \
  .venv/bin/python scripts/refit_final.py train --source-run artifacts/runs/ode_pretrain \
  --run-dir artifacts/refit/ode_pretrain --device cuda --workers 4
env LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-} \
  .venv/bin/python scripts/refit_final.py train --source-run artifacts/runs/graph_gru_pretrain \
  --run-dir artifacts/refit/graph_gru_pretrain --device cuda --workers 4

# 2. 三个非图分类模型 refit。
for model in tcn spectrogram_cnn fusion; do
  env LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-} \
    .venv/bin/python scripts/refit_final.py train --source-run artifacts/runs/${model} \
    --run-dir artifacts/refit/${model} --device cuda --workers 4
done

# 3. 两个图分类模型 refit；来源分类初训须本来就用相应预测预训练。
env LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-} \
  .venv/bin/python scripts/refit_final.py train --source-run artifacts/runs/ode_finetune \
  --init-refit-run artifacts/refit/ode_pretrain \
  --run-dir artifacts/refit/odebrain_inspired_hms --device cuda --workers 4
env LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-} \
  .venv/bin/python scripts/refit_final.py train --source-run artifacts/runs/graph_gru \
  --init-refit-run artifacts/refit/graph_gru_pretrain \
  --run-dir artifacts/refit/graph_gru --device cuda --workers 4
```

若初训某个图分类模型确实从随机初始化开始，则其分类 refit 也不传 `--init-refit-run`；脚本核对初始化策略一致。中断后在同一目录和相同参数重新执行 `train ... --resume`，从 `latest.pt` 恢复模型、优化器和 RNG。最后固定轮次的 checkpoint 是 `final.pt`；它不是在重训验证集上选择的“最佳”模型。每个 run 分开保存 `config.json`、`window_snapshot.json`、`validation_record.json`、`normalizer_record.json`、`graph_normalizer.npz`、`history.jsonl`、`latest.pt`、`final.pt` 和 `completed.json`。

## 计算成本与时间安排

对每个 run，固定轮数为初训 `best.pt` 的零基 `epoch + 1`。`plan` 输出开发集窗口数、批大小、参数量、每轮与总更新次数；后者是 `ceil((train_windows+val_windows)/batch_size) × 固定轮数`。先从已完成初训的实际每轮耗时估计服务器占用，也可把实测值传给 `plan --measured-seconds-per-epoch <秒>` 得到投影小时数。没有实测时间时不填 GPU 小时。完整流程共七次 refit：两次预测预训练、五次分类；ODE 的 RK4 积分需单独实测。不同 run 可在空闲 GPU 上并行，但每个 run 仍保持自身固定预算。

## 冻结测试与正式报告

五个分类 refit 完成后，再各执行一次测试评估；预测预训练 refit 不做六分类测试：

```bash
for model in tcn spectrogram_cnn fusion graph_gru odebrain_inspired_hms; do
  env LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-} \
    .venv/bin/python scripts/refit_final.py evaluate --run-dir artifacts/refit/${model} \
    --device cuda --workers 4
done

env LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-} \
  .venv/bin/python scripts/refit_final.py summarize \
  --run tcn=artifacts/refit/tcn \
  --run spectrogram_cnn=artifacts/refit/spectrogram_cnn \
  --run fusion=artifacts/refit/fusion \
  --run graph_gru=artifacts/refit/graph_gru \
  --run odebrain_inspired_hms=artifacts/refit/odebrain_inspired_hms \
  --output results/refit/final_comparison.json
```

测试结果写入 `results/refit/<run目录名>/test_metrics.json` 和 `test_predictions.csv`。`summarize` 会重新核对五个 refit 的源 run、最终 checkpoint、split hash、完整测试窗口/患者集以及逐窗口概率文件，才生成统一主表。

**正式报告主模型表应使用这五个 refit 的测试结果。** 初训 `val` 指标只用于固定轮数和超参数，不充当最终测试效果；初训 `test` 结果若保留，必须单独命名为探索性对照。`scripts/write_model_cards.py --protocol refit` 独立重核 `final_comparison.json`，读取 `final.pt` 并把正式卡写入 `docs/model_cards_refit/`；初训卡仍位于 `docs/model_cards/`。PPT 的正式主表也必须使用同一 refit 汇总，不能混用初训和 refit 行。

### 最终模型间的配对差异区间

五个模型配置和重训轮数已由开发集固定后，可只读冻结的 `test_predictions.csv` 与清单生成描述性区间：

```bash
env LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-} \
  .venv/bin/python scripts/paired_refit_ci.py \
  --manifest data/curated/split_manifest.csv \
  --comparison results/refit/final_comparison.json \
  --results-dir results/refit \
  --output results/refit/paired_kl_intervals.json \
  --n-boot 5000 --seed 42
```

脚本逐模型复核预测 CSV 的 SHA-256、完整测试窗口、患者、EEG 记录、硬标签、专家软标签、概率和主表 KL；`eeg_id` 若未写入旧预测 CSV，则用 `window_id` 从冻结清单连接。输出分别报告 **患者等权**、**EEG 记录等权**、**窗口等权** 的 ODE 减去各基线模型 KL 差。三个点估计的平均单位不同；95% 百分位区间都以患者为成对重采样单位，并在每次抽样中保留该患者的全部 EEG 记录和窗口。正差值表示 ODE 的 KL 较高。该区间只用于最终模型结果的描述和不确定性表达，不能按测试区间反过来挑选配置，也不能与初训或消融的不同协议结果直接相减。最终模型的区间不能代替独立患者划分或多随机种子重复。

## 最终 ODE 重训模型的动力学图

ODE 分类 refit 和 ODE 预测预训练 refit 均完成后，可从最终分类 `final.pt` 导出动力学案例：

```bash
env LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-} \
  .venv/bin/python scripts/export_dynamics.py \
  --checkpoint artifacts/refit/odebrain_inspired_hms/final.pt \
  --manifest data/curated/split_manifest.csv \
  --output results/dynamics_refit --device cuda
```

导出器会复核分类 refit 的完成记录、固定训练轮数和最终 checkpoint，沿 `refit.init_refit_run` 复核对应的开发集预测预训练 refit，并检查来源 checkpoint 的 SHA-256、同一患者划分、正的预测损失权重及预测头权重一致。来源标准化也由 refit 完成记录核对。任何来源缺失或权重变化都会拒绝导出。图中的 PCA 投影仍只用原 `train` 窗口拟合，案例来自预测任务合格的 `val/test` 窗口；导出的 `training_lineage` 标记 `train_val_fixed_epoch_refit`，并记录两个 checkpoint 的哈希。
