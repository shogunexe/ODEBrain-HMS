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

**正式报告主模型表应使用这五个 refit 的测试结果。** 初训 `val` 指标只用于固定轮数和超参数，不充当最终测试效果；初训 `test` 结果若保留，必须单独命名为探索性对照。现有 `scripts/write_model_cards.py` 与 `scripts/prepare_formal_slides.py` 的门槛仍绑定初训 `best.pt` 和初训 test 文件，尚不支持 refit `final.pt`。因此在这些门槛完成专门适配和重新核对前，不能将 refit 结果直接放入现有模型卡或正式 PPT，也不能在同一张主表混用初训和 refit 行。`final_comparison.json` 是独立、可追溯的 refit 汇总，供后续适配使用。
