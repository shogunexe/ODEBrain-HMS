# 单划分种子稳定性实验

原方案优先要求患者分组三折验证；本机实际只有两张 V100S，已将正式协议降为固定患者单划分加初始化种子重复。这里的标准差只反映同一患者划分上的训练随机性，**不是跨患者划分的方差**。主实验 seed 42，额外重复 seed 43 和 44；三组均使用初训验证集在 seed 42 冻结的配置与最佳轮数，在完整 `train+val` 上从头训练，再分别评估相同的 22,638 个留出测试窗口。图模型每个种子分别重新做预测预训练，再将该种子的预训练权重用于图分类。

主实验全部完成后，分别在空闲 GPU 上运行：

```bash
scripts/run_seed_stability.sh 43 0
scripts/run_seed_stability.sh 44 1
.venv/bin/python scripts/aggregate_seed_stability.py
```

脚本对每个 run 保存配置、固定训练轮数来源、完整开发集窗口清单、最终 checkpoint 和测试预测。每个种子的 `final_comparison.json` 由 `scripts/refit_final.py summarize` 重新核对五模型及全测试窗口。汇总脚本再次核对全部源 run、权重、预测、指标、划分和固定轮数，输出五个模型在三个种子下的 KL、患者等权 KL 和 Macro-F1 的均值及样本标准差。完整输出位于 `results/seed_stability/summary.json`。实验未运行完时，不在报告中填写任何种子均值或标准差。

资源占用较大；以主实验和正式 PPT/PDF 完成为先。如果截止时间前不能完成全部三个种子，需在最终材料中明确写成单划分单种子，而不能把 bootstrap 置信区间误称为种子重复。
