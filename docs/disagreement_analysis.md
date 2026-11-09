# 正式专家分歧与配对患者分析

完成 TCN、图 GRU、ODE 分类模型的正式训练及完整 `val`、`test` 评估后，在项目根目录运行：

```bash
LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-} \
  .venv/bin/python scripts/analyze_disagreement.py
```

默认读取 `artifacts/runs/tcn`、`artifacts/runs/graph_gru`、`artifacts/runs/ode_finetune`，以及对应 `results/<run 名>/` 下的正式 `val`、`test` 指标和预测。run 路径不同时使用 `--tcn-run`、`--gru-run`、`--ode-run`。默认输出 `results/disagreement_analysis.json`。`--n-boot`（默认 2000，至少 100）和 `--seed`（默认 42）控制 bootstrap。

脚本使用**完整验证集**的六类专家票数熵中位数预先固定阈值；测试集熵小于等于阈值为低分歧，高于阈值为高分歧。测试集不参与阈值选择。每个分层报告窗口数、患者数、窗口等权 KL/准确率、患者等权 KL，以及按患者重采样的 KL 和准确率 95% 置信区间。ODE 相对图 GRU、TCN 的配对差异也在完整测试集和两个分层中按患者重采样，差异方向为 ODE 减基线；KL 负差异表示 ODE 更好，准确率正差异表示 ODE 更好。

运行前逐项验证完整原始 metadata 与审计/划分、患者互斥和 split hash；每个 run 的正式配置、完成记录、checkpoint、源配置、冻结窗口清单、预训练来源及 `val`/`test` 指标；三个测试预测必须逐一覆盖相同的 held-out `window_id`，并与 manifest 的 `patient_id`、硬标签、六类专家软标签一致。正式指标的窗口 KL 还须与预测 CSV 重算一致。输入缺失、smoke/synthetic 标记、身份或哈希不符、分层患者少于两人时直接拒绝，**不生成正式结果**。输出记录 manifest、checkpoint、指标和预测文件的 SHA256，方便交付复核。
