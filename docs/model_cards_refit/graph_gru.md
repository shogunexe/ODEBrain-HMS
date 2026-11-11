# graph_gru — HMS 最终重训模型卡

## 用途与身份

- 状态：正式完整数据评估；六类离线 EEG 活动分类。
- 模型版本：`a2b5311e2d47c633df1408352039a184e38a70349375746d38f7e5be22d4983e`（最终 checkpoint SHA256）。
- 输入来源：HMS 训练集；分类使用 50 秒上下文，专家标注对应中央 20–30 秒。
- 模型属于本项目基线，结果只对应此处固定患者划分。

## 数据与划分

- 患者级互斥划分哈希：`f334b0e31f5faf520f85a2d4c79631caf7edef2102460a8bcd629fc2a699a82f`。
- Manifest 文件 SHA256：`685b4ee306a37b73ce01a3c4671eeed11850c6c0ef2c6c11131c79646d7f09c0`。
- 初训训练：67114 窗口 / 1245 患者；初训验证：16835 窗口 / 311 患者；最终重训合计：83949 窗口 / 1556 患者；独立测试：22638 窗口 / 389 患者。
- 软标签由六类专家票数归一化；测试患者未进入轮数选择或最终重训。

## 输入与预处理

- 采样率：200 Hz；19 个 EEG 导联（不含 EKG）：Fp1, F3, C3, P3, F7, T3, T5, O1, Fz, Cz, Pz, Fp2, F4, C4, P4, F8, T4, T6, O2。
- 通道：原始 EEG=False；同窗 STFT 频谱=False；1 秒图帧=True。
- 以 `eeg_label_offset_seconds` 裁出 50 秒；短内部缺失最多插值 20 采样点，长缺失保留掩码并填零；超过 5% 缺失或恒定导联的窗口不进入有效集。
- 频谱分支：STFT 256 采样点窗、64 采样点步长，幅值 `log1p`。图分支：每导联 5 个频带的 1 秒节点属性，相关系数作为统计关联代理边。
- 最终图归一化参数仅从训练+验证患者拟合；文件 SHA256：`f99155a7c6abbfd82610b0b27c7be872a470c1a1454d0ae2225af0b3c16cf265`。

## 训练与评估

- 初训从验证集选择第 8 轮（从 1 计数），依据 `kl`；源 checkpoint SHA256：`d2030b322304f199c472a045b0225fefdfbcc2db9b6e67e7ada82222b69fdd2f`。
- 最终模型在训练+验证患者上固定训练 8 轮；不在重训阶段重新选择轮数。
- 重训随机种子：42；配置：`artifacts/refit/graph_gru/config.json`；配置 SHA256：`f0e277c1f795dd1f08e2718631e6b8518d997fc43a60f067e9c6bacb929314a4`。
- 训练参数：`{"batch_size": 8, "classification_weight": 1.0, "epochs": 8, "forecast_weight": 0.0, "grad_clip": 1.0, "lr": 0.001, "patience": 8, "weight_decay": 0.01}`。
- 主指标：KL(target || prediction)，逐窗口平均；患者等权 KL 与 macro F1 为补充。
- 初始化权重来自同样以训练+验证患者重训的预测任务：`artifacts/refit/graph_gru_pretrain/final.pt`；SHA256：`bd2030b7cc063ac24c1e88377e5b1363c9b6801113195c37de80f9ccfcf64ef5`。

| 划分 | 窗口 | 患者 | 窗口 KL ↓ | 患者等权 KL ↓ | Macro F1 |
|---|---:|---:|---:|---:|---:|
| test | 22638 | 389 | 1.361070 | 1.437356 | 0.405649 |

- 测试预测逐窗口覆盖冻结测试集；指标与预测已由 `scripts/refit_final.py summarize` 重算核对。
- 五模型正式对比表：`results/refit/final_comparison.json`；SHA256：`4f26fa1c151b74b06813f69ee3a696a3d07ea6d25089b5ee76ff5b4accb6d64d`。
- 结果只对应此处固定患者划分、预处理和文件哈希；跨中心、设备、人群泛化未经验证。

## 限制

- 只用于离线研究展示；不能用于临床诊断、实时发作预警或患者治疗。
- 专家票数存在分歧；脑电相关边与潜在 ODE 轨迹不构成生理因果解释。

## 本地复现

```bash
.venv/bin/python scripts/refit_final.py train --source-run artifacts/runs/graph_gru --run-dir artifacts/refit/reproduce_graph_gru --init-refit-run artifacts/refit/graph_gru_pretrain
.venv/bin/python scripts/refit_final.py evaluate --run-dir artifacts/refit/reproduce_graph_gru
```

权重仅位于本地 `artifacts/refit/graph_gru/final.pt`；本脚本不发布权重或生成下载链接。
