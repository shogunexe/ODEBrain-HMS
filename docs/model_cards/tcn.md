# tcn — HMS 模型卡

## 用途与身份

- 状态：正式完整数据评估；六类离线 EEG 活动分类。
- 模型版本：`8d79b69e91c704a12816dc81793ea77156673cfc3a8b6514673100ea0737a1d5`（checkpoint SHA256）。
- 输入来源：HMS 训练集；分类使用 50 秒上下文，专家标注对应中央 20–30 秒。
- 模型属于本项目基线，结果只对应此处固定患者划分。

## 输入与预处理

- 采样率：200 Hz；19 个 EEG 导联（不含 EKG）：Fp1, F3, C3, P3, F7, T3, T5, O1, Fz, Cz, Pz, Fp2, F4, C4, P4, F8, T4, T6, O2。
- 通道：原始 EEG=True；同窗 STFT 频谱=False；1 秒图帧=False。
- 以 `eeg_label_offset_seconds` 裁出 50 秒；短内部缺失最多插值 20 采样点，长缺失保留掩码并填零；超过 5% 缺失或恒定导联的窗口不进入有效集。
- 频谱分支：STFT 256 采样点窗、64 采样点步长，幅值 `log1p`。图分支：每导联 5 个频带的 1 秒节点属性，相关系数作为统计关联代理边；节点标准化参数只从训练患者拟合。
- 训练图归一化文件 SHA256：`c2f04c4f7e0e8536881229f8c751dc3a422003205aeabf35d53efe0e5e0dc5a3`。

## 数据与划分

- 患者级互斥划分哈希：`f334b0e31f5faf520f85a2d4c79631caf7edef2102460a8bcd629fc2a699a82f`。
- Manifest 文件 SHA256：`685b4ee306a37b73ce01a3c4671eeed11850c6c0ef2c6c11131c79646d7f09c0`。
- 训练：67114 窗口 / 1245 患者；验证：16835 窗口 / 311 患者；测试：22638 窗口 / 389 患者。
- 软标签由六类专家票数归一化；测试集按固定协议独立评估。

## 训练与评估

- 训练配置：`artifacts/runs/tcn/config.json`；配置 SHA256：`760abcacc0544ef27f8a512ecc22803237d7a4c18ac4ffb38ae68b23937f87be`。
- 训练参数：`{"batch_size": 16, "epochs": 30, "forecast_weight": 0.0, "grad_clip": 1.0, "lr": 0.001, "patience": 8, "weight_decay": 0.01}`。
- 主指标：KL(target || prediction)，逐窗口平均；患者等权 KL 与 macro F1 为补充。

| 划分 | 窗口 | 患者 | 窗口 KL ↓ | 患者等权 KL ↓ | Macro F1 |
|---|---:|---:|---:|---:|---:|
| val | 16835 | 311 | 1.048955 | 1.143651 | 0.499582 |
| test | 22638 | 389 | 1.196578 | 1.178658 | 0.457906 |

- 验证和测试的逐窗口预测 CSV、指标 JSON 均与冻结划分逐 ID 核对；各文件 SHA256 见同名 JSON 模型卡。

## 限制

- 只用于离线研究展示；不能用于临床诊断、实时发作预警或患者治疗。
- 专家票数存在分歧；脑电相关边与潜在 ODE 轨迹不构成生理因果解释。
- 指标只适用于当前 HMS 划分、预处理和文件哈希；跨中心/设备/人群泛化未经本模型卡验证。

## 本地复现

```bash
.venv/bin/python -m src.cli train --config configs/models/tcn.yaml --run-dir artifacts/runs/tcn
.venv/bin/python -m src.cli evaluate --run-dir artifacts/runs/tcn --split val
.venv/bin/python -m src.cli evaluate --run-dir artifacts/runs/tcn --split test
```

权重仅位于本地 `artifacts/runs/tcn/best.pt`；本脚本不发布权重或生成下载链接。
