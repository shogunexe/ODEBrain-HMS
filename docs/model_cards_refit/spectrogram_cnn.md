# spectrogram_cnn — HMS 最终重训模型卡

## 用途与身份

- 状态：正式完整数据评估；六类离线 EEG 活动分类。
- 模型版本：`d9f089eff0d457f6e373dc6ddd3a130a5656d8f5edf5a8f19b90467b29050c57`（最终 checkpoint SHA256）。
- 输入来源：HMS 训练集；分类使用 50 秒上下文，专家标注对应中央 20–30 秒。
- 模型属于本项目基线，结果只对应此处固定患者划分。

## 数据与划分

- 患者级互斥划分哈希：`f334b0e31f5faf520f85a2d4c79631caf7edef2102460a8bcd629fc2a699a82f`。
- Manifest 文件 SHA256：`685b4ee306a37b73ce01a3c4671eeed11850c6c0ef2c6c11131c79646d7f09c0`。
- 初训训练：67114 窗口 / 1245 患者；初训验证：16835 窗口 / 311 患者；最终重训合计：83949 窗口 / 1556 患者；独立测试：22638 窗口 / 389 患者。
- 软标签由六类专家票数归一化；测试患者未进入轮数选择或最终重训。

## 输入与预处理

- 采样率：200 Hz；19 个 EEG 导联（不含 EKG）：Fp1, F3, C3, P3, F7, T3, T5, O1, Fz, Cz, Pz, Fp2, F4, C4, P4, F8, T4, T6, O2。
- 通道：原始 EEG=False；同窗 STFT 频谱=True；1 秒图帧=False。
- 以 `eeg_label_offset_seconds` 裁出 50 秒；短内部缺失最多插值 20 采样点，长缺失保留掩码并填零；超过 5% 缺失或恒定导联的窗口不进入有效集。
- 频谱分支：STFT 256 采样点窗、64 采样点步长，幅值 `log1p`。图分支：每导联 5 个频带的 1 秒节点属性，相关系数作为统计关联代理边。
- 此模型不使用图输入；恒等图归一化占位文件 SHA256：`c2f04c4f7e0e8536881229f8c751dc3a422003205aeabf35d53efe0e5e0dc5a3`。

## 训练与评估

- 初训从验证集选择第 5 轮（从 1 计数），依据 `kl`；源 checkpoint SHA256：`3a500f410fbc5152c71bbc489eaccbd6f9816404411ba87359d99f19ac392597`。
- 最终模型在训练+验证患者上固定训练 5 轮；不在重训阶段重新选择轮数。
- 重训随机种子：42；配置：`artifacts/refit/spectrogram_cnn/config.json`；配置 SHA256：`d9fd5a88e51231cf4993deb89fedc00f1ad2115bd3057a1beb1f17b3b8e211d7`。
- 训练参数：`{"batch_size": 16, "epochs": 5, "forecast_weight": 0.0, "grad_clip": 1.0, "lr": 0.001, "patience": 8, "weight_decay": 0.01}`。
- 主指标：KL(target || prediction)，逐窗口平均；患者等权 KL 与 macro F1 为补充。

| 划分 | 窗口 | 患者 | 窗口 KL ↓ | 患者等权 KL ↓ | Macro F1 |
|---|---:|---:|---:|---:|---:|
| test | 22638 | 389 | 1.322836 | 1.292575 | 0.295653 |

- 测试预测逐窗口覆盖冻结测试集；指标与预测已由 `scripts/refit_final.py summarize` 重算核对。
- 五模型正式对比表：`results/refit/final_comparison.json`；SHA256：`4f26fa1c151b74b06813f69ee3a696a3d07ea6d25089b5ee76ff5b4accb6d64d`。
- 结果只对应此处固定患者划分、预处理和文件哈希；跨中心、设备、人群泛化未经验证。

## 限制

- 只用于离线研究展示；不能用于临床诊断、实时发作预警或患者治疗。
- 专家票数存在分歧；脑电相关边与潜在 ODE 轨迹不构成生理因果解释。

## 本地复现

```bash
.venv/bin/python scripts/refit_final.py train --source-run artifacts/runs/spectrogram_cnn --run-dir artifacts/refit/reproduce_spectrogram_cnn
.venv/bin/python scripts/refit_final.py evaluate --run-dir artifacts/refit/reproduce_spectrogram_cnn
```

权重仅位于本地 `artifacts/refit/spectrogram_cnn/final.pt`；本脚本不发布权重或生成下载链接。
