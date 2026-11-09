# 项目执行状态

更新：2026-11-09 02:00 Asia/Shanghai。

| 阶段 | 状态 | 证据 / 下一步 |
|---|---|---|
| S0 环境盘点 | 已验证 | 实际为 2×V100S 32GB、40 CPU、251GB RAM、共享文件系统；Python 3.8 项目环境、Spark 3.4.4、PyTorch 2.0.1 均已安装。 |
| S0 官方来源审计 | 已完成 | 固定 ODEBrain SHA `31e251ede69ed82fb47c627d2590286b6376c50a`；独立改造明确命名 ODEBrain-inspired。 |
| S1 真实小样本闭环 | 已验证（smoke） | 36 EEG 文件/364 有效窗口；TCN 与 ODE 的 CPU 两轮训练及验证预测均运行。验证输出标 `smoke_subset=true`，不得解释为正式泛化。6 个 ODE 动力学案例已导出，均标 `real_hms_smoke`。 |
| S2 Spark 流水线 | 真实子集 smoke 已验证 | Spark local[4] 处理 364 窗口并输出 Parquet、event log；LR/RF 跑通，但平衡抽样数据不得用于正式结论。 |
| S3 深度模型 | 真实两阶段 smoke 已运行 | TCN、频谱 CNN、融合、GNN+GRU、ODEBrainInspiredHMS 的合成张量前后向测试通过；ODE 图预测预训练与六分类微调在真实 36 文件子集 CPU 上运行。 |
| S4-S6 正式实验、展示、交付 | 全量原始数据与审计完成；正式实验运行中 | 官方 ZIP 全包 CRC 校验通过；17,300 EEG 和 11,138 频谱文件已解压。分类与预测图缓存均已建成。TCN 正式训练及完整测试已完成；频谱 CNN 正在训练。Spark 21 行边界缺失窗口已按统一特征算法修复，106,587/106,587 行均合格；正式 LR/RF 已完成完整测试。其余正式模型、图预测对照及最终结果尚未完成。24 页 DRAFT PPTX 已用本地 LibreOffice 转成 24 页 DRAFT PDF，均非正式交付。 |

## 当前约束

- 服务器与文档背景不符：2 张 V100S 而非 8 张 A100；不按 8 卡预算承诺训练。
- 已冻结的 36 文件子集按类别和患者刻意平衡，不能作为正式泛化实验。
- GPU 可用性会变化，正式训练启动前需重新检查占用。
- 全量质量审计覆盖 106,800 行、17,089 个被元数据引用的 EEG；106,587 个有效窗口，211 个缺失过多、2 个常量通道窗口排除。患者级 train/val/test 为 67,114/16,835/22,638 个窗口，划分 SHA256 `f334b0e31f5faf520f85a2d4c79631caf7edef2102460a8bcd629fc2a699a82f`。
- 旧预测缓存 `data/training/graph_forecast` 因 EEG `1001717358` 的部分前 5 秒全常量窗口失败；分类 50 秒窗口仍合格。新缓存 `data/training/graph_forecast_v2` 采用独立 `eligibility_reason`，已校验 106,434/106,587 个预测 eligible 窗口，排除 153 个；测试集 22,606/22,638 个 eligible，排除 32 个。抽查 10 个真实测试窗口，缓存与直接计算张量完全相同。分类 manifest、划分和缓存不变。正式预测仅评价 eligible 子集并披露排除原因。见 [资格协议](forecast_eligibility.md)。
- TCN 正式训练在 29 轮后结束，完整 test 评估覆盖 22,638 窗口/389 患者，窗口平均 KL 为 1.1966（`results/tcn/test_metrics.json`）；这只是单模型结果，不代表最终模型比较。`hms_deep_queue` 已继续频谱 CNN。Spark 特征表 train/val/test 分别为 67,114/16,835/22,638 行且全部 `quality_ok`；正式 LR/RF 测试窗口平均 KL 为 1.2617/1.1568（`results/spark_baselines/metrics.json`）。`hms_benchmark_queue` 已在完整基线完成后重新启动。
- 本地 LibreOffice 7.6.7.2 与中文字体已放在忽略目录 `artifacts/pdf-tools/`；`scripts/export_slides_pdf.sh` 已将 DRAFT PPTX 转换为 DRAFT PDF，PPTX/PDF/渲染 PNG 均为 24 页。正式 PPTX/PDF 仍须真实结果和证据门禁通过；见 [导出验证](pdf_export.md)。
- GitHub 已登录，但公共仓库创建和推送须在可复查内容完成后进行；不虚构资源链接。

## 下一个可执行闭环

1. 监控 GPU 深度训练队列，执行图预测预训练、分类微调与验证/测试，并核对 eligible 子集。
2. 完成三方法、每设置三重复的性能基准，并用完整开发集 refit 检查最终测试结论。
3. 冻结正式结果，生成模型卡、静态图、PPTX/PDF，验证资源链接。
