# 项目执行状态

更新：2026-11-09 16:28 Asia/Shanghai。

| 阶段 | 状态 | 证据 / 下一步 |
|---|---|---|
| S0 环境盘点 | 已验证 | 实际为 2×V100S 32GB、40 CPU、251GB RAM、共享文件系统；Python 3.8 项目环境、Spark 3.4.4、PyTorch 2.0.1 均已安装。 |
| S0 官方来源审计 | 已完成 | 固定 ODEBrain SHA `31e251ede69ed82fb47c627d2590286b6376c50a`；独立改造明确命名 ODEBrain-inspired。 |
| S1 真实小样本闭环 | 已验证（smoke） | 36 EEG 文件/364 有效窗口；TCN 与 ODE 的 CPU 两轮训练及验证预测均运行。验证输出标 `smoke_subset=true`，不得解释为正式泛化。6 个 ODE 动力学案例已导出，均标 `real_hms_smoke`。 |
| S2 Spark 流水线 | 全量特征、LR/RF、性能基准已完成 | 106,587 个有效窗口均有质量合格特征；LR/RF 覆盖 22,638 个测试窗口。三种方法、三种数据量共 36 次基准运行完成，每种配置三次。 |
| S3 深度模型 | ODE 分类初训运行中 | TCN、频谱 CNN、融合模型和 ODE 图预测预训练均完成；ODE 分类初训已到第 19 轮，其后自动接 GraphGRU 预测预训练与分类。 |
| S4-S6 正式实验、展示、交付 | 三个分类模型重训完成；图模型重训运行中 | TCN、频谱 CNN、融合的 `train+val` 固定轮次重训及完整测试均完成；ODE 图预测重训在 GPU 0 运行。正式模型卡、动力学、预测对照和预览 PPT/PDF 已设自动后处理队列。24 页 DRAFT PPTX/PDF 已核对页数，仍不是正式交付。 |

## 当前约束

- 服务器与文档背景不符：2 张 V100S 而非 8 张 A100；不按 8 卡预算承诺训练。
- 已冻结的 36 文件子集按类别和患者刻意平衡，不能作为正式泛化实验。
- GPU 可用性会变化，正式训练启动前需重新检查占用。
- 全量质量审计覆盖 106,800 行、17,089 个被元数据引用的 EEG；106,587 个有效窗口，211 个缺失过多、2 个常量通道窗口排除。患者级 train/val/test 为 67,114/16,835/22,638 个窗口，划分 SHA256 `f334b0e31f5faf520f85a2d4c79631caf7edef2102460a8bcd629fc2a699a82f`。
- 旧预测缓存 `data/training/graph_forecast` 因 EEG `1001717358` 的部分前 5 秒全常量窗口失败；分类 50 秒窗口仍合格。新缓存 `data/training/graph_forecast_v2` 采用独立 `eligibility_reason`，已校验 106,434/106,587 个预测 eligible 窗口，排除 153 个；测试集 22,606/22,638 个 eligible，排除 32 个。抽查 10 个真实测试窗口，缓存与直接计算张量完全相同。分类 manifest、划分和缓存不变。正式预测仅评价 eligible 子集并披露排除原因。见 [资格协议](forecast_eligibility.md)。
- 初训单划分完整测试覆盖 22,638 窗口/389 患者：TCN KL 1.1966，频谱 CNN 1.2136，时频融合 1.1530。这些是初训阶段评估，最终主表以 `train+val` 重训后的独立测试为准。Spark 特征表 train/val/test 分别为 67,114/16,835/22,638 行，均 `quality_ok`；正式 LR/RF 测试 KL 为 1.2617/1.1568。三方法性能基准正在完成全量 Spark 重复。
- `hms_refit_queue` 使用 GPU 0 按冻结最佳轮次重训七个 run；`hms_deep_queue` 使用 GPU 1 继续图模型初训。后处理队列等待图模型完成，随后生成图预测对照、动力学、模型卡、专家分歧和消融。两个额外种子 43/44 已设接力队列，分别在 GPU 1/0 释放后启动；正式结果预览队列优先使用 GPU 1。相关队列与日志位于忽略目录 `artifacts/`。
- Spark 正式基准 `results/spark_benchmark/{runs,summary}.json` 共 36 次运行和 12 个三重复配置。全量中位数：串行 4063 秒、四进程 1046 秒、Spark 单 worker 4043 秒、Spark 四 worker 2146 秒；同一共享盘和一台物理机，不能推断多机加速。
- 本地 LibreOffice 7.6.7.2 与中文字体已放在忽略目录 `artifacts/pdf-tools/`；`scripts/export_slides_pdf.sh` 已将 DRAFT PPTX 转换为 DRAFT PDF，PPTX/PDF/渲染 PNG 均为 24 页。正式 PPTX/PDF 仍须真实结果和证据门禁通过；见 [导出验证](pdf_export.md)。
- GitHub 已登录，但公共仓库创建和推送须在可复查内容完成后进行；不虚构资源链接。

## 下一个可执行闭环

1. 监控初训、全量开发集重训和 Spark 基准队列，核对进度与异常。
2. 核对图预测 eligible 测试、消融、专家分歧及重训结果，并据此生成最终模型卡和静态图。
3. 冻结正式结果，生成 PPTX/PDF，逐页核查并验证资源链接。
