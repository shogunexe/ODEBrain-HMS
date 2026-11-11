# ODEBrain-HMS 项目独立审阅清单

核查时间：2026-11-11 19:49 Asia/Shanghai。范围：原始项目方案、当前代码、正式结果 JSON、运行完成标记、训练历史、交付目录；未重新训练模型，也未把测试集用于修正配置。本清单把“产物存在”“结果可核对”和“原方案全部完成”分开。

## 已完成的工作及证据

| 工作 | 状态与可核查证据 |
|---|---|
| 来源与环境 | 已记录官方 ODEBrain commit 和 HMS 改造边界：`docs/odebrain_adaptation.md`；初始服务器盘点：`docs/environment.md`、`environment.json`。模型明确标为 ODEBrain-inspired，不声称是官方完整移植。 |
| 真实数据与划分 | 官方训练包完整性校验、17,300 个 EEG 和 11,138 个频谱文件解压；106,800 条元数据经质量审计得 106,587 个有效分类窗口。患者级 train/val/test 为 67,114/16,835/22,638 窗口，分别 1,245/311/389 患者；`results/data_audit.json`、`results/split_summary.json` 与 `data/curated/split_manifest.csv` 可复查。审计核验无重复窗口 ID、三组患者交集为零。 |
| Spark | 全量特征、LR/RF 基线与 10/30/100% 数据量、串行/四进程/Spark 1/4 槽、每配置三次的 36 次性能运行完成；`results/spark_baselines/metrics.json`、`results/spark_benchmark/{runs,summary}.json` 与 `results/spark_benchmark/spark_event_logs/`。LR/RF 测试 KL 为 1.2617/1.1568；另用统一 float64 指标代码抽核一致。实验是单物理机共享磁盘，不能当多机扩展证据。 |
| 五个深度分类模型 | TCN、频谱 CNN、融合、GraphGRU、ODEBrain-inspired 完成初训、全测试与 train+val 固定轮数重训。最终主表 `results/refit/final_comparison.json` 逐模型校验 checkpoint、配置、全 22,638 个测试窗口及预测文件 SHA256。seed 42 的融合/ODE/GraphGRU KL 分别为 1.0530/1.6276/1.3611。 |
| 三种子稳定性 | seed 42/44/45 的最终五模型测试和 `results/seed_stability/summary.json` 完成；融合 KL 1.1616±0.1346，TCN 1.2224±0.0196，GraphGRU 1.2258±0.1262，ODE 1.5408±0.1004。标准差仅对应同一患者划分上的初始化变化，不是三折患者验证。原计划 seed 43 因旧队列失败未运行，45 按资源可用性替代。 |
| 图预测与动力学 | 初训与最终重训的 1/3/5 秒图节点预测对照已生成；最终结果 `results/refit/forecast_comparison.json` 覆盖 22,606 个合格测试窗口、387 位患者。ODE MAE 为 0.6565/0.6833/0.7031，GraphGRU 为 0.6415/0.6709/0.6885，persistence 为 0.7830/0.8137/0.8438。六个最终权重真实案例及静态图在 `results/dynamics_refit/`。 |
| 展示与模型卡 | `app/main.py` 有五页结果浏览；`docs/model_cards_refit/` 有五个最终模型卡；`docs/model_cards/` 是初训探索卡。旧 24 页 `slides/ODEBrain_DRAFT_pending_experiments.{pptx,pdf}` 已导出，但明确是草稿。 |

项目测试在 11 月 11 日复查为 77 项通过，含性能图分组修复的定向测试。测试覆盖关键接口和部分数值性质，不能代替全新环境复现或论文层面的研究验证。

## 发现的问题

### 高优先级：阻止宣称全部交付完成

1. **正式交付文件未齐。** `results/ablation.json` 尚未生成，消融第二条分类臂仍在训练，第三条未开始；因此 `results/formal_ui_capture.json`、正式图表、本地预览 PPT/PDF 和最终 PPT/PDF 均未生成。当前只有旧 DRAFT。原方案要求最终 PDF 是主要交付。见 `docs/status.md`、`results/formal_delivery_inputs.json`、原方案第 16–18 节。
2. **预测指标尚未达到方案口径。** `scripts/evaluate_forecast.py` 只保存反标准化总体 MAE，缺少要求的标准化 MAE/MSE、逐频带误差和便于独立复算的逐窗口误差摘要。现有预测 MAE 是真实结果，但不能称预测评价已完全按方案执行。见原方案第 251 行。
3. **公开资源与复现交接未完成。** 当前 Git 无 remote，公开仓库、权重访问地址及已验证超链接尚无；也未见全新环境按 README 运行 smoke 的记录。原方案第 17–18 节要求这两项。不要把本机单次测试或 DRAFT PDF 称为最终交付。
   当前 `results/dynamics_refit/result_index.json` 和案例元数据已在本地 Git，但案例 `.npz`、PNG 与 SVG 尚未作为公开资源包冻结；从仅含当前 Git 历史的干净克隆无法直接打开这些图。发布前要确定图源、再分发许可、必要权重和资源包地址，并实际从干净克隆验证链接。

### 中优先级：研究解释与方法缺口

4. **不能声称 ODE 分类优于离散图模型。** seed 42 按窗口平均 KL，ODE 1.6276，高于 GraphGRU 1.3611；三种子 ODE 均值 1.5408，高于 GraphGRU 1.2258。患者等权 KL 在 seed 42 为 ODE 1.4139、GraphGRU 1.4374，方向相反；只读配对患者 bootstrap 粗核显示差值约 −0.023、95% 区间约 [−0.134, 0.086]，尚非正式保存的分析产物，不能据此宣称患者等权差异。最终主表应把两种汇总口径分开。
5. **最终重训模型的配对差异区间缺失。** `results/disagreement_analysis.json` 只分析初训模型；`results/refit/final_comparison.json` 有最终点估计和单模型指标，但没有最终模型间以患者为单位的配对 bootstrap 置信区间。见原方案第 249 行。
6. **优先消融未全覆盖。** 当前 `configs/ablations.yaml` 比较预测预训练有无和 RK4 步长，尚无固定深度模型的软标签/硬标签消融。也没有在同一权重上缩小积分步长的数值收敛试验；不同步长分别训练后比较测试 KL 不能替代数值收敛。见原方案第 179、236–241 行及 `docs/ablation.md`。
7. **ODE 成本证据不足。** 训练历史有每轮耗时，模型和案例有部分 NFE；还缺固定批量、GPU 同步推理耗时/吞吐、峰值显存、求解失败次数的正式汇总。幻灯片不能声称已完整比较“效果与成本”。见原方案第 253 行。
8. **初训、消融和最终主表协议不同。** 消融用训练集训练、验证早停的 `best.pt`；最终主表用 train+val 固定轮数重训 `final.pt`。两套测试 KL 可以分别描述，不能当同预算同权重的直接消融。初训 CLI 种子主要由命令传递，源配置本身未记录实际种子；正式 refit 和消融各自有更明确的种子证据。

### 冗余与已修正错误

9. 消融重新跑了与初训相近的 0.25 秒 ODE 预测预训练和分类链，耗费额外 GPU 时间；独立消融来源可追踪，因此属于**可解释的重复计算**。三种子重复先于消融和 PDF 完成，与原方案“时间不足时优先核心交付”的排程不一致，但种子结果本身有效。初训与最终重训各有模型卡、预测和动力学案例是版本追踪所需；最终汇报只使用 `refit` 作为主结果。失败的 v1 预测缓存和 smoke 缓存可在核验后清理。
10. 曾有两处真实后处理错误，现已修复：模型卡与消融门禁把“有效 EEG 记录占比”错误要求为 100%，实际审计比例为 99.5904%；专家分歧分析的患者配对 bootstrap 有重复掩码。对应修复在 `scripts/write_model_cards.py`、`scripts/run_ablations.py`、`scripts/analyze_disagreement.py`，并有测试。
11. 本次审阅发现性能页把不同方法和数据规模连成一条折线，已按方法及规模分轨、对同配置重复取中位数，见 `app/main.py` 和 `tests/test_app_formal_results.py`。过时的覆盖率、预测状态、动力学路径和幻灯片示例输入说明也已更正。正式 UI 截图尚需消融完成后再实测。

## 审阅结论与限制

目前没有发现患者划分交叉或图预测输入读取切点后数据；正式指标文件带完整测试范围、模型和哈希证据。**项目仍未达到原方案的 S6 完成交接。**主要剩余工作是消融完成后生成并逐页检查正式结果预览，补预测指标/成本/配对区间或在最终 PDF 明示删减，完成干净环境复现与资源发布核验。原方案允许时间不足时削减额外实验，但必须在总结中写清。
