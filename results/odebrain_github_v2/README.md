# ODEBrain v2 正式结果索引

这些结果对应 `src/models/odebrain_hms.py` 的 `github_trajectory_v2`，不是旧版 `legacy_v1`。新版项目汇报 PPT 由汇报方本地保存；公开的方法、数值和解释边界见 [`docs/odebrain_github_v2_results.md`](../../docs/odebrain_github_v2_results.md)。

| 文件 | 内容 |
|---|---|
| `final_comparison.json` | 五个深度模型在同一患者互斥测试集上的窗口/患者/记录 KL 与宏 F1；正式 train+val 重训。 |
| `classification_weighted_metrics.json` | 七模型的软标签 KL、逐类硬分类指标和混淆矩阵；LR/RF 为 train-only 基线。 |
| `paired_kl_intervals.json` | 按患者成组重采样的模型间配对 KL 差描述性 95% 区间。 |
| `forecast_comparison.json` | 独立未来节点属性预测任务的 +1/+3/+5 秒 MAE。 |
| `six_class_field/*.json` | 六类局部投影场的设置与数值摘要；图像为同一共享 ODE 场的不同局部切面。 |
| `cases/case_pair.json` | 两个冻结测试案例的专家票、模型概率与轨迹投影数据。 |

`figures/` 中的对比图与 `six_class_field/`、`cases/` 中的可视化用于展示，不构成 EEG 物理动力学或临床因果解释。所有六分类结果来自既定患者划分的冻结测试集；不同训练预算和不同评价单位不能混为一个排行榜。
