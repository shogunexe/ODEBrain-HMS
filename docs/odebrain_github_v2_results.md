# ODEBrain v2 HMS 六分类：结果与解释边界

本页记录新版项目汇报所用的 `github_trajectory_v2` 结果，汇报 PPT 由汇报方本地保存。模型是以 [ODEBrain 官方仓库](https://github.com/HHJIAnmo/ODEBRAIN/tree/31e251ede69ed82fb47c627d2590286b6376c50a)为参照的 HMS 独立适配，包含官方形式的原始波形 CNN、三层 Tanh ODE 函数及 12 个潜在状态的轨迹均值分类；电极图构造、HMS 输入适配和六类软标签目标均由本项目实现。详细偏离见[源码审计](odebrain_adaptation.md)。

## 任务与数据

输入是 50 秒 EEG，专家票对应中央 20–30 秒。六类票数归一化为软标签，主要指标为测试窗口的平均 KL（越低越好）；宏 F1 使用多数票类别作硬标签，回答的是不同问题。患者互斥划分的冻结测试集有 22,638 个窗口、389 位患者、3,931 条 EEG 记录。五个深度模型在验证集选定轮数后，统一按固定轮数在 train+val 上重训。Spark MLlib 的 LR/RF 是 train-only 基线，训练预算不同。

## 冻结测试结果

| 模型 | 窗口 KL↓ | 记录等权 KL↓ | 患者等权 KL↓ | 窗口宏 F1↑ |
|---|---:|---:|---:|---:|
| Spark LR | 1.262 | 1.344 | 1.345 | 0.285 |
| Spark RF | 1.157 | 1.180 | 1.146 | 0.390 |
| TCN | 1.200 | 1.287 | 1.183 | **0.464** |
| 频谱 CNN | 1.323 | 1.395 | 1.293 | 0.296 |
| 时频融合 | **1.053** | 1.030 | **1.110** | 0.448 |
| GraphGRU | 1.361 | 1.176 | 1.437 | 0.406 |
| ODEBrain v2 | 1.082 | **1.026** | 1.149 | 0.423 |

逐窗口和逐类指标见 [`classification_weighted_metrics.json`](../results/odebrain_github_v2/classification_weighted_metrics.json)，深度模型主表见 [`final_comparison.json`](../results/odebrain_github_v2/final_comparison.json)。窗口 KL 衡量所有标注窗口的软概率拟合；记录和患者等权指标先对组内窗口求均值，再对组平均，不能替代窗口指标或 Kaggle 官方私有测试分数。

时频融合的窗口 KL 最低，TCN 的宏 F1 最高，说明软标签拟合与硬分类排序不完全相同。ODE v2 的窗口 KL 比时频融合高 0.0291。以患者为重采样单位的配对 bootstrap 给出的窗口 KL 差（ODE−融合）95% 描述性区间为 `[-0.0364, 0.1006]`，包含 0；记录等权的点估计差为 −0.0038，区间 `[-0.0662, 0.0640]`。这组结果不能宣称 ODE 在六分类上稳定或显著优于融合，区间也未用于测试集选模。详见 [`paired_kl_intervals.json`](../results/odebrain_github_v2/paired_kl_intervals.json)。

逐类硬标签结果还显示 ODE v2 的 LRDA F1 约 0.01，说明其总 KL 排名不能掩盖特定类别的失败。未来电极节点对数频带功率预测是独立辅助任务，合格测试子集为 22,606 个窗口。+5 秒 MAE：ODE `0.7032`、GraphGRU `0.6885`；ODE 没有优于该离散图模型。详见 [`forecast_comparison.json`](../results/odebrain_github_v2/forecast_comparison.json)。

## Neural ODE 在此任务中的价值

Neural ODE 学习潜在状态的变化率 `dz/dt=fθ(z)`，然后沿指定时间点求解状态。它为 EEG 中持续变化的波形、节律与电极间关联提供可研究的潜在轨迹；本项目用 0–1 秒 ODE 时间的 12 个状态均值进行分类。这里的积分时间是模型的潜在时间，**不是**继续观测到 1 秒或 5 秒新 EEG。ODE 的连续形式也允许检查轨迹和向量场，不过本项目使用规则的每秒图快照，尚未实测不规则采样优势。

六类图共用同一个自主 ODE 函数 `fθ`，在不同输入的高维起点附近取局部切面，因而呈现不同的二维流线。场图、轨迹和投影场强只显示训练集 PCA 二维投影，提供返回原 EEG 检查的线索；它们不是脑电的物理传播、疾病严重度或模型误判的因果解释。目前没有观察到可验证的六类独立吸引子。图来自正式检查点，但投影和数值求解一致性仍应审计。

## 与原论文的关系

[Jia 等人的 ODEBrain 论文](https://arxiv.org/abs/2602.23285)使用 TUSZ 做 **seizure detection（发作检测）** 二分类，并评估未来图预测；它不是本项目的 HMS 六类专家投票任务，也不是直接的未来发作预警。原论文的性能增益不能搬到 HMS，两个数据集和指标的绝对分数也不可直接比较。本项目的结果支持时频结合是当前六类软标签任务的强基线，ODE 更适合作为需要连续状态分析或未来图预测时进一步检验的方法。

参考方法：[Chen 等，Neural Ordinary Differential Equations，NeurIPS 2018](https://papers.neurips.cc/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html)；[Rubanova 等，Latent Ordinary Differential Equations for Irregularly-Sampled Time Series，NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html)；[Kaggle HMS 竞赛](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)。
