# 正式实验结果与解释边界

本页只汇总已经冻结的 HMS 测试集结果。全部 106,800 条元数据经过质量审计，106,587 个分类窗口有效；患者互斥划分的测试集有 22,638 窗口、389 位患者、3,931 条 EEG 记录，划分 SHA256 为 `f334b0e31f5faf520f85a2d4c79631caf7edef2102460a8bcd629fc2a699a82f`。完整数值、预测文件哈希及复算程序见 [最终主表](../results/refit/final_comparison.json)、[三种等权指标](../results/classification_weighted_metrics.json)和[口径说明](classification_weighted_metrics.md)。

KL 使用六类专家投票比例作目标，越低越好。窗口等权是各测试窗口平均；患者等权先在每位患者内取窗口平均再平均；EEG 记录等权按 `eeg_id` 同样计算。后两者仍评价窗口预测，并非先将一位患者或一条记录合成一个预测。Kaggle 官方评分口径不能由这里的患者或记录等权值代替。宏平均 F1 使用专家共识硬标签，和 KL 的软目标不同。

## 最终分类主表

五个深度模型均按验证集选定的轮数，在 train+val 上固定轮次重训；下表为预设 seed 42 的同一完整测试集。

| 模型 | 窗口 KL | 患者 KL | EEG 记录 KL | 窗口宏 F1 |
|---|---:|---:|---:|---:|
| TCN | 1.2000 | 1.1831 | 1.2866 | 0.4638 |
| 频谱 CNN | 1.3228 | 1.2926 | 1.3949 | 0.2957 |
| 时频融合 | 1.0530 | 1.1096 | 1.0300 | 0.4484 |
| GraphGRU | 1.3611 | 1.4374 | 1.1757 | 0.4056 |
| ODEBrain-inspired HMS | 1.6276 | 1.4139 | 1.7171 | 0.2101 |

Spark MLlib 的逻辑回归与随机森林使用 train-only 训练协议，同测试集上分别为窗口/患者/记录 KL `1.2617/1.3449/1.3444` 与 `1.1568/1.1462/1.1798`。Spark 是特征处理与 MLlib 运行框架，逻辑回归和随机森林才是模型。这两行与 train+val 深度模型采用不同训练预算，应单独注明。

三种子 `42/44/45` 的窗口 KL 均值为：时频融合 `1.1616`、TCN `1.2224`、GraphGRU `1.2258`、频谱 CNN `1.3810`、ODE `1.5408`；完整三种口径及样本标准差见[种子稳定性汇总](../results/seed_stability/summary.json)。按窗口等权，ODE 没有优于 GraphGRU。患者等权的 seed 42 点估计方向相反，但[最终权重患者配对 bootstrap](../results/refit/paired_kl_intervals.json)给出 ODE 减 GraphGRU `−0.0235`，95% 区间 `[−0.1337, 0.0869]`；这是描述性区间，不能称患者等权优势已证实。区间没有用于测试集选模，也没有做多重比较后的显著性声明。

逐类 precision、recall、F1、OVR AUC、误报率及三种加权混淆矩阵均在[辅助指标 JSON](../results/classification_weighted_metrics.json)。这些指标补充说明硬分类错误，主要评价仍是专家软标签 KL。

## 消融与预测

三条[固定消融](../results/ablation.json)均为 train-only 训练、验证早停的 `best.pt`，全量测试结果如下；[三种等权及逐类指标](../results/ablation_weighted_metrics.json)由相同的冻结预测文件独立复算。

| 消融臂 | 窗口 KL | 患者 KL | EEG 记录 KL |
|---|---:|---:|---:|
| 预测预训练，RK4 步长 0.25 秒 | 1.3219 | 1.3654 | 1.2478 |
| 从头训练，RK4 步长 0.25 秒 | 1.2688 | 1.3711 | 1.3426 |
| 预测预训练，RK4 步长 0.50 秒 | 1.1079 | 1.1788 | 1.0505 |

不同步长分别训练后的测试 KL 不是同一权重上的数值收敛实验；这三行也不能与最终 train+val 重训主表直接相减。没有用这些测试值反过来选模型配置。

[最终预测对照](../results/refit/forecast_comparison.json)只覆盖预测合格的 22,606 个测试窗口、387 位患者、3,922 条 EEG 记录。原始尺度图节点属性 1/3/5 秒 MAE：ODE 为 `0.6565/0.6833/0.7031`，GraphGRU 为 `0.6415/0.6709/0.6885`，persistence 为 `0.7830/0.8137/0.8438`。它是辅助图属性预测任务，不代表发作预警或临床诊断。

## 交付追溯

正式界面截图和 24 页 PDF 使用[证据清单](../results/slides_evidence_preview.json)中的文件 SHA256 绑定实验来源；公开仓库和权重 Release 的匿名下载校验在发布后写入 `results/formal_resources.json`，正式证据清单与最终 PDF 才会生成。六个真实 HMS 案例的图及权重资源见[公开资源说明](public_resources.md)。
