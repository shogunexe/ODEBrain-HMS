# 正式测试分类辅助指标

生成：`LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-} .venv/bin/python scripts/supplement_classification_metrics.py`。正式产物是 `results/classification_weighted_metrics.json`。脚本仅读取最终五个深度模型的 `test_predictions.csv`、LR/RF 的 Spark Parquet 和冻结的测试 manifest；逐一核对完整窗口覆盖、标签、文件 SHA 和已有三种 KL，绝不重训或以测试集指标选择配置。

三种口径在**同一 22,638 个标注窗口**上计算，类别顺序为 `seizure, lpd, gpd, lrda, grda, other`：

| JSON 键 | 每个窗口权重 | 混淆矩阵元素总和 |
|---|---:|---:|
| `window_equal` | 1 | 22,638 |
| `patient_equal` | 1 / 该患者测试窗口数 | 389 |
| `eeg_record_equal` | 1 / 该 `eeg_id` 测试窗口数 | 3,931 |

患者与 EEG 记录等权是**窗口指标的分组加权**。模型仍为每个窗口给出一组概率；没有做患者或 EEG 记录级投票，也没有把一个组变成一个独立预测。加权混淆矩阵的元素可为小数。每类 `false_positive_weight` 为加权误报数量；`false_positive_rate = FP / (FP + TN)`，以该类以外的加权窗口为分母。它不是每小时误报次数。`ovr_auroc` 为每一类对其余类的加权 one-vs-rest ROC AUC。Recall、F1、precision 和 macro F1 都从相同权重的混淆矩阵计算。硬标签取专家共识 `class_id`；KL 对专家投票的六类软标签计算。

`kl_mean` 在三种口径分别等于正式主表或 Spark 补充产物中的窗口、患者、EEG 记录等权 KL。深度模型主表采用 train+val 固定轮数重训，Spark MLlib 分类器采用训练集训练；两种训练协议与初训及消融均不同，测试指标不得相减解释为直接消融，也不得反向用于选择模型或超参数。
