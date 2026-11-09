# Spark 特征与传统基线

## 输入、输出

窗口 manifest 的 CSV、JSONL 或 Parquet 一行对应一个标注窗口。必需列：`eeg_path`（原始 EEG Parquet 的绝对路径）、`eeg_id`、`patient_id`、`eeg_label_offset_seconds`、`split`（`train`/`val`/`test`）、`class_id`（固定类别顺序中的 0–5）；可选 `window_id`、原始 HMS `label_id`、`sample_rate`（默认 200）、`audit_reason`。只处理三个有效 split 且 `audit_reason=ok` 的行；保留在 manifest 中的排除行不会进入特征表。`label_id` 是标注记录 ID，不是分类标签。原始 EEG Parquet 应为时间行、通道列；`.npz` 只支持合成 smoke。

每个窗口从偏移开始读取 50 秒，即 10,000 个 200 Hz 采样点。特征覆盖标准差、峰峰值、线长、Hjorth 参数、五个频带的绝对及相对功率、谱熵、左右频带功率差、导联相关和逐秒相关图摘要。输出 `eeg_features_v1` Parquet，含标识、质量、错误原因和数值特征，以 `split` 分区。Spark 使用 `repartition(eeg_path)` + 分区内排序，同一原始文件的窗口只读一次；时间序列不被打散到不同分区。窗口越界和读取失败保留为 `quality_ok=false`，不会悄悄填零。

HMS 个别窗口在起点或终点有全部导联共同缺失的 0.75–2 秒。若至少 12 个导联在边界缺失、两端公共裁切各不超过窗口的 5%，特征抽取先用全部导联相同的有效时间区间裁切；`missing_fraction` 仍按原 50 秒计算，`duration_seconds` 记录裁后的有效时长。边界长缺口不做插值，内部超过 0.25 秒的缺口仍使对应导联无效。正式特征表中的 21 个边界缺失窗口已用同一函数重算，修复记录在 `data/features/eeg_features_v1_summary.json`，旧表备份在 `artifacts/feature_repair_backup_20261109/`。

这里的逐秒相关图仅输出人工特征摘要；ODE 图预测训练的完整图快照须由对应数据模块生成。特征含整个 50 秒上下文，仅供离线分类，不可作为严格未来预测分支的输入。

## 运行

本机 Python 3.8 环境和 Java 8 的启动方式：

```bash
export LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-}
export PYSPARK_PYTHON="$PWD/.venv/bin/python"
.venv/bin/python scripts/smoke_spark_features.py
.venv/bin/python scripts/smoke_spark_baselines.py
.venv/bin/spark-submit scripts/build_features.py --config configs/spark.yaml
.venv/bin/spark-submit scripts/train_spark_baselines.py --features data/features/eeg_features_v1 --output artifacts/runs/spark_baselines --master 'local[2]'
.venv/bin/python scripts/benchmark_spark.py --manifest data/curated/split_manifest.csv --output results/spark_benchmark --workers 1 2 4 --repeats 3
```

`configs/spark.yaml` 使用 `spark_features: {manifest, output, master, partitions, event_log_dir}`。首次实验先用本机小数据运行；连接真实 Standalone master 时设置 `--master spark://...`，并保证所有 Worker 可读取相同的 `eeg_path` 及依赖。

Spark LR 使用训练集拟合中位数填补及标准化；RF 使用训练集拟合中位数填补。测试集只做变换和评价。预测 Parquet 保留窗口、患者、原始 `label_id`、`class_id` 和六类概率。数据不足或缺失某一类别时，模型的六维概率输出须检查：正式六分类训练要求训练集涵盖全部六类。

只下载少数患者的运行请在两条作业上加 `--experiment-status real_data_smoke`，结果 JSON 会带该标记。其指标只验证代码链路，不进入正式模型表。

2026-11-09 全量正式运行：`data/features/eeg_features_v1` 的 106,587 个窗口与审计清单 ID 完全一致，全部 `quality_ok=true`（train 67,114 / val 16,835 / test 22,638）。Spark LR/RF 结果在 `results/spark_baselines/metrics.json`，每个模型的三份预测 Parquet 也逐项覆盖对应 split 的全部窗口。重试日志与退出码在 `artifacts/spark_formal_retry.log`、`artifacts/spark_formal_retry.exitcode`（0）。

性能脚本按原始文件的 SHA-256 稳定排序抽取嵌套 10%/30%/100% 子集，记录原始字节、文件、窗口和区间并集时长。默认每种设置重复三次，记录读取 manifest、计算和写出的时间，另记 Spark 会话创建与端到端时间，并保存 Spark event log。同一 Python 进程中的后续 Spark 会话可能复用 JVM，故会话创建时间不是冷启动时间。`local[n]` 是单物理机的进程/线程并行；共享磁盘和未受控的缓存状态写入记录。脚本对比相同并行槽位，不设置 RAM 硬上限，正式报告中不能声称严格等内存。`--synthetic` 的输出仅用于接口 smoke，不能当正式性能结论。
