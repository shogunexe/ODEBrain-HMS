# 严格预测窗口资格

分类审计使用完整 50 秒 EEG；图预测只观察窗口前 `cut` 秒（默认 5 秒），并在未来 `1,3,5` 秒取图属性监督。因此，分类有效窗口不一定适合预测。预测资格单独保存在 `hms_graph_forecast_cache_v2`，不会修改分类 manifest、患者划分或分类缓存。

构建命令：

```bash
.venv/bin/python -m src.data.graph_cache build \
  --manifest data/curated/split_manifest.csv \
  --output data/training/graph_forecast_v2 \
  --task forecast --cut 5 --horizons 1,3,5 --workers 4
```

构建按 EEG 记录读取一次；每个窗口先独立检查观测段，再检查未来目标。观测段常量、缺失超限或没有有效图节点，以及未来目标常量、缺失超限或任一请求时点没有有效节点，都会得到明确排除原因。未来 EEG 只生成监督目标，绝不进入观测图或观测段标准化。一个记录内的无效窗口不妨碍其余窗口入选。

`index.json` 包含原 manifest SHA256、所有窗口 ID 和逐窗口 `eligibility_reason`。顶层 `forecast_eligibility` 保存全体分类有效窗口及各 split 的 `eligible`、`excluded`、原因计数与按字典序排列的 eligible 窗口 ID 的 SHA256。构建失败后重新运行同一命令，会复用有效分片并原子重建缺失或损坏的分片。旧 v1 预测缓存不能直接复用，应使用新的 v2 目录。

`HMSWindowDataset(..., task="forecast", graph_cache_dir="data/training/graph_forecast_v2")` 自动只保留 eligible 窗口；`dataset.forecast_exclusion_counts` 是该 split 的排除原因计数，`dataset.graph_cache.eligibility_summary` 提供正式评估覆盖门槛。预测评估应声明 `eligible/test` 与原分类测试窗口数，两者不同则结果只能称为预测 eligible 子集结果。没有缓存时，Dataset 不预先筛选，访问无效窗口会抛出带窗口 ID 和原因的 `ValueError`，不应在正式全量训练中使用该路径。
