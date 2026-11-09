# 结果浏览与动力学案例

从仓库根目录运行：

```bash
streamlit run app/main.py -- --results results
```

界面固定为数据概览、样本分析、模型对比、ODE 动力学、大数据性能五页。它只读取结果文件，不执行训练或推理。每个 Plotly 图可按需生成当前视图的 PNG/SVG，案例也提供预计算静态图。缺少结果时显示“未生成”；带 `synthetic=true` 的性能数据明确标成 smoke 数据。

真实 ODE 案例可由联合训练 checkpoint，或先预测预训练再分类微调的 checkpoint 导出：

```bash
python scripts/export_dynamics.py \
  --checkpoint artifacts/runs/<finetuned_run>/best.pt \
  --manifest data/curated/split_manifest.csv \
  --graph-normalizer artifacts/runs/<finetuned_run>/graph_normalizer.npz \
  --output results/dynamics
```

`result_index.json` 指向各案例的 `case.json`、`case.npz` 和六组 PNG/SVG。训练集窗口按 ID 排序，默认最多 256 个用于拟合该模型共用 PCA；验证和测试不参与拟合。案例在验证集预先规定低/高分歧、典型正确和典型错误四种规则，对测试集应用同一规则。PCA 输入为每个时点有效节点隐状态均值；方向箭头对真实高维向量场先取相同节点均值，再做线性投影。速度和积分长度则对全部有效节点计算，时间单位是物理秒。不同模型各自拟合的 PCA 坐标不可直接比较。

导出器拒绝缺少真实 checkpoint、患者划分 manifest、训练集标准化器或原始 EEG 的运行。分类头必须使用正的分类损失训练；预测头必须在同一 checkpoint 使用正的预测损失训练，或与带哈希记录的预测预训练 checkpoint 中的头权重逐项相同。预测头在分类微调后虽保留权重，其输入表征可能变化，因此每个案例仍展示实际未来误差。每个案例做两次推理：完整 0–50 秒窗口用于离线分类，独立的严格预测分支只读取前 5 秒，用于轨迹与未来节点属性预测。未来图没有预测邻接边。HMS 专家标签只覆盖原窗口 20–30 秒；分类概率是一组值，静态图中的空白时间不表示持续标签。

导出器依赖 PyTorch、NumPy、Pandas、SciPy、Matplotlib；界面另依赖 Streamlit 和 Plotly。可从 `src.cli` 的训练输出使用 `best.pt` 与同一运行目录的训练集图特征标准化器。
