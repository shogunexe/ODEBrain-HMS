# ODEBrain GitHub 轨迹版：正式重跑步骤

本版本是 HMS 适配实现，不是官方 TUSZ 权重。官方 `ODEFunc` 的三层 Tanh MLP、初始化和 `odeint_adjoint(..., method='rk4')`，原始 EEG 的两层二维 CNN，以及 **12 个 ODE 状态时间均值分类**已经在 `src/models/odebrain_hms.py` 的 `github_trajectory_v2` 路径实现。分类 CNN 对 HMS 的 50 秒输入、预测 CNN 对 5 秒输入增加了自适应池化；这是必要的形状适配，不是官方原始 12 秒设置。旧检查点继续走 `legacy_v1`，旧图和旧数字不能视为新版结果。

以下从项目根目录运行。`CUDA_VISIBLE_DEVICES` 应设为用户分配的空闲 GPU 编号。所有新目录含 `github_v2`，不会覆盖旧结果。预训练 batch 64、分类 batch 32；两个独立作业可分别占用一张空闲 A800。初训用原患者划分的 `train`/`val` 选轮次；正式测试只在固定轮数 `train+val` 重训完成后读取一次。`--graph-normalizer` 复用旧预测 run 的训练患者统计量，程序核对清单、图缓存规则及哈希；已确认与本次中断前重新计算出的统计文件 SHA-256 完全一致。若运行中断，在同一训练命令尾部加 `--resume artifacts/runs/.../latest.pt`（初训）或 `--resume`（refit）。

```bash
export LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=1

# 1. 仅用前 5 秒 EEG/图做未来 +1/+3/+5 秒节点属性预测预训练。
.venv/bin/python -m src.cli train \
  --config configs/models/odebrain_github_pretrain.yaml \
  --graph-normalizer artifacts/runs/ode_pretrain/graph_normalizer.npz \
  --run-dir artifacts/runs/odebrain_github_v2_pretrain --device cuda --workers 4

# 2. 从该预训练检查点初始化，用完整 50 秒上下文训练专家软标签六分类。
.venv/bin/python -m src.cli train \
  --config configs/models/odebrain_github_hms.yaml \
  --init-checkpoint artifacts/runs/odebrain_github_v2_pretrain/best.pt \
  --run-dir artifacts/runs/odebrain_github_v2_hms --device cuda --workers 4

# 可在另一张空闲卡上并行：同结构分类模型从随机初始化开始，作预训练消融。
.venv/bin/python -m src.cli train \
  --config configs/models/odebrain_github_hms.yaml \
  --graph-normalizer artifacts/runs/ode_pretrain/graph_normalizer.npz \
  --run-dir artifacts/runs/odebrain_github_v2_scratch --device cuda --workers 4

# 3. 冻结初训最佳轮次后，分别在 train+val 上重训预测和分类。
.venv/bin/python scripts/refit_final.py plan \
  --source-run artifacts/runs/odebrain_github_v2_pretrain
.venv/bin/python scripts/refit_final.py train \
  --source-run artifacts/runs/odebrain_github_v2_pretrain \
  --run-dir artifacts/refit/odebrain_github_v2_pretrain --device cuda --workers 4
.venv/bin/python scripts/refit_final.py train \
  --source-run artifacts/runs/odebrain_github_v2_hms \
  --init-refit-run artifacts/refit/odebrain_github_v2_pretrain \
  --run-dir artifacts/refit/odebrain_github_v2_hms --device cuda --workers 4

# 4. 冻结测试集：与既有四个 refit 基线同一患者划分对比。
.venv/bin/python scripts/refit_final.py evaluate \
  --run-dir artifacts/refit/odebrain_github_v2_hms --device cuda --workers 4
.venv/bin/python scripts/refit_final.py summarize \
  --run tcn=artifacts/refit/tcn \
  --run spectrogram_cnn=artifacts/refit/spectrogram_cnn \
  --run fusion=artifacts/refit/fusion \
  --run graph_gru=artifacts/refit/graph_gru \
  --run odebrain_inspired_hms=artifacts/refit/odebrain_github_v2_hms \
  --results-root results/refit \
  --output results/odebrain_github_v2/final_comparison.json

# 5. 用同一预测目标对比新版 ODE、GraphGRU 与 last-value persistence。
.venv/bin/python scripts/evaluate_forecast.py \
  --ode-run-dir artifacts/refit/odebrain_github_v2_pretrain \
  --gru-run-dir artifacts/refit/graph_gru_pretrain \
  --protocol refit --device cuda --workers 4 \
  --output results/odebrain_github_v2/forecast_comparison.json

# 6. 从 JSON 真值重画分类与预测对比图，从新版检查点导出 ODE 场案例。
.venv/bin/python scripts/supplement_classification_metrics.py \
  --deep results/odebrain_github_v2/final_comparison.json \
  --deep-results-dir results/refit \
  --output results/odebrain_github_v2/classification_weighted_metrics.json
.venv/bin/python scripts/plot_odebrain_github_v2.py \
  --classification results/odebrain_github_v2/final_comparison.json \
  --forecast results/odebrain_github_v2/forecast_comparison.json \
  --weighted results/odebrain_github_v2/classification_weighted_metrics.json \
  --pretrain-run artifacts/runs/odebrain_github_v2_pretrain \
  --classification-run artifacts/runs/odebrain_github_v2_hms \
  --scratch-run artifacts/runs/odebrain_github_v2_scratch \
  --output-dir results/odebrain_github_v2/figures
.venv/bin/python scripts/export_dynamics.py \
  --checkpoint artifacts/refit/odebrain_github_v2_hms/final.pt \
  --manifest data/curated/split_manifest.csv \
  --output results/odebrain_github_v2/dynamics --device cuda
.venv/bin/python scripts/plot_odebrain_six_class_field.py \
  --checkpoint artifacts/refit/odebrain_github_v2_hms/final.pt \
  --output results/odebrain_github_v2/six_class_field --device cuda
.venv/bin/python scripts/plot_odebrain_case_pair_v2.py \
  --checkpoint artifacts/refit/odebrain_github_v2_hms/final.pt \
  --predictions results/refit/odebrain_github_v2_hms/test_predictions.csv \
  --pca results/odebrain_github_v2/six_class_field/six_class_train_pca.npz \
  --output results/odebrain_github_v2/cases --device cuda
```

新版 32 页展示稿由汇报方本地保存，不在公开仓库发布。上面的命令重算其数据和主要图，不会自动覆盖展示稿；`scripts/revise_odebrain_presentation_v2.py` 仅用于旧版 26 页中间稿升级，不能直接对当前 32 页文件运行。

导出的 `ode_field_trajectory.png/.svg` 只显示该样本轨迹上访问过的状态处，训练集 PCA 投影后的局部向量场 `W f(z)`；右图是原始高维向量场在有效节点上的范数。箭头不是全状态空间的全局场，也不能解释为脑信号的因果流。图中 0–1 秒标出分类头所用的 12 点积分区间；这张预测轨迹图的初值只用 0–5 秒 EEG，而正式分类概率另用完整 0–50 秒 EEG 的初值。+1/+3/+5 秒状态用于节点属性预测；这里的 ODE 时间是从 5 秒预测观察截止点开始的潜在状态时间。
