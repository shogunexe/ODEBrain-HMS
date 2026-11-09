# HMS 深度模型卡生成

`scripts/write_model_cards.py` 只接收完整 HMS 训练集上完成的五个分类模型运行。输入是 `artifacts/runs/` 中的实际训练目录、全量患者划分、正式 `val`/`test` 指标和逐窗口预测；输出为 `docs/model_cards/` 下各模型的 Markdown 与 JSON。脚本不会上传 checkpoint，也不会填入权重链接。

全量训练与两次评估完成后，从项目根目录运行：

```bash
LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-} \
  .venv/bin/python scripts/write_model_cards.py \
  --run tcn=artifacts/runs/tcn \
  --run spectrogram_cnn=artifacts/runs/spectrogram_cnn \
  --run fusion=artifacts/runs/fusion \
  --run graph_gru=artifacts/runs/graph_gru \
  --run odebrain_inspired_hms=artifacts/runs/ode_finetune
```

如果实际 run 目录名称不同，替换等号右侧。脚本默认读取 `data/curated/split_manifest.csv`、`data/raw/train.csv`、`data/curated/split_manifest.json` 和 `results/<run 目录名>/{val,test}_metrics.json`；可用 `--manifest`、`--metadata`、`--split-summary`、`--results-root`、`--output-dir` 指定路径。`--run` 必须恰好提供五个模型，每个一次。

## 校验与拒绝条件

- Manifest 覆盖完整 metadata 的 EEG ID 和标注行；所有 EEG 文件到位，质量审计完整，训练、验证、测试患者互斥。split summary 的哈希和计数必须一致。
- 每个 run 的 `config.json`、`completed.json`、`best.pt`、`graph_normalizer.npz`、`window_snapshot.json` 均存在。source YAML、manifest、冻结窗口清单和最佳 checkpoint 的 SHA256 必须与训练完成记录一致；checkpoint 内配置与磁盘配置完全一致，模型参数可严格装载。
- 如从预训练初始化，源 checkpoint 必须存在、SHA256 相符，且源配置也来自同一完整 manifest。
- 正式 `val` 和 `test` 指标必须分别来自完整划分，`smoke_subset=false`、`metadata_eeg_coverage=1.0`、split hash、run 目录、模型名、窗口数和患者数都要一致；逐窗口预测 CSV 的窗口和患者 ID 集合必须精确等于冻结划分。
- 缺任一正式结果、任一 smoke/synthetic 标记、非有限指标或不匹配哈希，整批模型卡均拒绝写入。所有模型都通过后才生成文件。

模型卡写明 19 个 EEG 导联及输入模态、50 秒窗口与中央 20–30 秒标签区、预处理、训练与评估数据、KL 等指标、限制、权重 SHA256 和本地复现命令。ODE 卡明确为本项目的 ODEBrain-inspired HMS 改造。JSON 卡包含输入文件与逐窗口预测的 SHA256，便于交付时复核。生成前不应将当前 smoke 指标写进正式卡。
