# 公开案例图与最终权重资源包

正式代码仓库中的 `results/dynamics_refit/result_index.json` 和六份 `case.json` 记录了案例来源与图名。PNG、SVG 和模型权重由独立 ZIP 交付。这样可在干净克隆中恢复静态图，并用 SHA256 核对全部文件。资源包不含 Kaggle 原始文件、EEG 窗口数组 `case.npz` 或 PCA 数组 `train_pca.npz`。

## 固定内容

- 六个预设的验证/测试案例，每例六组 PNG 和六组 SVG；共 72 张静态图。
- 五个最终分类模型和 ODE、GraphGRU 两个最终预测预训练模型的 `final.pt`、`config.json`、图归一化参数，以及训练完成、验证来源、逐轮训练历史和窗口 ID 快照记录。共 7 组，每组 8 个文件。
- 消融预训练和三条分类臂的 `best.pt`、生成的 YAML 配置、实际配置、图归一化参数、完成记录、逐轮历史和窗口 ID 快照。资源清单逐一校验完成记录中的权重、源配置和窗口快照 SHA256；其分类测试结果按独立消融协议报告。
- `PUBLIC_RESOURCE_MANIFEST.json` 逐文件记录路径、大小和 SHA256；`PUBLIC_RESOURCE_NOTICE.txt`、根目录的 [LICENSE.md](../LICENSE.md) 与 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) 记录项目权利声明和第三方来源。

本地已冻结 `odebrain_hms_resources_v1.zip`：**20,164,756 字节**、165 个有效载荷文件，SHA256 为 `6ed72bc556f6a9ceb2928bc97c143e703c99b956c1df512418ecf0e1b1f6b6ce`。已经逐文件校验并安装进干净克隆。资源包位于 `artifacts/release/`，该目录不进入 Git；公开下载地址待正式发布后填写并实测。

```bash
export LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-}
.venv/bin/python scripts/package_public_resources.py build
.venv/bin/python scripts/package_public_resources.py verify \
  --archive artifacts/release/odebrain_hms_resources_v1.zip
sha256sum artifacts/release/odebrain_hms_resources_v1.zip
```

下载者在干净克隆中取得 ZIP 后，先把文件校验值与正式发布页核对，再运行：

```bash
python scripts/package_public_resources.py install \
  --archive /path/to/odebrain_hms_resources_v1.zip
```

`install` 会逐文件核对清单；若目标路径已有不同内容，会报错停止。安装后可直接打开 `results/dynamics_refit/cases/formal_test_row_100107/synchronized.svg` 等静态图，或核对 `artifacts/refit/odebrain_inspired_hms/final.pt` 与模型卡中的 SHA256。这里的六例由固定选择规则取得，不能替代全测试集指标。

## 动态界面与完整复算

Streamlit 的样本页和 ODE 动力学页会从 `case.npz` 读取信号与轨迹，公开资源包只提供静态图。要在干净克隆中使用动态界面，须从 [Kaggle HMS 数据页](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data)按规则取得原始数据，运行项目审计、划分、图缓存和 `scripts/export_dynamics.py`，在本地重建这些 NPZ。不能用空白信号代替真实案例。`scripts/reproduce.sh smoke` 在没有 Kaggle 数据时只验证合成接口和代码测试；它不会复算正式指标。

## 来源与再使用

案例图基于 [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data?select=train_eegs) 的选定 EEG 窗口，经过本项目裁剪、图特征计算、模型推断和绘图。数据创建者署名及详细来源见 [第三方声明](../THIRD_PARTY_NOTICES.md)。Kaggle 数据页标注 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)；公开图保留数据来源和改动说明，使用者须遵守该许可的署名与非商业条件。原始数据由 Kaggle 提供，Kaggle 和数据创建者并未认可本项目。ODEBrain 参考实现及其 MIT 许可与本项目改造的关系见 [方法来源](odebrain_adaptation.md)。项目代码与权重仅供检查和复现，未授予一般复制、修改或再分发许可；详见 [项目声明](../LICENSE.md)。模型权重不能用于临床诊断。

公开仓库发布时，资源包应作为固定 tag 的 GitHub Release 附件上传。发布后使用实际的公开地址运行以下命令。它会匿名 HTTP GET 仓库页面及完整附件，比较附件与本地冻结 ZIP 的大小和 SHA256，再核对附件内部逐文件清单；全部通过后才写 `results/formal_resources.json`，供正式 PPT/PDF 使用。

```bash
.venv/bin/python scripts/verify_public_resources.py \
  --repository-url "$REPOSITORY_URL" \
  --weights-url "$WEIGHTS_URL"
```
