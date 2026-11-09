# ODEBrain 来源审计与 HMS 适配

## 来源和许可

- 官方仓库：[HHJIAnmo/ODEBRAIN](https://github.com/HHJIAnmo/ODEBRAIN)，审计固定 commit `31e251ede69ed82fb47c627d2590286b6376c50a`（2026-11-05 获取）。仓库根目录 `LICENSE` 为 MIT，版权声明为 `Copyright (c) 2025 Haohui JIA`。
- 论文：[arXiv:2602.23285v1](https://arxiv.org/html/2602.23285v1)，ICLR 2026。论文描述了频谱图节点/边编码、原始波形分支、连续状态求解和未来图节点预测。
- 本项目文件 `src/models/odebrain_hms.py` 独立重写了相似计算结构，**不是官方代码的逐行移植，也不是官方在 HMS 上已验证的模型**。没有复制官方源码，所以不需要将其作为 vendored 代码管理；引用和版权信息仍在此保留。

## 实际代码入口与计算图

官方 README 的示例命令使用 `--model_name evobrain`，但实际入口要以 `main.py` 为准：它从 `model/EvoBrain.py` 导入 `EvoBrain_classification`。`main.py` 约 171–175 行创建分类模型；约 335–354 行训练时调用它，并将分类 CE/BCE 与后续节点特征 MSE 相加。`data/dataloader_detection.py` 和 `data/dataloader_prediction.py` 从 TUSZ 衍生 19 通道 EEG、频谱节点属性及相关性/距离图，不能直接读取 HMS。

`model/EvoBrain.py` 中 `EvoBrain.forward`（约 938–983 行）的实际路径：节点特征和边特征分别进入 GRU；节点输入时间序列反转；最后隐藏表示进入双层 GNN；全窗口原始 EEG 进入 CNN；图嵌入与原始嵌入拼接后用 `torchdiffeq.odeint_adjoint(..., method='rk4')` 积分。`EvoBrain_classification.forward`（约 1042–1084 行）对 ODE 状态求时间均值进行分类，同时线性解码图节点特征，取后六帧作为预测。官方 `ODEFunc`（约 763–783 行）在代码中是自主 MLP 向量场；论文 §4.2 的门控/自适应衰减公式未在该实际调用路径中清晰体现，因此不得把论文公式当作此代码已实现的计算图。官方当前代码还有固定 12 帧、6 个预测帧、每秒 200 点、19 通道和若干 200 维/36 维假设，README 参数名称与实际路径需实测核对。

论文附录 A.3 声称训练使用自适应 RK45 和相对容差 `1e-5`，而上述公开代码实际调用固定方法 `rk4`；本项目以审计到的代码调用为参考并明确记录固定步长，不声称复现论文数值设置。论文 §4.3 描述先做图预测预训练再进行下游分类；公开 `main.py` 的当前训练循环则直接将分类损失与预测 MSE 相加。两者不能混作同一已验证流程。

## 模块映射

| 官方模块 | 官方位置 | 本项目部分 | 保留和修改 |
|---|---|---|---|
| 节点、边时间编码 | `model/EvoBrain.py`，`snn_node`、`snn_edge` | `node_encoder`、`edge_encoder` | 保留分支和节点反向时间编码；改为显式 mask 的 `GRUCell`，尺寸由 HMS 配置给出。 |
| 双层 GNN | `model/EvoBrain.py`，`gnnx2` | `graph_conv1/2` | 保留两层节点聚合；改为小电极图的稠密加权消息传递，边历史调制相关性边。没有复制 PyG GCN 权重。 |
| 原始 EEG CNN | `model/EvoBrain.py`，`CNNEncoder` | `raw_encoder` | 保留原始波形分支；改为长度自适应 1D CNN。预测时只允许显式 `observed_eeg`，`eeg` 不会进入严格预测分支。 |
| Neural ODE | `model/EvoBrain.py`，`ODEFunc` + `odeint_adjoint` | `vector_field` + `_integrate` | 保留自主 MLP 向量场和 RK4；改为本地可微显式 RK4，以**秒**为时间单位，当前未使用 adjoint。步长可配置。 |
| 图预测 | `model/EvoBrain.py`，`pred` | `forecast_head` | 预测可观测节点属性 `graph_nodes`，而非额外声称预测边；目标仅用于损失，不作为输入。 |
| 分类 | `model/EvoBrain.py`，`fc` | `classifier` | 新增 HMS 六类头；固定使用从观察截止时刻积分到 +1 秒的节点状态，使分类梯度经过 ODE。该时刻是模型定义，不代表有 +1 秒 EEG 输入。 |

因此当前实验名为 **ODEBrainInspiredHMS**。原论文的完整方法、官方初始化细节、PyG 层、adjoint、论文中的随机性/门控/自适应衰减与 HMS 表现均尚未声称复现。若未来要称为完整官方移植，必须逐项对齐并运行官方 TUSZ 参考路径的数值测试。

## HMS 输入、输出和泄漏边界

`graph_nodes[B,L,N,Fg]` 是按固定电极顺序、每观察帧一个节点频谱属性向量；`graph_adj[B,L,N,N]` 是同一观察帧从既有 EEG 估计的非负相关性代理；`graph_mask[B,L,N]` 标识有效节点。`N` 必须是配置中的 EEG 电极数，EKG 不得混为节点。若 HMS 源文件仅有双极导联，数据适配层须明确导联映射，不能把双极导联无声称作原始电极。该图定义沿用了官方“频谱节点和相关性边”的思路，但具体电极映射、频带计算和图稀疏化属于 HMS 改造；对应数据实现需另行审计。

`future_graph_nodes[B,H,N,Fg]` 和 `future_graph_mask[B,H,N]` **只能**作为目标。模型的编码器、初值和求解器均不读这些张量。`future_times_s[H]` 是从观察截止点起的秒数，例如 `[1,3,5]`。如果有未来目标，普通 `eeg` 键会被忽略；只可用明确裁到切点之前的 `observed_eeg[B,N,Tpast]` 启用原始分支。上游必须保证滤波、插值、标准化、STFT 与图估计也不越过预测切点；模型内部无法从张量本身证明这一点。离线分类无未来目标时可用完整 50 秒上下文 `eeg`，并须在结果中说明包含中央标签之后的信号。

`forward(batch)` 返回 `logits[B,6]`。`forward(batch, return_dynamics=True)` 返回 `(logits, dynamics)`，其中 `z0[B,N,D]`、`trajectory[B,Q,N,D]`、`times_s[Q]`、`vector_field[B,Q,N,D]`、`predicted_graph_nodes[B,H,N,Fg]`、`future_graph_mask` 和 `nfe` 可供训练及可视化。`D=hidden_dim+raw_dim`。轨迹从观察截止时刻 `t=0` 出发，时间是秒；分类固定用 `t=1 秒` 的 ODE 状态节点池化，因此完整图预测时间轴一定包含 1 秒，整窗口只是一组分类概率。`vector_field` 是该轨迹上的高维导数，`nfe` 包含额外可视化求导次数。向量场是自主函数 `f(z)`，没有条件上下文作为单独参数；样本条件仅通过 `z0` 输入。

图属性必须在**训练患者**上拟合归一化量，预测误差使用 mask 及相同归一化。提供的 `masked_forecast_mse` 是基础训练损失；分类建议使用专家软标签 KL 或等价交叉熵。训练流程应先只对训练患者做未来图预测预训练，再微调六分类，可选加预测辅助损失。真实训练、泛化和数值稳定性结论须由实验产物支持，合成前后向测试仅证明接口。

## 当前边界

- 不输出预测邻接矩阵，因此不能展示“预测连接”或边准确率。`edge_weights` 是模型内部聚合权重，也不是因果/解剖连接。
- 目前未提供随机潜变量、论文门控项、自适应衰减、条件向量场、可变每样本预测时间或自适应误差求解器。
- 不将潜在轨迹速度作为脑生理量；投影方向必须对完整高维导数施加同一个训练集拟合线性投影。
- 官方仓库面向 TUSZ；本项目 HMS 六分类效果未知，需与相同图输入的离散 GNN+GRU 及 persistence 预测基线比较。
