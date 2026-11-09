# 服务器环境盘点

检测时间：2026-11-05 11:08 Asia/Shanghai。路径：`/export/home/limianchong/CSproject`。

| 项目 | 实测 |
|---|---|
| OS | Linux 3.10.0-693.el7.x86_64 |
| CPU | 40 逻辑核，2×Intel Xeon Gold 5218R |
| RAM | 251 GiB，总可用约 235 GiB |
| GPU | 2×Tesla V100S-PCIE-32GB，各约 32 GiB，检查时空闲 |
| 驱动 / CUDA driver | 550.163.01 / CUDA 12.4 |
| 默认 Python | 3.6.8；另有 3.8.9，需设置其 lib 路径 |
| 默认 PyTorch | 1.10.1+cu102，CUDA 可用 |
| Java | OpenJDK 8u275 |
| Spark/HDFS/Docker | 初始 PATH 中均未找到；项目 `.venv/bin/spark-submit` 已安装 PySpark 3.4.4。HDFS/Docker 未部署。 |
| 存储 | 工作目录位于共享文件系统，约 50 TiB 可用；不是本地 SSD |
| 网络 | PyPI 可达；Kaggle API 未认证 |
| 凭据 | Kaggle 可认证，账号持有人已接受 HMS 规则且已成功下载元数据与 EEG；GitHub CLI 已登录（密钥未记录） |
| Git | 目录初始不是仓库 |

所有 GPU 资源以实际调度和 `CUDA_VISIBLE_DEVICES` 为准。当前主机单物理机，不将多个进程称作跨物理机集群。

18:22 再查：两张 V100S 各被其他用户进程占用约 5.3 GiB，GPU 计算模式为 `Exclusive_Process`，本项目 CUDA 训练启动失败。项目改用 CPU 做小样本正确性验证，不抢占他人的设备。
