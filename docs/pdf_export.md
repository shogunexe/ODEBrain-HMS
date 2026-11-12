# CentOS 7 本地 PPTX → PDF 导出

## 目前状态（2026-11-08）

服务器原先没有 `libreoffice`/`soffice`，系统为 CentOS 7、glibc 2.17。现已在项目的忽略目录 `artifacts/pdf-tools/` 解包官方 LibreOffice 7.6.7.2 RPM 包，并放入 Noto Sans CJK SC 字体。没有使用 `sudo`，没有改动系统软件或全局字体目录。

本地转换器：

```text
artifacts/pdf-tools/extracted/opt/libreoffice7.6/program/soffice
LibreOffice 7.6.7.2 dd47e4b30cb7dab30588d6c79c651f218165e3c5
```

`artifacts/` 已由项目 `.gitignore` 忽略；工具和字体不进入仓库。LibreOffice 7.6.7.2 是官方归档版，已不再获得安全更新；本流程只对本项目自己生成、可信任的 PPTX 做离线转换。

## 一键使用

在项目根目录运行：

```bash
scripts/export_slides_pdf.sh
```

默认输入为 `slides/ODEBrain_DRAFT_pending_experiments.pptx`，输出为同名 PDF。脚本使用独立临时 LibreOffice profile、项目内字体配置，核对 PPTX/PDF/渲染 PNG 页数，并留下联系表供目视检查。可以指定其他输入和输出目录：

```bash
scripts/export_slides_pdf.sh /absolute/path/to/deck.pptx /absolute/path/to/output_dir
```

输出文件名始终继承输入 PPTX 的 stem；DRAFT 输入会生成 DRAFT PDF。脚本**不会**宣称文件是正式结果，也不能替代真实实验数据审核。

## 本机安装过程和来源

从 [The Document Foundation 官方归档](https://downloadarchive.documentfoundation.org/libreoffice/old/7.6.7.2/rpm/x86_64/) 下载 `LibreOffice_7.6.7.2_Linux_x86-64_rpm.tar.gz`（228 MiB）；[LibreOffice 7.6 文档](https://books.libreoffice.org/en/WG76/WG7600-Preface.html)列出 Linux 内核 ≥3.10、glibc ≥2.17，与本机匹配。下载文件 SHA256：

```text
3dee69f6fec2dafc33b085676dca0add3e79b335c024a32d19b7aeadd19b9f20  LibreOffice_7.6.7.2_Linux_x86-64_rpm.tar.gz
```

中文字体来自 [Noto CJK 官方仓库](https://github.com/notofonts/noto-cjk)，文件 `Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf`，SHA256 为 `2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b`。安装时在项目根目录执行以下命令；`rpm2cpio`、`cpio` 为这台服务器已有的系统命令：

```bash
mkdir -p artifacts/pdf-tools/download artifacts/pdf-tools/extracted artifacts/pdf-tools/fonts
curl -L --fail -o artifacts/pdf-tools/download/LibreOffice_7.6.7.2_Linux_x86-64_rpm.tar.gz \
  https://downloadarchive.documentfoundation.org/libreoffice/old/7.6.7.2/rpm/x86_64/LibreOffice_7.6.7.2_Linux_x86-64_rpm.tar.gz
sha256sum artifacts/pdf-tools/download/LibreOffice_7.6.7.2_Linux_x86-64_rpm.tar.gz
tar -xzf artifacts/pdf-tools/download/LibreOffice_7.6.7.2_Linux_x86-64_rpm.tar.gz -C artifacts/pdf-tools/download
for pkg in artifacts/pdf-tools/download/LibreOffice_7.6.7.2_Linux_x86-64_rpm/RPMS/*.rpm; do
  rpm2cpio "$pkg" | (cd artifacts/pdf-tools/extracted && cpio -idmu --quiet)
done
curl -L --fail -o artifacts/pdf-tools/fonts/NotoSansCJKsc-Regular.otf \
  https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf
sha256sum artifacts/pdf-tools/fonts/NotoSansCJKsc-Regular.otf
```

导出脚本会自动生成 `artifacts/pdf-tools/fonts.conf`，其中包含系统 fontconfig 配置、项目内字体目录和 `Microsoft YaHei` → `Noto Sans CJK SC` 替代规则。验证命令：

```bash
FONTCONFIG_FILE="$PWD/artifacts/pdf-tools/fonts.conf" fc-match 'Microsoft YaHei:lang=zh-cn'
artifacts/pdf-tools/extracted/opt/libreoffice7.6/program/soffice --version
```

预期字体匹配为 `NotoSansCJKsc-Regular.otf`，版本为 `LibreOffice 7.6.7.2`。项目目录迁移后重新运行导出脚本，即会重建字体配置中的路径。

## DRAFT 实测与检查

执行 `scripts/export_slides_pdf.sh` 后，生成：

- `slides/ODEBrain_DRAFT_pending_experiments.pdf`，约 422 KiB。PDF 含转换时间元数据，重复导出的 SHA256 会变化。
- `artifacts/pdf-tools/render/ODEBrain_DRAFT_pending_experiments.knTOALPc/`：24 张页面 PNG 及 `contact.png`；后续运行会生成另一随机后缀目录。

检查结果：PPTX 24 页、PDF 24 页、PNG 24 张；每张 PNG 为 800×450（16:9）；Python 检查所有 PPTX 图形的边界，越过画布者为 0。已目视查看 24 页联系表及第 1、24 页大图，中文可见，没有发现缺页或明显裁切。2026-11-08 重建草稿后，第 23、24 页的链接改置于浅色底板，重新导出并目视检查两页；LibreOffice 仍将链接显示为蓝色，但在浅色背景上清晰可读。

脚本所做的页数、渲染与画布边界检查无法单独保证文字自动换行、公式或每一个链接的最终视觉质量。上述 DRAFT 是历史过程文件。

## 正式版实测与检查（2026-11-12）

[最终 PPTX](../slides/ODEBrain_final.pptx) 经 `scripts/export_slides_pdf.sh` 转为[最终 PDF](../slides/ODEBrain_final.pdf)，PPTX/PDF/渲染 PNG 均为 **24 页**；PDF 页面为 959.981×540 pt、16:9，文件 1,840,825 字节。已目视查看 24 页联系表及主结果、消融、预测、资源、参考页面大图，未见缺页、明显裁切或字体不可读。结果图包含真实全量测试值；第 23 页的公开仓库与权重下载是实际验证过的 HTTPS 地址，`strings` 检查 PDF 中保留了这两个链接。发布资源的匿名下载、SHA256 和逐文件校验证据见[公开资源证据](../results/formal_resources.json)。
