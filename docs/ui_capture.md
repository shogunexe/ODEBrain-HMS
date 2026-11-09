# Streamlit 界面截图

在项目根目录使用 Python 3.8 虚拟环境。服务器的 `/usr/bin/google-chrome` 为 Chrome 78，没有 ChromeDriver；`scripts/capture_ui.py` 通过 Chrome DevTools Protocol 控制无头浏览器，在 Streamlit 页面标题、所需 Plotly SVG 和字体实际加载且布局连续稳定后截取 PNG。截图会检查尺寸、PNG 格式、内容对比度，拒绝运行中骨架和空白图。该 Chrome 版本缺少 Streamlit 前端用到的 `String.replaceAll`，工具仅在本次隔离浏览器会话中提前注入兼容补丁，不修改应用代码。

首次准备：

```bash
LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-} \
  .venv/bin/pip install websocket-client==1.6.4
```

中文字体也是必要条件；没有中文字体会显示方框，工具会拒绝截图。当前服务器已在用户字体目录安装 `Noto Sans CJK SC`。新机器可从 Noto 官方字体仓库获取 `NotoSansCJKsc-Regular.otf`，放入 `~/.local/share/fonts/` 后运行 `fc-cache -f ~/.local/share/fonts`，并用 `fc-list :lang=zh family` 核对。

已有本机 Streamlit 服务（当前 tmux `hms_app`，端口 8501）：

```bash
LD_LIBRARY_PATH=/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-} \
  .venv/bin/python scripts/capture_ui.py \
  --page 'ODE 动力学'
```

若端口没有服务，可加 `--launch-app`；工具自行启动应用并设置 `--server.fileWatcherType none`，结束后关闭该子进程。默认截图为 1600×1000，默认捕获所选页所需的最少已绘制图表：数据概览 3、样本分析 3、模型对比 2、ODE 动力学 3、大数据性能 1。`--min-charts` 只能提高门槛，不能降低。`--timeout` 默认 75 秒。需要其他页面或服务地址时使用 `--page` 和 `--url`。

当前 manifest 与原始 EEG 已更新，但现有 ODE 案例和模型结果仍来自较早的真实小子集，且正式交付输入尚不完整。截图默认写入 `artifacts/smoke/ui_capture.png` 和同目录的 `ui_capture.json`；JSON 中 `experiment_status="smoke"`、实际 manifest split hash、PNG SHA256、Chrome 版本、中文字体验证、绘制图数量和未通过正式门禁的原因都可复查。工具会在 smoke PNG 右上角加 `REAL HMS SMOKE · 非正式证据`，`artifacts/smoke/ui_ode_capture.png` 的轨迹图还带应用自身水印。**这些图不能用于正式 PPT 证据页。**

正式状态须由 `results/formal_delivery_inputs.json` 指向的完整正式产物通过 `scripts/prepare_formal_slides.py` 的数据审计、患者划分、Spark、深度模型、对照、动力学等门禁；工具跳过其中需要截图本身的 UI 门禁，以避免循环依赖。同时实际 `split_manifest.csv`、其摘要哈希和动力学索引 split hash 必须一致。正式通过后，默认写 `results/formal_ui.png` 与 `results/formal_ui_capture.json`，后者的 `image` 和 `split_hash` 可供正式幻灯片准备器读取。若缺少任一正式结果或发现 smoke/synthetic，状态只能是 smoke；程序拒绝把 smoke 输出写成 `formal_*` 文件名。

若截图失败，工具不写 PNG 或元数据。常见原因是服务未启动、页面没有所需图、Chrome 前端报错、中文字体缺失、正式结果哈希不一致。截图工具只读取实验文件，不训练模型，也不把页面存在当成正式模型结果。
