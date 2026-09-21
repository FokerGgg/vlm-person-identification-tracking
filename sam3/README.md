# SAM3 英文描述选人和视频跟踪

第一阶段离线实验：**英文衣着描述 → SAM3 选人/分割/视频跟踪 → OpenCLIP 外观记忆与身份确认 → 框选视频和实验记录**。

该目录可以独立运行。主流程使用 `facebook/sam3` 和 OpenCLIP `ViT-B-32 / laion2b_s34b_b79k`，包含短暂异常后的恢复、远处孤立掩码清理及有限历史窗口。`Qwen/Qwen3-VL-4B-Instruct` 是可选的独立动作分析实验，尚未参与跟踪决策。

当前仍是研究原型，未实现机器人控制、三维坐标或避障，也未达到实时跟随速度。单个开发视频上的改善不能代表可靠的换装重识别；Qwen 动作测试仍存在外套与包的混淆。

## 1. 在另一台电脑下载项目

先安装 Git 和 **Python 3.12**。Windows PowerShell 中执行：

```powershell
git clone https://github.com/FokerGgg/vlm-person-identification-tracking.git
cd vlm-person-identification-tracking\sam3
```

也可以在 GitHub 点击 **Code → Download ZIP**，解压后用 VS Code 打开其中的 `sam3` 文件夹。

本目录的安装步骤与仓库根目录旧 YOLO 版本独立。之后的命令都在 `sam3` 目录执行。已有旧克隆时，先在仓库目录执行 `git pull`，再进入 `sam3`。

## 2. 安装环境

推荐在 NVIDIA 显卡电脑运行。已验证组合：Python 3.12、PyTorch 2.11.0 / CUDA 12.8、torchvision 0.26.0、Transformers 5.16.1。显卡驱动需要支持所选 PyTorch CUDA 版本。

可以运行随项目提供的安装脚本（只建立当前目录的环境和依赖，模型另行准备）：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

或者逐条执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-phase1.txt -c requirements-phase1-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe phase1_run.py --check-env
```

不需要激活虚拟环境；命令已明确使用 `.venv` 内的 Python。不要直接复制其他电脑的 `.venv`。VS Code 的 Python 解释器选本目录的 `.venv\Scripts\python.exe`。

Linux/macOS 用户需要自行创建 Python 3.12 环境，路径改为 `.venv/bin/python` 并选择与设备匹配的 PyTorch；本发布包的安装脚本只针对 Windows。默认 SAM3 后处理在 Windows 使用 CPU 兼容实现。

## 3. 登录 Hugging Face 并准备 SAM3

在 [facebook/sam3 模型页](https://huggingface.co/facebook/sam3)申请模型访问权限。权限属于 Hugging Face 账号；另一台电脑使用同一已获批账号登录即可。

```powershell
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\python.exe phase1_prepare.py --backend sam3 --config-only
.\.venv\Scripts\python.exe phase1_prepare.py --backend sam3
```

第一条按终端提示完成登录；第二条检查访问权限；第三条下载 SAM3 和 OpenCLIP，并用依赖包示例图验证调用。SAM3 权重约 3.2 GiB，OpenCLIP 约 0.56 GiB，首次准备需要网络。示例图测试不代表真实跟踪准确率。

模型缓存默认写入当前目录的 `.cache/huggingface/hub`。登录凭据保存在 Hugging Face 的标准用户配置位置，不随代码上传。

## 4. 放入自己的视频并运行

```powershell
New-Item -ItemType Directory -Force dataset
```

将测试视频复制为 `dataset\1.mp4`，或在命令中换成自己的完整路径。原始测试视频未上传到这个仓库。

先跑短片，检查是否选对目标、显存是否够用：

```powershell
.\.venv\Scripts\python.exe phase1_run.py --video dataset/1.mp4 --query "person wearing a brown jacket with black and white panels and beige pants" --backend sam3 --max-objects 8 --frame-stride 3 --max-frames 60 --output outputs/first_test
```

将英文描述换成自己视频中目标人物的初始衣着。输出目录必须是新目录，重复运行时更换名称；省略 `--output` 会自动生成新的结果目录。

验证整段视频（拥挤场景可增加候选容量，但会增加计算及显存需求）：

```powershell
.\.venv\Scripts\python.exe phase1_run.py --video dataset/1.mp4 --query "person wearing a brown jacket with black and white panels and beige pants" --backend sam3 --max-objects 24 --frame-stride 3 --max-frames 0 --save-observations --output outputs/full_test
```

`--max-frames 0` 表示处理到结尾；`--frame-stride 3` 每三帧处理一帧。输出视频保持源时间播放，播放 FPS 不是实际推理速度。12GB 的 RTX 4080 Laptop 应先做短片验证；之前完整测试的实际显卡为 RTX 5070 Ti，不能混用速度和显存结论。

主要产物：

- `annotated.mp4`：红框为确认目标，蓝框为其他候选。
- `predictions.jsonl`：逐帧框、身份状态、候选和时间。
- `report.json`：环境、参数、吞吐和显存记录。
- 使用 `--save-observations` 时增加外观特征和原始目标掩码，用于调试与重放。

导出只显示目标的 H.264 视频：

```powershell
.\.venv\Scripts\python.exe phase1_render.py --run outputs/full_test --output outputs/full_test/view
```

结果位于 `outputs/full_test/view/target_only.mp4`。

## 5. 可选：独立动作理解实验

只做 SAM3 跟踪时可以跳过这一步。Qwen 权重额外约 8.3 GiB，当前脚本在本地单独运行；视频不会上传到推理服务。

```powershell
.\.venv\Scripts\hf.exe download Qwen/Qwen3-VL-4B-Instruct --cache-dir .cache/huggingface/hub --include "*.json" "*.safetensors" "*.txt" "*.jinja"
```

复制 `examples/dataset1_diagnostic.json` 并按视频实际长度修改各段 `start`、`end`。这些时间是示例开发视频的片段；不要直接套用到其他视频。`expected_for_review_only` 字段仅供事后核对，不进入模型提示。

```powershell
.\.venv\Scripts\python.exe phase1_actions.py --video dataset/1.mp4 --predictions outputs/full_test/predictions.jsonl --cases examples/dataset1_diagnostic.json --target-view --no-reference-image --output outputs/actions_test
```

脚本使用已确认的跟踪框定位目标，再分析片段。没有可靠目标框或视频短于配置区间时会拒绝该输入。结果与局限见 [ACTIONS_RESULTS.md](ACTIONS_RESULTS.md)。

## 验证、文件和已有结果

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

当前 40 项测试不需要下载模型权重，也不需要提供视频；使用合成观测和临时短视频检查逻辑、接口、采样、掩码清理等。通过测试不等于在新视频上达到跟踪准确率。

| 文件 | 用途 |
| --- | --- |
| `phase1_run.py` | 主跟踪入口 |
| `phase1_prepare.py` | 模型下载、访问和示例图检查 |
| `phase1_render.py` | 导出便于播放的目标视频 |
| `phase1_label.py`、`phase1_evaluate.py` | 人工标注和稀疏/连续标注评估 |
| `phase1_replay.py` | 同一批观测上的身份逻辑重放 |
| `phase1_actions.py` | 独立视频 VLM 动作诊断 |
| `phase1/` | 模型适配、身份记忆和掩码/历史处理 |
| `tests/` | 40 项自动检查 |

[实现细节](PHASE1.md) · [异常分割修复记录](PHASE1_UPDATE.md) · [首轮实验记录](DATASET1_RESULTS.md)

历史报告中的 `outputs/`、`dataset/` 和本地标注路径只对应原开发电脑，相关视频与运行产物未随代码上传。模型、虚拟环境、数据集、下载缓存都由 `.gitignore` 排除，另一台电脑克隆时只获取源代码、测试和文档。

可选 `yolo-clip` 对照后端需要自行准备本地 `yolo11n.pt` 并通过 `--yolo` 指定路径；SAM3 主流程不需要 YOLO 权重或 `third_party/sam2`。
