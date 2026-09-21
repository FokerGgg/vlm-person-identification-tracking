# Go2：第一阶段实验

> 跨电脑安装以 [README.md](README.md) 为准。以下实验报告来自原开发电脑；`outputs/`、测试视频、模型缓存和本地标注不随 GitHub 上传，相关路径仅供本地复现实验时使用。

用户视频的首次全片结果见 [DATASET1_RESULTS.md](DATASET1_RESULTS.md)：已完成 46 秒视频的 463 个采样帧推理，换装和后段恢复有成功表现，中间仍有漏跟，当前尚不能实时运行。

## 项目边界与路线建议

相机标签为 RealSense D435，具备彩色和深度采集能力。当前已有的 30 FPS 彩色视频可以用于阶段一；彩色 MP4 不提供原始米制深度。后续应录制同步 RGB、深度、时间戳、深度尺度和相机内参，并标定相机到机器人机身的外参。

建议按以下依赖推进：

1. **语言选人＋持续身份保持**：英文衣着描述选中唯一目标；持续跟踪；脱外套、遮挡和出画后恢复。允许拒绝不确定匹配。
2. **三维定位＋机器人自身定位**：从有效的人体深度估计目标坐标，转换至机器人和地图坐标系，记录带时间戳的轨迹。单帧 bbox 或“画面右侧”不足以确定房间里的位置。
3. **轨迹跟随＋局部规划＋丢失后搜索**：人拐弯后，机器人先沿已观测的可行走轨迹到达拐角/门口，再观察和重新识别。需要地图、里程计/SLAM 和通行区域；长时间不可见时后续路线是未知的，不能无限外推。前往最后已知位置期间仍须避障、限速和超时退出。

避障应从首次实机移动就启用，和阶段二/三联合工作。初次感知置信度低时停止直接追人；后续可由经过约束的导航状态机决定是否沿已知安全路线继续前进。当前代码不会发送任何机器狗运动指令。

来源：[D435 产品页](https://www.realsenseai.com/products/stereo-depth-camera-d435/)、[RealSense 三维投影说明](https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/)、[Nav2 动态跟随示例](https://docs.nav2.org/rolling/tutorials/general_tutorials/navigation2_dynamic_point_following/navigation2_dynamic_point_following/)。上述系统分工是本项目的工程建议。

## 已实现与尚待验证

- 新入口 `phase1_run.py` 保留“视频路径＋英文描述→框选视频”的终端使用方式。
- `sam3` 后端使用 Transformers 的 **facebook/sam3** 流式接口，包括原生短文本提示。此后端不是 SAM3.1，不会把普通 SAM3 权重标成 3.1。
- 两组提示为 `person` 和衣着描述；用它们的掩码 IoU 关联首次匹配。锁定后身份管理只使用通用 person 轨迹和身份记忆，不要求旧衣着继续匹配。两组提示仍会被模型计算，因此初始化语义分支并没有释放显存。
- SAM3 的外观裁剪用最终可见掩码屏蔽背景，并重新计算可见框。人群中 bbox 相交不能判断前后遮挡，因此 SAM3 不再使用 YOLO 的 bbox 重叠否决规则；遮挡拒绝仍依赖掩码是否存在、置信度、运动连续性和外观检查。可见掩码并不等于可靠的遮挡检测器。
- `yolo-clip` 后端是 YOLO11n＋ByteTrack＋OpenCLIP 的对照版本；原来三个脚本保留，便于作旧版本基线。
- 统一状态：SEARCHING / CONFIRMING / TRACKING / OCCLUDED / VERIFYING / LOST / SEARCHING_IDENTITY / AMBIGUOUS。
- 持久目标标识为 `target-1`；它与模型临时 tracker ID 分开。遮挡、身份冲突或丢失时不输出陈旧目标框。恢复需要多帧、相似度和候选差距共同满足条件。
- 有限长度多外观记忆保留首次锚点；高置信连续观察可更新外观。首次语言描述不进入恢复评分或更新门槛。
- 默认特征为整个人像和下半身 OpenCLIP 外观。**下半身特征也受裤子、鞋子和遮挡影响，不是换装不变身份特征。** 它只是用于检验脱外套场景的基线，有换人和记忆污染风险。
- 可接入本地导出的 TorchScript ReID 编码器；目前没有下载、训练或验证 DIFFER/CAL 的权重。不能直接传入它们的任意 `.pth` 检查点。
- SAM3 的时序分割记忆和本项目身份历史提供时序信息。Qwen3-VL-4B 已单独补做动作理解测试，但仍有误判，尚未接入跟踪决策；详见 ACTIONS_RESULTS.md。`appearance_shift_possible` 只是外观分数规则的提示。
- 用户视频 `dataset/1.mp4` 已用于首次真实实验，见下面的运行命令。单个视频的抽查不能代表真实初次选人率、换装成功率或遮挡恢复率；单元测试和示例图推理只验证逻辑/接口。

## 当前环境

本轮验证：40 项逻辑、接口、后处理、历史窗口和视频读写测试通过，包括抽帧后的源帧编号、时间戳和播放速度，以及清理缓存后上游对象指针选择保持一致；依赖一致性检查通过。YOLO/CLIP 和 SAM3 完整权重均已可用。早期示例图记录分别见 `outputs/yolo-clip_smoke.json` 和 `outputs/sam3_smoke.json`，不代表换装能力。

早期示例图测试跳过了后处理。真实视频暴露这一问题后，已安装 `kernels`，但所请求 cv-utils v1 没有本机 Windows/PyTorch 2.11 的兼容构建。因此新增 `phase1/sam3_postprocessing.py`：`--sam3-postprocessing auto` 在 Windows/CPU 上采用 NumPy 贪心掩码 NMS 与 OpenCV 8 连通分量，供上游孔洞填充/碎片清理使用。原生模型权重不变，未修改第三方安装文件。报告明确记录 `portable_cpu_nms_and_connected_components`，它是本项目兼容实现，不是官方 CUDA 性能复现。`native` 模式要求原生内核可执行，否则直接报错。[官方 cv-utils 接口](https://huggingface.co/kernels/kernels-community/cv-utils)

兼容层经重复掩码、空输入、对角连通、独立批次、奇数尺寸、孔洞和孤立碎片测试。尚未与本机不可用的 CUDA 内核进行逐位一致性比较。

当前前向流式模式保留全部 conditioning 记忆和最近 32 个采样帧的 non-conditioning 输出（`--sam3-history-frames`），窗口不得小于检查点所需记忆/对象指针数量。会话子类保留逻辑总帧数，并显式传入连续模型帧号，避免清理旧帧后编号重复或对象指针被错误截断。源视频帧号单独记录。此适配器针对 Transformers 5.16.1 测试，依赖已固定；升级时应重新验证。它只支持当前逐次前向输入，不支持回看旧帧交互修正或反向传播，也不等于保证无限时长内存恒定：conditioning 锚点数量和对象生命周期仍需长时间测试。

工作区在 `D:\capstone`。已创建 Python 3.12 独立环境 `.venv`。当前主机 GPU 是 RTX 5070 Ti，约 16 GB；部署目标是 RTX 4080 Laptop，12 GB。报告会记录实际 GPU，不能混用两台机器的速度。

在 VS Code 的 PowerShell 终端：

```powershell
# 请先进入仓库的 sam3 目录
.\.venv\Scripts\python.exe phase1_run.py --check-env
```

在另一台机器上安装（先有 Python 3.12；不要直接复制 Windows 虚拟环境）：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-phase1.txt
```

`requirements-phase1-lock.txt` 记录本机实际安装的版本。GPU 驱动需与所选 CUDA PyTorch 兼容。

## SAM3 模型授权

当前主机已完成登录，`facebook/sam3` 的配置访问与完整权重下载均已成功。换到其他主机时，需在 [模型页面](https://huggingface.co/facebook/sam3)获得访问权限，并在该主机交互式登录：

```powershell
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\python.exe phase1_prepare.py --backend sam3 --config-only
.\.venv\Scripts\python.exe phase1_prepare.py --backend sam3
```

不要将 token 写在命令参数、代码、报告或聊天中。`config-only` 只检查配置访问；下一条才下载模型并做示例图接口验证。模型缓存位于项目的 `.cache` 内。

## 放入视频后运行

当前视频是 `D:\capstone\dataset\1.mp4`，960×544，1387 帧，约 46.3 秒。目标最初穿棕色、黑白拼色外套和米色裤子。按下面的方式运行，输出目录要使用一个新名字：

```powershell
.\.venv\Scripts\python.exe phase1_run.py --video dataset/1.mp4 --query "person wearing a brown jacket with black and white panels and beige pants" --backend sam3 --max-objects 24 --frame-stride 3 --max-frames 0 --output outputs/dataset1_sam3_repeat
```

`--frame-stride 3` 每三帧处理一帧，源视频约 30 FPS 时相当于约 10 FPS 采样；并非模型能以 10 FPS 实时推理。默认 stride 为 1，逐帧处理。`--max-frames` 限制处理的采样帧数，默认 900；`--max-frames 0` 处理到结束。

不能据此承诺无限时间流式运行：SAM3 的内部跟踪状态还需要长时间测试。SAM3 默认总对象上限为 8，包含两组提示产生的对象。本视频即使设为 24 也持续触发容量警告；拥挤场景中未进入候选集合的新人可能无法恢复。简单增加容量会增加耗时和显存，还需要处理陈旧/重复轨迹的生命周期。

程序逐帧向模型输入视频，不预读未来帧，不做双向传播，不在每段自动重新按衣服锁定。Hugging Face 提醒，流式模式会关闭依赖未来帧的 hotstart 启发式，可能比预加载视频模式产生更多误检/重复轨迹，所以两者不能当成同样的实机条件。[官方流式文档](https://huggingface.co/docs/transformers/model_doc/sam3_video)

输出目录必须是新目录，避免覆盖实验。输出：

- `annotated.mp4`：当前确认目标为红框，其他候选为蓝框；不确定状态不标成目标。
- `predictions.jsonl`：每帧时间、状态、目标/临时编号、归一化 xyxy 框、候选和耗时。
- `report.json`：模型配置、版本、硬件、帧数、延迟分位数、GPU 峰值显存。Python 能正常处理的中断/异常会保存部分报告；Windows 强制终止进程可能来不及写报告或完成视频封装，不能将这样的文件作为完成结果。

抽帧输出 FPS 为输入 FPS / stride，保持原始播放速度，不补造未处理帧的预测；真实处理吞吐另行统计。JSONL 的帧号是原视频帧号。时间戳按 `frame_index / source_fps` 计算，假设恒定帧率。当前片段抽查的解码 PTS 与该估计相近，但严格时间测量应使用原始 PTS；变帧率视频应先转恒定帧率。`--fps 30` 仅在确定源视频真实帧率时使用，不能用来假造性能。

可将完成的实验导出为只显示目标的 H.264 视频，方便播放；不会重新推理或修改预测：

```powershell
.\.venv\Scripts\python.exe phase1_render.py --run outputs/dataset1_sam3_repeat --output outputs/dataset1_sam3_repeat/view
```

## 标注与评估

将同一人的真实框作为标注，脱衣后身份不变。完整不可见/无法定位时为 `null`；“没有标注”应省略该帧，不应写 `null`。

可先给短片做人工标注：

```powershell
.\.venv\Scripts\python.exe phase1_label.py --video data/videos/test01.mp4 --output data/labels/test01.jsonl --end 900 --scenario jacket_removal
```

窗口快捷键：B 画框；A 目标不可见；C 明确接受上一帧框；U 撤销；Q 保存退出。可加 `--predictions outputs/test01_sam3/predictions.jsonl` 显示预测作为橙色建议框，P 明确接受，错误建议必须重画；不会自动把模型结果当真值。

`test01.mp4` 在此是其他新视频的示例占位符。当前视频的初步关键帧草稿为 `data/labels/dataset1_review_draft.jsonl`：由助手目视原视频帧绘制，19 帧，尚未由用户复核，不能当成稠密真值或独立测试集。如果运行 stride=3，标注器使用 `--step 3`，评估器使用 `--frame-step 3`。只有按这个步长连续标注的事件区间才能估计采样时间尺度上的恢复延迟；稀疏关键帧不提供精确恢复时间。

标注格式（坐标归一化）：

```json
{"frame_index": 0, "target_box": [0.2, 0.1, 0.5, 0.9], "scenario": "normal"}
{"frame_index": 1, "target_box": null, "scenario": "occlusion"}
```

```powershell
.\.venv\Scripts\python.exe phase1_evaluate.py --predictions outputs/test01_sam3/predictions.jsonl --labels data/labels/test01.jsonl --output outputs/test01_sam3/evaluation.json
```

指标包括：可见目标框成功率、目标不可见时错误跟随比例、错误框/漏跟帧数、首次输出是否正确、重现后的恢复时间和分场景统计。默认 IoU>=0.5 为定位正确。评估文件必须覆盖本次预测中存在的帧；标注连续事件区间才能测可靠恢复时间。

这些是单目标定位指标。真正的 IDSW/IDF1/HOTA 需要全体行人的身份标注及标准评测工具，本工具不冒充这些指标。不可见时始终拒绝不会获得“成功率高”的误导结论：可见漏跟和恢复失败会分别记录。

## 必测场景与实验控制

至少准备：唯一衣着目标、多个相同衣着候选、转身/背对、在可见过程中脱外套、交叉遮挡、短时出画再入画、遮挡后换装重现、相似衣着路人接近。分别统计，不把“持续可见脱衣”和“消失后换装”混成一个任务。

按**人和原始视频**划分调参集/测试集，避免把同一视频的相邻片段分到两边。阈值只是初始实验值，应在调参集确定并冻结，再评测测试集。不能以模型自己输出的相似度替代真实身份正确性。

依次比较：旧脚本、统一新框架的 yolo-clip、统一新框架的 SAM3、SAM3＋经过验证的换装 ReID。旧脚本没有统一日志，需补充标注评估后才能数值对比。用 `--no-recovery` 做去掉身份恢复的消融；SAM3/CLIP 语义分数尺度不同，默认阈值分别设置，必须报告。

## 可选换装 ReID 编码器接口

使用 `--reid-encoder path/to/encoder.ts`。模型输入为 RGB float32 `[N,3,256,128]`，已做 ImageNet mean/std 归一化；输出是非零、有限的 `[N,D]` 特征张量。若模型原训练预处理不同，应在导出封装中适配，不能盲目使用这个约定。当前实现以它替换身份特征，OpenCLIP 仍用于 yolo-clip 后端的英文选人。

[DIFFER 官方代码](https://github.com/xliangp/DIFFER)提供专门的换装 ReID 方法；适配其训练权重并在机器人视角验证是后续工作，不是当前已完成的能力。

## 验证命令

异常分割修复与实测结果见 [PHASE1_UPDATE.md](PHASE1_UPDATE.md)；已补做的通用视频 VLM 诊断及误判见 [ACTIONS_RESULTS.md](ACTIONS_RESULTS.md)。`phase1_run.py` 默认对已锁定人物启用保守远处孤立块清理，`--sam3-mask-cleanup off` 可关闭；`--save-observations` 保存外观特征和原始目标掩码，`phase1_replay.py` 可在同一批观测上比较身份逻辑。`phase1_actions.py` 为独立的本地动作分析入口，默认自由描述，尚未参与身份决策。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe phase1_prepare.py --backend yolo-clip
```

最后一条默认重复输入依赖包自带示例图三次，验证模型调用和状态接口，不检验真实运动、换装或跟随精度。SAM3 的同类验证可用 `phase1_prepare.py --backend sam3 --frames 3`。
