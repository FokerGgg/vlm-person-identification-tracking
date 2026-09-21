# 异常分割修复与动作理解补测

> 跨电脑安装以 [README.md](README.md) 为准。以下实验报告来自原开发电脑；`outputs/`、测试视频、模型缓存和本地标注不随 GitHub 上传，相关路径仅供本地复现实验时使用。

原实现完成了 SAM3 语言选人、时序跟踪和 OpenCLIP 外观记忆，但没有接入通用视频 VLM。因此首轮跟住脱外套后的目标，不能证明模型理解了脱外套动作。动作理解是第一阶段原要求的一部分，之前没有完成；现在已补测，结果仍不足以认定该能力可靠。

## 异常分割的原因与修复

源帧 819（27.36 秒），原始掩码里人体主体仍正确，面积为 9040 像素；左侧边缘多出三个不相连的区域，面积分别为 325、72、3 像素。计算所有非零像素的外接框时，这 400 个远处噪点把人物框撑大了。

旧身份逻辑拒绝坏框后，直接进入全局重新识别；尽管下一帧原编号和人体框恢复，拥挤场景下相似候选的分数差仍让程序长时间不输出目标。

已增加两层处理：

1. 短暂中断后，在最后可信观测的 0.5 秒内，原编号同时满足位置重叠、外观及连续两次观测时可以恢复。附近有相似且重叠的候选时仍拒绝。异常观测不更新身份记忆，也不输出旧框冒充当前检测。
2. 对已经选中的人物做保守掩码清理：主体须占总面积至少 90%；仅移除面积不超过主体 5%、且与主体框距离超过主体最大边长一半的孤立块。附近的手/包碎片及多个大区域保留。不会补画被遮挡的身体，不修改 SAM3 内部时序状态。

这组阈值是在本开发视频上验证的启发式规则，仍需独立视频测试。并非从此所有错误分割都能消除。

## 对照结果和证据范围

三种结果都保持原英文查询、原源帧顺序、每三帧取一帧，共 463 个采样帧；无人工重新指定目标。

| 版本 | 27.36 秒附近 | 后续 |
| --- | --- | --- |
| 旧版 | 中断，一直到约 38.98 秒才恢复 | 约 30、32 秒人仍可见时漏跟 |
| 短暂中断修复 | 27.36 秒拒绝坏框，27.56 秒恢复 | 跟到约 32.47 秒的真实遮挡，约 38.98 秒恢复 |
| 加保守掩码清理 | 该处连续跟踪，没有这次中断 | 真实遮挡阶段仍不输出确认目标 |

短暂中断修复运行了完整视频的 SAM3 推理。用该次保存的掩码/外观特征重放旧身份逻辑，463 帧的身份决策与旧版逐项一致，复现了原漏跟。

掩码清理在上述全片缓存上验证：只改变源帧 819 的 400 个噪点，对这一帧重新计算人体框和 OpenCLIP 特征，再从头按时序重放身份决策；其余模型观测复用。它不是另一次 SAM3 全片推理，也不能用来报新的推理速度。清理已接入正常运行入口，后续新视频默认启用，可用 `--sam3-mask-cleanup off` 做对照。

同一份 19 帧目视标注草稿中，16 帧可见目标现在全部定位正确（IoU≥0.5），3 帧不可定位时均无确认框。额外检查了异常前后、32.5 秒附近遮挡和重现后的画面。草稿未由用户复核，覆盖率约 4.1%；这不是全片 100% 准确率，未证明零身份切换或任意消失后的恢复能力。

通过 40 项测试，覆盖坏框、不复用过期框、相似人交叉、原编号被复用、记忆保护、孤立块/身体碎片、真实视频读写、观测重放一致性、VLM 采样时间等。当前主机 RTX 5070 Ti 的完整推理约 0.60 采样帧/秒，仍不实时；这不是 4080 Laptop 性能。

## 查看结果

- 最终清理版视频（缓存模型输出重放）（本地文件：`outputs/dataset1_sam3_mask_refined/view/target_only.mp4`）
- 短暂中断修复视频（完整 SAM3 实跑）（本地文件：`outputs/dataset1_sam3_glitch_fix/view/target_only.mp4`）
- 27.36 秒掩码原始异常与恢复（本地文件：`outputs/dataset1_sam3_glitch_fix/diagnostics/mask_check_0.jpg`）
- 32.47 秒真实遮挡（本地文件：`outputs/dataset1_sam3_glitch_fix/diagnostics/mask_check_1.jpg`）
- 最终重放报告（本地文件：`outputs/dataset1_sam3_mask_refined/report.json`）
- [动作理解测试结果](ACTIONS_RESULTS.md)

## 复现

正常完整实验，保存观测便于比较后续改动：

```powershell
.\.venv\Scripts\python.exe phase1_run.py --video dataset/1.mp4 --query "person wearing a brown jacket with black and white panels and beige pants" --backend sam3 --max-objects 24 --frame-stride 3 --max-frames 0 --save-observations
```

动作测试单独运行，输出目录需要使用新名称；模型已经缓存：

```powershell
.\.venv\Scripts\python.exe phase1_actions.py --video dataset/1.mp4 --predictions outputs/dataset1_sam3_streaming/predictions.jsonl --cases examples/dataset1_diagnostic.json --prompt-mode open --target-view --no-reference-image --output outputs/actions_new
```

VLM 当前是独立诊断工具，其文本尚未用于改写目标身份。下一步需要独立视频和更可靠的动作/衣物区分结果，才能评估将低频视频理解接入事件记忆。跟随控制、空间坐标及避障仍属后续阶段。
