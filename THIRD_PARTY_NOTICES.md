# Third-party notices

This project calls third-party packages at runtime but does not vendor their
source code, model weights, checkpoints, or datasets. Each component remains
subject to its own license and model/data terms.

| Component | Use in this project | Upstream terms |
| --- | --- | --- |
| Ultralytics YOLO | Person detection and ByteTrack integration | [AGPL-3.0 or Enterprise licensing](https://docs.ultralytics.com/) |
| OpenCLIP | Text and image embeddings | [MIT License](https://github.com/mlfoundations/open_clip/blob/main/LICENSE) |
| Meta SAM 2 / SAM 2.1 | Separate mask-propagation experiment | [Apache License 2.0 for code and checkpoints](https://github.com/facebookresearch/sam2#license) |
| Meta SAM 3 | SAM3 text-prompted video segmentation in `sam3/` | [Official model page and model terms](https://huggingface.co/facebook/sam3) |
| Qwen3-VL-4B-Instruct | Optional independent action-understanding diagnostic | [Official model page and model terms](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| Hugging Face Transformers | SAM3/Qwen model implementations | [Upstream repository and license](https://github.com/huggingface/transformers) |
| MOT17-09 demo excerpt | Demonstration asset only | [CC BY-NC-SA 3.0 dataset terms](https://www.codabench.org/competitions/10049/) |

The files `assets/demo.gif` and `assets/demo.mp4` are derived from the MOT17-09
sequence. They are attributed to MOTChallenge and remain under the applicable
MOTChallenge dataset terms; they are not covered by any license later selected
for the original source code in this repository.

Before production, commercial deployment, or relicensing, review the upstream
terms rather than relying only on this summary.

