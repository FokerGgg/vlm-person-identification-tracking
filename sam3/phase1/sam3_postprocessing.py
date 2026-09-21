"""Portable cv-utils subset used by Transformers SAM3 video.

Native version-1 wheels are not available for every Windows/PyTorch pair.
This adapter implements greedy mask-IoU NMS and 8-connected components with
NumPy/OpenCV. It is a CPU fallback, not an official CUDA kernel or a speedup.
Only the two calls in transformers.models.sam3_video are supported.
"""
import sys


class PortableCVUtils:
    @staticmethod
    def generic_nms(ious, scores, iou_threshold, use_iou_matrix=False):
        import numpy as np
        import torch
        if not use_iou_matrix or ious.shape != (scores.numel(), scores.numel()):
            raise ValueError("Portable SAM3 NMS requires a square IoU matrix")
        overlaps = ious.detach().float().cpu().numpy()
        order = np.argsort(-scores.detach().float().cpu().numpy(), kind="stable")
        keep = []
        while order.size:
            best = int(order[0])
            keep.append(best)
            order = order[1:][overlaps[best, order[1:]] <= iou_threshold]
        return torch.tensor(keep, device=scores.device, dtype=torch.int64)

    @staticmethod
    def cc_2d(mask, get_counts=True):
        import cv2
        import numpy as np
        import torch
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("Portable SAM3 connected components expects N,1,H,W")
        source = mask.detach().to(device="cpu", dtype=torch.uint8).numpy()
        labels, counts = np.zeros(source.shape, np.int32), np.zeros(source.shape, np.int32)
        for i in range(len(source)):
            _, component, stats, _ = cv2.connectedComponentsWithStats(
                np.ascontiguousarray(source[i, 0] != 0, dtype=np.uint8), connectivity=8)
            area = stats[:, cv2.CC_STAT_AREA].copy()
            area[0] = 0
            labels[i, 0], counts[i, 0] = component, area[component]
        label_tensor = torch.from_numpy(labels).to(mask.device)
        return (label_tensor, torch.from_numpy(counts).to(mask.device)) if get_counts else label_tensor


def configure(mode="auto", device="cuda"):
    import torch
    from transformers.models.sam3_video import modeling_sam3_video as module
    if mode == "auto":
        mode = "portable" if sys.platform == "win32" or device == "cpu" else "native"
    if mode == "portable":
        module.cv_utils_kernel = PortableCVUtils()
        return "portable_cpu_nms_and_connected_components"
    if mode != "native":
        raise ValueError("Unknown SAM3 postprocessing mode")
    from kernels import get_kernel
    kernel = get_kernel("kernels-community/cv-utils", version=1)
    # Check execution, not merely successful import: upstream can silently skip
    # failed kernels. In native mode an incompatible build must fail visibly.
    kernel.generic_nms(torch.eye(2, device=device), torch.tensor([.9, .8], device=device),
                       .5, use_iou_matrix=True)
    kernel.cc_2d(torch.ones((1, 1, 8, 8), device=device, dtype=torch.uint8), get_counts=True)
    module.cv_utils_kernel = kernel
    return "native_cv_utils"
