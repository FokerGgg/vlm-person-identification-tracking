"""Manual-ROI SAM 2.1 video-mask propagation experiment.

This experiment is intentionally separate from the semantic YOLO/OpenCLIP
pipeline. It requires an editable installation of Meta's SAM 2 repository and
a SAM 2.1 checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track a manually selected person with SAM 2.1",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Directory containing numerically named image frames",
    )
    parser.add_argument(
        "--sam2-dir",
        type=Path,
        required=True,
        help="Local clone of https://github.com/facebookresearch/sam2",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/sam2.1_hiera_small.pt"),
        help="Absolute path or path relative to --sam2-dir",
    )
    parser.add_argument(
        "--model-config",
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="SAM 2 Hydra model configuration name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sam21_person_tracking.mp4"),
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--object-id", type=int, default=1)
    return parser.parse_args()


def frame_sort_key(path: Path) -> tuple[int, int | str]:
    """Sort numeric frame stems numerically and other names lexicographically."""
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.name


def resolve_checkpoint(checkpoint: Path, sam2_dir: Path) -> Path:
    if checkpoint.is_absolute():
        return checkpoint
    return sam2_dir / checkpoint


def draw_mask_and_box(
    frame: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    if mask is None or not mask.any():
        cv2.putText(
            frame,
            "TARGET NOT VISIBLE",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return frame

    overlay = frame.copy()
    overlay[mask] = (
        0.55 * overlay[mask]
        + 0.45 * np.array([0, 255, 255])
    ).astype(np.uint8)

    ys, xs = np.where(mask)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        overlay,
        "SAM2 TARGET",
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    return overlay


def main() -> None:
    args = parse_args()
    frames_dir = args.frames_dir.resolve()
    sam2_dir = args.sam2_dir.resolve()
    checkpoint = resolve_checkpoint(args.checkpoint, sam2_dir).resolve()
    output = args.output.resolve()

    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {frames_dir}")
    if not sam2_dir.is_dir():
        raise FileNotFoundError(f"SAM 2 directory not found: {sam2_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM 2 checkpoint not found: {checkpoint}")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device != "cuda":
        raise RuntimeError("This SAM 2.1 experiment requires CUDA")

    frame_names = sorted(
        (
            path
            for path in frames_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        key=frame_sort_key,
    )
    if not frame_names:
        raise RuntimeError(f"No image frames found in: {frames_dir}")

    first_frame = cv2.imread(str(frame_names[0]))
    if first_frame is None:
        raise RuntimeError(f"Cannot read first frame: {frame_names[0]}")

    height, width = first_frame.shape[:2]
    print(f"Frames: {len(frame_names)}")
    print(f"Resolution: {width} x {height}")

    display_frame = first_frame.copy()
    cv2.putText(
        display_frame,
        "Select target person, then press ENTER",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    roi = cv2.selectROI(
        "SAM2 Target Selection",
        display_frame,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyAllWindows()

    x, y, box_width, box_height = roi
    if box_width <= 0 or box_height <= 0:
        raise RuntimeError("No target selected")

    target_box = np.array(
        [x, y, x + box_width, y + box_height],
        dtype=np.float32,
    )

    print("Loading SAM 2.1...")
    predictor = build_sam2_video_predictor(
        args.model_config,
        str(checkpoint),
        device=device,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output}")

    try:
        with torch.inference_mode(), torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            inference_state = predictor.init_state(
                video_path=str(frames_dir),
            )
            predictor.reset_state(inference_state)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=args.object_id,
                box=target_box,
            )

            for frame_index, object_ids, mask_logits in (
                predictor.propagate_in_video(inference_state)
            ):
                if frame_index >= len(frame_names):
                    break

                frame = cv2.imread(str(frame_names[frame_index]))
                if frame is None:
                    raise RuntimeError(
                        f"Cannot read frame: {frame_names[frame_index]}"
                    )

                target_mask = None
                for index, object_id in enumerate(object_ids):
                    if int(object_id) != args.object_id:
                        continue
                    target_mask = (mask_logits[index] > 0.0).cpu().numpy()
                    if target_mask.ndim == 3:
                        target_mask = target_mask[0]
                    break

                writer.write(draw_mask_and_box(frame, target_mask))

                if frame_index % 50 == 0:
                    print(f"Tracking: {frame_index}/{len(frame_names)}")
    finally:
        writer.release()

    print("Finished")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
