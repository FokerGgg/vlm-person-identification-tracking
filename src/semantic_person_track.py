from __future__ import annotations

from pathlib import Path
import argparse
import math

import cv2
import numpy as np
import open_clip
from PIL import Image
import torch
from ultralytics import YOLO


BLUE = (255, 0, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Natural-language person tracking with occlusion-aware identity recovery"
    )
    parser.add_argument("--video", required=True, help="Input video")
    parser.add_argument("--query", required=True, help="Natural-language target description")
    parser.add_argument(
        "--reference-image",
        default=None,
        help="Optional clean crop of the target; strongly recommended",
    )
    parser.add_argument(
        "--output",
        default="outputs/semantic_person_track_occlusion_fixed.mp4",
        help="Output video",
    )

    parser.add_argument("--yolo", default="yolo11n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--det-conf", type=float, default=0.20)
    parser.add_argument("--crop-pad", type=float, default=0.05)

    # Initial acquisition.
    parser.add_argument("--acquire-threshold", type=float, default=0.11)
    parser.add_argument("--min-margin", type=float, default=0.025)
    parser.add_argument("--confirm-frames", type=int, default=5)

    # Fixed reference-image gates.
    parser.add_argument("--reference-threshold", type=float, default=0.52)
    parser.add_argument("--strong-reference-threshold", type=float, default=0.72)
    parser.add_argument("--reference-weight", type=float, default=0.30)

    # Re-identification.
    parser.add_argument("--keep-appearance-threshold", type=float, default=0.46)
    parser.add_argument("--reid-threshold", type=float, default=0.58)
    parser.add_argument("--reid-margin", type=float, default=0.025)
    parser.add_argument("--reid-confirm-frames", type=int, default=5)
    parser.add_argument("--release-frames", type=int, default=45)

    # Occlusion handling.
    parser.add_argument(
        "--occlusion-overlap",
        type=float,
        default=0.18,
        help="intersection/target-area ratio that starts OCCLUDED state",
    )
    parser.add_argument(
        "--clear-overlap",
        type=float,
        default=0.08,
        help="candidate must fall below this overlap before ReID/feature update",
    )
    parser.add_argument(
        "--separation-frames",
        type=int,
        default=5,
        help="number of clean frames required after an occlusion",
    )

    # Target-memory update. Small EMA is deliberately conservative.
    parser.add_argument("--feature-ema", type=float, default=0.03)
    parser.add_argument("--feature-update-appearance", type=float, default=0.66)
    parser.add_argument("--feature-update-det-conf", type=float, default=0.35)

    return parser.parse_args()


def l2_normalize(feature: torch.Tensor) -> torch.Tensor:
    return feature / feature.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def encode_images(
    images: list[Image.Image],
    model: torch.nn.Module,
    preprocess,
    device: str,
) -> torch.Tensor:
    batch = torch.stack([preprocess(image) for image in images]).to(device)
    with torch.inference_mode():
        features = model.encode_image(batch)
    return l2_normalize(features.float())


def crop_person(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    pad_ratio: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | tuple[None, None]:
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame_w - 1, x2 + pad_x)
    y2 = min(frame_h - 1, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None, None

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def box_area(box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def intersection_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return float(max(0, ix2 - ix1) * max(0, iy2 - iy1))


def overlap_fraction(
    target_box: tuple[int, int, int, int],
    other_box: tuple[int, int, int, int],
) -> float:
    """Fraction of target box covered by another person's box."""
    area = max(box_area(target_box), 1.0)
    return intersection_area(target_box, other_box) / area


def max_person_overlap(
    index: int,
    raw_boxes: list[tuple[int, int, int, int]],
) -> float:
    if not raw_boxes:
        return 0.0

    target = raw_boxes[index]
    best = 0.0
    for j, other in enumerate(raw_boxes):
        if j == index:
            continue
        best = max(best, overlap_fraction(target, other))
    return best


def score_to_quality(score: float) -> float:
    return float(np.clip((score + 0.10) / 0.35, 0.0, 1.0))


def reference_to_quality(score: float) -> float:
    return float(np.clip((score - 0.20) / 0.60, 0.0, 1.0))


def box_motion_quality(
    current: tuple[int, int, int, int],
    previous_reliable: tuple[int, int, int, int] | None,
    frame_width: int,
    frame_height: int,
) -> float:
    if previous_reliable is None:
        return 0.5

    x1, y1, x2, y2 = current
    px1, py1, px2, py2 = previous_reliable

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0

    diagonal = max(math.hypot(frame_width, frame_height), 1.0)
    distance = math.hypot(cx - pcx, cy - pcy) / diagonal

    return math.exp(-5.0 * distance)


def draw_status(frame: np.ndarray, text: str) -> None:
    cv2.rectangle(
        frame,
        (10, 10),
        (min(frame.shape[1] - 10, 980), 62),
        BLACK,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (24, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        YELLOW,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    detector = YOLO(args.yolo)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.clip_pretrained,
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    texts = [
        args.query,
        "a different person",
        "a generic pedestrian",
        "a mannequin or robot",
    ]
    text_tokens = tokenizer(texts).to(device)

    with torch.inference_mode():
        text_features = l2_normalize(
            clip_model.encode_text(text_tokens).float()
        )

    reference_feature: torch.Tensor | None = None
    if args.reference_image:
        reference_path = Path(args.reference_image)
        if not reference_path.exists():
            raise FileNotFoundError(
                f"Reference image not found: {reference_path}"
            )

        reference_bgr = cv2.imread(str(reference_path))
        if reference_bgr is None:
            raise RuntimeError(
                f"Cannot read reference image: {reference_path}"
            )

        reference_rgb = cv2.cvtColor(
            reference_bgr,
            cv2.COLOR_BGR2RGB,
        )
        reference_feature = encode_images(
            [Image.fromarray(reference_rgb)],
            clip_model,
            preprocess,
            device,
        )[0]
        print(f"Reference image enabled: {reference_path}")
    else:
        print(
            "Warning: no reference image. Similar clothing may cause false locks."
        )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    frame_id = 0

    # SEARCHING -> TRACKING -> OCCLUDED -> POST-OCCLUSION -> RECOVERING
    target_acquired = False
    locked_id: int | None = None

    pending_id: int | None = None
    pending_hits = 0

    reid_pending_id: int | None = None
    reid_hits = 0

    target_feature: torch.Tensor | None = None
    last_reliable_box: tuple[int, int, int, int] | None = None

    occlusion_active = False
    separation_hits = 0
    lost_frames = 0
    post_reid_freeze = 0

    lock_count = 0
    id_switch_count = 0
    occlusion_count = 0

    def reset_reid_confirmation() -> None:
        nonlocal reid_pending_id, reid_hits
        reid_pending_id = None
        reid_hits = 0

    def evaluate_reid(
        candidates: list[dict],
    ) -> tuple[dict | None, bool]:
        """Return (best_candidate, valid). Only clean candidates are eligible."""
        clean = [
            c
            for c in candidates
            if c["overlap"] <= args.clear_overlap
        ]
        clean.sort(key=lambda item: item["reid"], reverse=True)

        if not clean:
            return None, False

        best = clean[0]
        second_score = clean[1]["reid"] if len(clean) > 1 else 0.0
        margin = best["reid"] - second_score

        valid = (
            best["id"] >= 0
            and best["appearance"] >= args.keep_appearance_threshold
            and best["reid"] >= args.reid_threshold
            and margin >= args.reid_margin
        )
        return best, valid

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = detector.track(
            frame,
            persist=True,
            tracker=args.tracker,
            classes=[0],
            conf=args.det_conf,
            device=0 if device == "cuda" else "cpu",
            verbose=False,
        )[0]

        candidates: list[dict] = []

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()

            ids_tensor = result.boxes.id
            ids = (
                ids_tensor.detach().cpu().numpy().astype(int)
                if ids_tensor is not None
                else np.full(len(xyxy), -1, dtype=int)
            )

            raw_boxes = [
                tuple(map(int, raw_box.tolist()))
                for raw_box in xyxy
            ]

            pil_crops: list[Image.Image] = []
            crop_records: list[
                tuple[
                    int,
                    tuple[int, int, int, int],
                    tuple[int, int, int, int],
                ]
            ] = []

            for index, raw_box in enumerate(raw_boxes):
                crop, padded_box = crop_person(
                    frame,
                    raw_box,
                    args.crop_pad,
                )
                if crop is None or padded_box is None or crop.size == 0:
                    continue

                rgb_crop = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2RGB,
                )
                pil_crops.append(Image.fromarray(rgb_crop))
                crop_records.append(
                    (index, raw_box, padded_box)
                )

            if pil_crops:
                image_features = encode_images(
                    pil_crops,
                    clip_model,
                    preprocess,
                    device,
                )

                similarities = image_features @ text_features.T
                semantic_scores = (
                    similarities[:, 0]
                    - similarities[:, 1:].mean(dim=1)
                ).detach().cpu().tolist()

                if reference_feature is not None:
                    reference_scores = (
                        image_features @ reference_feature
                    ).detach().cpu().tolist()
                else:
                    reference_scores = [0.0] * len(pil_crops)

                for crop_index, (
                    source_index,
                    raw_box,
                    padded_box,
                ) in enumerate(crop_records):
                    semantic_score = float(
                        semantic_scores[crop_index]
                    )
                    reference_score = float(
                        reference_scores[crop_index]
                    )

                    rank_score = semantic_score
                    if reference_feature is not None:
                        rank_score += (
                            args.reference_weight * reference_score
                        )

                    overlap = max_person_overlap(
                        source_index,
                        raw_boxes,
                    )

                    candidates.append(
                        {
                            "id": int(ids[source_index]),
                            "raw_box": raw_box,
                            "box": padded_box,
                            "feature": image_features[crop_index],
                            "det_conf": float(confs[source_index]),
                            "semantic": semantic_score,
                            "reference": reference_score,
                            "rank": rank_score,
                            "appearance": 0.0,
                            "motion": box_motion_quality(
                                padded_box,
                                last_reliable_box,
                                width,
                                height,
                            ),
                            "reid": 0.0,
                            "overlap": overlap,
                        }
                    )

        # Appearance/ReID scores are always computed from frozen target memory.
        if target_feature is not None:
            for candidate in candidates:
                candidate["appearance"] = float(
                    torch.dot(
                        candidate["feature"],
                        target_feature,
                    ).item()
                )

                semantic_quality = score_to_quality(
                    candidate["semantic"]
                )

                if reference_feature is not None:
                    candidate["reid"] = (
                        0.55 * max(candidate["appearance"], 0.0)
                        + 0.25
                        * reference_to_quality(
                            candidate["reference"]
                        )
                        + 0.15 * semantic_quality
                        + 0.05 * candidate["motion"]
                    )
                else:
                    candidate["reid"] = (
                        0.78 * max(candidate["appearance"], 0.0)
                        + 0.17 * semantic_quality
                        + 0.05 * candidate["motion"]
                    )

        selected: dict | None = None
        confirming: dict | None = None
        status = "SEARCHING"

        # ------------------------------------------------------------
        # 1) INITIAL ACQUISITION
        # ------------------------------------------------------------
        if not target_acquired:
            ranked = sorted(
                candidates,
                key=lambda item: item["rank"],
                reverse=True,
            )

            if ranked:
                best = ranked[0]
                second_score = (
                    ranked[1]["rank"]
                    if len(ranked) > 1
                    else -1.0
                )
                margin = best["rank"] - second_score

                text_ok = (
                    best["semantic"] >= args.acquire_threshold
                )

                if reference_feature is None:
                    reference_ok = True
                    strong_reference = False
                else:
                    reference_ok = (
                        best["reference"]
                        >= args.reference_threshold
                    )
                    strong_reference = (
                        best["reference"]
                        >= args.strong_reference_threshold
                        and best["semantic"] >= 0.0
                    )

                # Do not establish identity from an overlapping crop.
                clean_enough = (
                    best["overlap"] <= args.clear_overlap
                )

                valid = (
                    best["id"] >= 0
                    and clean_enough
                    and margin >= args.min_margin
                    and (
                        (text_ok and reference_ok)
                        or strong_reference
                    )
                )

                if valid:
                    if pending_id == best["id"]:
                        pending_hits += 1
                    else:
                        pending_id = best["id"]
                        pending_hits = 1

                    confirming = best
                    status = (
                        f"CONFIRMING ID {best['id']} "
                        f"{pending_hits}/{args.confirm_frames}"
                    )
                else:
                    pending_id = None
                    pending_hits = 0
                    if not clean_enough:
                        status = "SEARCHING - WAITING FOR CLEAN VIEW"

                if (
                    valid
                    and pending_hits >= args.confirm_frames
                ):
                    target_acquired = True
                    locked_id = best["id"]
                    selected = best

                    target_feature = best["feature"].clone()

                    if reference_feature is not None:
                        target_feature = l2_normalize(
                            (
                                0.70 * target_feature
                                + 0.30 * reference_feature
                            ).unsqueeze(0)
                        )[0]

                    last_reliable_box = best["box"]
                    lost_frames = 0
                    lock_count += 1
                    status = f"TRACKING ID {locked_id}"

                    pending_id = None
                    pending_hits = 0

        # ------------------------------------------------------------
        # 2) TRACKING / OCCLUSION / RECOVERY
        # ------------------------------------------------------------
        else:
            same_id = next(
                (
                    item
                    for item in candidates
                    if item["id"] == locked_id
                ),
                None,
            )

            # A. Locked ID is still visible.
            if same_id is not None:
                lost_frames = 0

                # Start occlusion state BEFORE appearance can be contaminated.
                if (
                    same_id["overlap"]
                    >= args.occlusion_overlap
                ):
                    if not occlusion_active:
                        occlusion_count += 1

                    occlusion_active = True
                    separation_hits = 0
                    selected = same_id
                    reset_reid_confirmation()

                    status = (
                        f"OCCLUDED ID {locked_id} "
                        f"O={same_id['overlap']:.2f} - MEMORY FROZEN"
                    )

                # Target is separating from the occluder.
                elif occlusion_active:
                    selected = same_id

                    if (
                        same_id["overlap"]
                        <= args.clear_overlap
                    ):
                        separation_hits += 1
                    else:
                        separation_hits = 0

                    reset_reid_confirmation()

                    if (
                        separation_hits
                        < args.separation_frames
                    ):
                        status = (
                            f"POST-OCCLUSION ID {locked_id} "
                            f"{separation_hits}/{args.separation_frames}"
                        )
                    else:
                        occlusion_active = False
                        separation_hits = 0
                        post_reid_freeze = (
                            args.separation_frames
                        )
                        last_reliable_box = same_id["box"]
                        status = (
                            f"TRACKING ID {locked_id} "
                            "- OCCLUSION CLEARED"
                        )

                # Normal clean tracking.
                else:
                    selected = same_id
                    last_reliable_box = same_id["box"]

                    # If the same tracker ID still looks plausible, keep it.
                    if (
                        same_id["appearance"]
                        >= args.keep_appearance_threshold
                    ):
                        reset_reid_confirmation()
                        status = f"TRACKING ID {locked_id}"

                    # If appearance strongly disagrees, verify alternatives
                    # over several clean frames instead of switching instantly.
                    else:
                        best, valid_reid = evaluate_reid(
                            candidates
                        )

                        if (
                            valid_reid
                            and best is not None
                            and best["id"] != locked_id
                        ):
                            if (
                                reid_pending_id
                                == best["id"]
                            ):
                                reid_hits += 1
                            else:
                                reid_pending_id = best["id"]
                                reid_hits = 1

                            confirming = best
                            status = (
                                f"VERIFYING SWITCH "
                                f"{locked_id}->{best['id']} "
                                f"{reid_hits}/{args.reid_confirm_frames}"
                            )

                            if (
                                reid_hits
                                >= args.reid_confirm_frames
                            ):
                                old_id = locked_id
                                locked_id = best["id"]
                                selected = best
                                confirming = None

                                last_reliable_box = best["box"]
                                post_reid_freeze = (
                                    args.separation_frames
                                )
                                reset_reid_confirmation()

                                if old_id != locked_id:
                                    id_switch_count += 1

                                status = (
                                    f"RE-IDENTIFIED "
                                    f"{old_id}->{locked_id}"
                                )
                        else:
                            reset_reid_confirmation()
                            status = (
                                f"TRACKING ID {locked_id} "
                                "- LOW APPEARANCE, HOLDING ID"
                            )

                # Conservative feature update:
                # never update during/just after occlusion or on overlapping crops.
                if (
                    selected is same_id
                    and not occlusion_active
                    and post_reid_freeze <= 0
                    and same_id["overlap"]
                    <= args.clear_overlap
                    and same_id["appearance"]
                    >= args.feature_update_appearance
                    and same_id["det_conf"]
                    >= args.feature_update_det_conf
                    and same_id["semantic"]
                    >= args.acquire_threshold * 0.45
                    and target_feature is not None
                ):
                    alpha = args.feature_ema
                    target_feature = l2_normalize(
                        (
                            (1.0 - alpha) * target_feature
                            + alpha * same_id["feature"]
                        ).unsqueeze(0)
                    )[0]

            # B. Locked tracker ID disappeared.
            else:
                lost_frames += 1

                # If disappearance happened during an occlusion, do NOT ReID
                # until people have clearly separated for several frames.
                if occlusion_active:
                    scene_clear = (
                        len(candidates) > 0
                        and all(
                            c["overlap"]
                            <= args.clear_overlap
                            for c in candidates
                        )
                    )

                    if scene_clear:
                        separation_hits += 1
                    else:
                        separation_hits = 0

                    reset_reid_confirmation()

                    status = (
                        f"OCCLUDED-LOST ID {locked_id} "
                        f"CLEAR {separation_hits}/{args.separation_frames}"
                    )

                    if (
                        separation_hits
                        >= args.separation_frames
                    ):
                        occlusion_active = False
                        separation_hits = 0
                        status = "RECOVERING AFTER OCCLUSION"

                # ReID is allowed only after the scene is clean.
                if not occlusion_active:
                    best, valid_reid = evaluate_reid(
                        candidates
                    )

                    if valid_reid and best is not None:
                        if (
                            reid_pending_id
                            == best["id"]
                        ):
                            reid_hits += 1
                        else:
                            reid_pending_id = best["id"]
                            reid_hits = 1

                        confirming = best
                        status = (
                            f"REIDENTIFYING ID {best['id']} "
                            f"{reid_hits}/{args.reid_confirm_frames}"
                        )

                        if (
                            reid_hits
                            >= args.reid_confirm_frames
                        ):
                            old_id = locked_id
                            locked_id = best["id"]
                            selected = best
                            confirming = None

                            lost_frames = 0
                            last_reliable_box = best["box"]
                            post_reid_freeze = (
                                args.separation_frames
                            )
                            reset_reid_confirmation()

                            if old_id != locked_id:
                                id_switch_count += 1

                            status = (
                                f"RE-IDENTIFIED "
                                f"{old_id}->{locked_id}"
                            )
                    else:
                        reset_reid_confirmation()
                        status = (
                            f"LOST {lost_frames}/{args.release_frames}"
                        )

                if lost_frames > args.release_frames:
                    status = "GLOBAL RE-IDENTIFICATION"

        if post_reid_freeze > 0:
            post_reid_freeze -= 1

        # ------------------------------------------------------------
        # VISUALIZATION
        # ------------------------------------------------------------
        for candidate in candidates:
            x1, y1, x2, y2 = candidate["box"]

            is_selected = selected is candidate
            is_confirming = (
                confirming is candidate
                and not is_selected
            )

            if is_selected:
                color = RED
                thickness = 4
                label = (
                    f"ID {candidate['id']} TARGET "
                    f"S={candidate['semantic']:.3f} "
                    f"A={candidate['appearance']:.2f} "
                    f"O={candidate['overlap']:.2f}"
                )
            elif is_confirming:
                color = ORANGE
                thickness = 3
                label = (
                    f"ID {candidate['id']} CHECK "
                    f"A={candidate['appearance']:.2f} "
                    f"R={candidate['reference']:.2f} "
                    f"O={candidate['overlap']:.2f}"
                )
            else:
                color = BLUE
                thickness = 2
                label = (
                    f"ID {candidate['id']} candidate "
                    f"S={candidate['semantic']:.3f} "
                    f"O={candidate['overlap']:.2f}"
                )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                thickness,
            )
            cv2.putText(
                frame,
                label,
                (x1, max(78, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                color,
                2,
                cv2.LINE_AA,
            )

        draw_status(frame, status)
        writer.write(frame)

        frame_id += 1
        if (
            frame_id % 50 == 0
            or frame_id == total_frames
        ):
            print(
                f"Progress: {frame_id}/{total_frames}"
            )

    cap.release()
    writer.release()

    print("Finished")
    print(f"Output: {output_path.resolve()}")
    print(f"Initial locks: {lock_count}")
    print(f"Confirmed tracker-ID changes: {id_switch_count}")
    print(f"Occlusion events: {occlusion_count}")


if __name__ == "__main__":
    main()
