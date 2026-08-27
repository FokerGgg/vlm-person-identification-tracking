from pathlib import Path
import argparse

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
BLACK = (0, 0, 0)


def parse_args():
    p = argparse.ArgumentParser(
        description="Natural-language person search with clean-view confirmation"
    )
    p.add_argument("--video", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--output", default="outputs/semantic_person_search_v2.mp4")
    p.add_argument("--yolo", default="yolo11n.pt")
    p.add_argument("--clip-model", default="ViT-B-32")
    p.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--det-conf", type=float, default=0.25)
    p.add_argument("--threshold", type=float, default=0.11)
    p.add_argument("--min-margin", type=float, default=0.025)
    p.add_argument("--confirm-frames", type=int, default=5)
    p.add_argument("--release-frames", type=int, default=20)
    p.add_argument("--crop-pad", type=float, default=0.05)
    p.add_argument(
        "--max-overlap",
        type=float,
        default=0.08,
        help="Do not confirm/save a target while its box overlaps another person too much",
    )
    p.add_argument(
        "--save-reference",
        default=None,
        help="Optional path to save a clean confirmed target crop, e.g. data/target_reference_auto.png",
    )
    return p.parse_args()


def l2_normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def area(box):
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def intersection(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def overlap_fraction(target, other):
    return intersection(target, other) / max(area(target), 1.0)


def max_overlap(index, boxes):
    if len(boxes) <= 1:
        return 0.0
    return max(
        overlap_fraction(boxes[index], boxes[j])
        for j in range(len(boxes))
        if j != index
    )


def crop_person(frame, box, pad_ratio):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px = int((x2 - x1) * pad_ratio)
    py = int((y2 - y1) * pad_ratio)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w - 1, x2 + px), min(h - 1, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return None, None
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def draw_status(frame, text):
    cv2.rectangle(
        frame,
        (10, 10),
        (min(frame.shape[1] - 10, 900), 60),
        BLACK,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (24, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        YELLOW,
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Target description:", args.query)

    detector = YOLO(args.yolo)

    print("Loading OpenCLIP...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.clip_pretrained,
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    # Positive prompt minus several generic negative prompts is more stable
    # than raw positive similarity alone.
    texts = [
        args.query,
        "a different person",
        "a generic pedestrian",
        "a mannequin or humanoid robot",
    ]
    tokens = tokenizer(texts).to(device)
    with torch.inference_mode():
        text_features = l2_normalize(
            clip_model.encode_text(tokens).float()
        )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {out_path}")

    positive_streak = 0
    negative_streak = 0
    target_active = False
    confirmed_box = None
    frame_id = 0
    best_clean_score = -1e9
    best_clean_crop = None
    reference_saved = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = detector.predict(
            frame,
            classes=[0],
            conf=args.det_conf,
            device=0 if device == "cuda" else "cpu",
            verbose=False,
        )[0]

        raw_boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            raw_boxes = [
                tuple(map(int, b.tolist()))
                for b in result.boxes.xyxy.detach().cpu().numpy()
            ]

        crops = []
        records = []
        for i, raw_box in enumerate(raw_boxes):
            crop, padded_box = crop_person(
                frame,
                raw_box,
                args.crop_pad,
            )
            if crop is None or padded_box is None or crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(Image.fromarray(rgb))
            records.append(
                {
                    "index": i,
                    "raw_box": raw_box,
                    "box": padded_box,
                    "crop_bgr": crop.copy(),
                    "overlap": max_overlap(i, raw_boxes),
                }
            )

        scores = []
        if crops:
            batch = torch.stack(
                [preprocess(img) for img in crops]
            ).to(device)

            with torch.inference_mode():
                image_features = l2_normalize(
                    clip_model.encode_image(batch).float()
                )
                sims = image_features @ text_features.T
                scores = (
                    sims[:, 0] - sims[:, 1:].mean(dim=1)
                ).detach().cpu().tolist()

        best_index = None
        best_score = -1e9
        second_score = -1e9

        if scores:
            order = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )
            best_index = order[0]
            best_score = float(scores[best_index])
            if len(order) > 1:
                second_score = float(scores[order[1]])

            best_record = records[best_index]
            margin = best_score - second_score
            clean = best_record["overlap"] <= args.max_overlap
            valid = (
                best_score >= args.threshold
                and margin >= args.min_margin
                and clean
            )

            if valid:
                positive_streak += 1
                negative_streak = 0

                if best_score > best_clean_score:
                    best_clean_score = best_score
                    best_clean_crop = best_record["crop_bgr"].copy()
            else:
                positive_streak = 0
                negative_streak += 1

            if not target_active and positive_streak >= args.confirm_frames:
                target_active = True
                confirmed_box = best_record["box"]

                if args.save_reference and best_clean_crop is not None:
                    ref_path = Path(args.save_reference)
                    ref_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(ref_path), best_clean_crop)
                    reference_saved = True
                    print(f"Saved clean reference: {ref_path.resolve()}")

            if target_active and negative_streak >= args.release_frames:
                target_active = False
                confirmed_box = None

            if target_active and valid:
                confirmed_box = best_record["box"]

        for i, record in enumerate(records):
            x1, y1, x2, y2 = record["box"]
            score = float(scores[i]) if i < len(scores) else -1.0
            overlap = record["overlap"]

            is_best = i == best_index
            is_target = target_active and is_best

            if is_target:
                color, thickness = RED, 4
                label = f"TARGET S={score:.3f} O={overlap:.2f}"
            elif is_best:
                color, thickness = ORANGE, 3
                label = (
                    f"BEST S={score:.3f} O={overlap:.2f} "
                    f"{positive_streak}/{args.confirm_frames}"
                )
            else:
                color, thickness = BLUE, 2
                label = f"candidate S={score:.3f} O={overlap:.2f}"

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
                (x1, max(75, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        if target_active:
            status = "TARGET CONFIRMED"
        elif best_index is not None and records[best_index]["overlap"] > args.max_overlap:
            status = "WAITING FOR CLEAN VIEW"
        else:
            status = f"SEARCHING {positive_streak}/{args.confirm_frames}"

        draw_status(frame, status)
        writer.write(frame)

        frame_id += 1
        if frame_id % 100 == 0 or frame_id == total:
            print(f"Progress: {frame_id}/{total}")

    cap.release()
    writer.release()

    print("Finished")
    print("Output:", out_path.resolve())
    if args.save_reference and not reference_saved:
        print(
            "Reference was not saved because no clean target confirmation was obtained."
        )


if __name__ == "__main__":
    main()
