"""Evaluate normalized target boxes against manually labeled video frames.

This is a single-target localization/recovery evaluator, not a HOTA/IDF1 or
MOT ID-switch implementation. Correct human identity must be reflected in GT.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from phase1.identity import iou


def load_rows(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            index = row["frame_index"]
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in rows:
                raise ValueError("Frame indices must be unique nonnegative integers")
            rows[index] = row
    if not rows:
        raise ValueError("No labeled/predicted frames found")
    return rows


def validate_box(box):
    if box is None:
        return
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("Boxes must be null or normalized xyxy lists")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in box):
        raise ValueError("Boxes must use finite coordinates within [0,1]")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Boxes must have positive area")


def evaluate(predictions, labels, threshold=0.5, frame_step=1):
    if not predictions or not labels:
        raise ValueError("Predictions and labels must both be nonempty")
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    if isinstance(frame_step, bool) or not isinstance(frame_step, int) or frame_step < 1:
        raise ValueError("frame_step must be a positive integer")
    unknown = set(labels)-set(predictions)
    if unknown:
        raise ValueError(f"Annotations contain {len(unknown)} frames without predictions; use labels for this run's frame range")
    counters = Counter()
    scenarios = defaultdict(Counter)
    first_emitted = None
    first_correct = None
    previous = None
    pending_reappearance = None
    recoveries = []
    for index in sorted(labels):
        gt, pred = labels[index], predictions[index]
        if "target_box" not in gt or "box" not in pred:
            raise ValueError("Each label needs target_box (null if absent); each prediction needs box")
        target, box = gt["target_box"], pred["box"]
        validate_box(target)
        validate_box(box)
        timestamp = pred["time_seconds"]
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Invalid prediction timestamp")
        if previous and timestamp <= previous[2]:
            raise ValueError("Prediction timestamps must increase")
        correct = target is not None and box is not None and iou(target, box) >= threshold
        if target is not None:
            outcome = "correct_target_frames" if correct else "missed_visible_frames" if box is None else "wrong_box_frames"
        else:
            outcome = "false_follow_while_absent_frames" if box is not None else "correct_abstention_frames"
        counters[outcome] += 1
        counters["visible_frames" if target is not None else "absent_frames"] += 1
        scenarios[gt.get("scenario", "unspecified")][outcome] += 1
        if box is not None and first_emitted is None:
            first_emitted = {"frame_index": index, "correct": correct}
        if correct and first_correct is None:
            first_correct = {"frame_index": index, "time_seconds": timestamp}
        contiguous = previous is not None and index == previous[0]+frame_step
        if not contiguous and pending_reappearance is not None:
            pending_reappearance["censored_reason"] = "annotation_gap"
            recoveries.append(pending_reappearance)
            pending_reappearance = None
        if contiguous and previous[1] is None and target is not None:
            pending_reappearance = {"reappearance_frame": index, "time_seconds": timestamp,
                                    "recovery_seconds": None}
        if pending_reappearance is not None:
            if correct:
                pending_reappearance["recovery_seconds"] = timestamp-pending_reappearance["time_seconds"]
                recoveries.append(pending_reappearance)
                pending_reappearance = None
            elif target is None:
                pending_reappearance["censored_reason"] = "target_disappeared_again"
                recoveries.append(pending_reappearance)
                pending_reappearance = None
        previous = index, target, timestamp
    if pending_reappearance is not None:
        pending_reappearance["censored_reason"] = "end_of_annotations"
        recoveries.append(pending_reappearance)
    return {"metric_type": "single_target_box_localization", "iou_threshold": threshold, "frame_step": frame_step,
            "evaluated_frames": len(labels), "prediction_frames": len(predictions),
            "annotation_coverage": len(labels)/len(predictions), "counts": dict(counters),
            "visible_target_success_rate": counters["correct_target_frames"]/counters["visible_frames"] if counters["visible_frames"] else None,
            "false_follow_when_absent_rate": counters["false_follow_while_absent_frames"]/counters["absent_frames"] if counters["absent_frames"] else None,
            "first_emission_on_labeled_frames": first_emitted, "first_correct_on_labeled_frames": first_correct,
            "reappearance_episodes": recoveries, "scenario_counts": {k: dict(v) for k, v in scenarios.items()},
            "limitations": ["No true MOT IDSW, IDF1 or HOTA without full person-ID annotations.",
                            "Reappearance latency requires consecutive annotations around each event.",
                            "null target_box means target absent/not localizable, not missing annotation.",
                            "Report scores on held-out clips; do not tune thresholds on the test split."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--frame-step", type=int, default=1, help="Expected sampling stride for recovery episodes; not inferred across missing annotations")
    args = p.parse_args()
    result = evaluate(load_rows(args.predictions), load_rows(args.labels), args.iou, args.frame_step)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
