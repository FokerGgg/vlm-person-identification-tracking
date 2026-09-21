"""Manually label a short RGB clip for single-target evaluation.

B: draw target box; A: absent/not localizable; P: accept a prediction proposal;
C: explicitly copy the previous label; U: undo; Q/Esc: save and quit.
Labels are never accepted automatically. Existing output files are refused.
"""
import argparse
import json
from pathlib import Path


def main():
    import cv2
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions", help="Optional proposals; every frame still requires explicit confirmation")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, help="Exclusive frame index")
    p.add_argument("--step", type=int, default=1, help="Label every Nth source frame; use 3 for a frame-stride 3 run")
    p.add_argument("--scenario", default="unspecified")
    args = p.parse_args()
    if args.start < 0 or args.step < 1 or args.end is not None and args.end <= args.start:
        p.error("Invalid frame interval")
    out = Path(args.output)
    if out.exists():
        p.error("Output already exists; choose a new label file")
    proposals = {}
    if args.predictions:
        from phase1_evaluate import load_rows
        proposals = load_rows(args.predictions)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        p.error("Cannot open video")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    index = args.start
    try:
        while args.end is None or index < args.end:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            proposal = proposals.get(index, {}).get("box")
            display = frame.copy()
            if proposal:
                coords = [int(v*s) for v, s in zip(proposal, (w, h, w, h))]
                cv2.rectangle(display, coords[:2], coords[2:], (0, 165, 255), 2)
            cv2.rectangle(display, (0, 0), (w, 55), (0, 0, 0), -1)
            cv2.putText(display, f"Frame {index}: B box / A absent / P proposal / C copy / U undo / Q save", (5, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(display, "Label the ORIGINAL PERSON, even after clothing changes.", (5, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imshow("Phase 1 annotation", display)
            key = cv2.waitKey(0) & 0xff
            if key in (ord("q"), 27):
                break
            if key == ord("u"):
                if rows:
                    index = rows.pop()["frame_index"]
                continue
            if key == ord("b"):
                x, y, bw, bh = cv2.selectROI("Draw ORIGINAL target", frame, fromCenter=False)
                cv2.destroyWindow("Draw ORIGINAL target")
                if not bw or not bh:
                    continue
                box = [x/w, y/h, (x+bw)/w, (y+bh)/h]
            elif key == ord("a"):
                box = None
            elif key == ord("p") and proposal:
                box = proposal
            elif key == ord("c") and rows:
                box = rows[-1]["target_box"]
            else:
                continue
            rows.append({"frame_index": index, "target_box": box, "scenario": args.scenario})
            index += args.step
    finally:
        cap.release()
        cv2.destroyAllWindows()
        with out.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row)+"\n")
        print(f"Saved {len(rows)} explicitly labeled frames: {out.resolve()}")


if __name__ == "__main__":
    main()
