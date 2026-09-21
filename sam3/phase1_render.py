"""Render a completed experiment as a target-only H.264 video and review sheets.

Reads saved decisions; never runs a model or modifies its predictions. Optional
labels are displayed only on comparison sheets, not used to alter the video.
"""
import argparse
import json
from pathlib import Path


def render(run_dir, output, labels_path=None, allow_partial=False):
    import cv2
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    from phase1_evaluate import load_rows

    run_dir, output = Path(run_dir), Path(output)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    partial = report["status"] != "completed"
    if partial and not allow_partial:
        raise ValueError("Only completed experiments can be rendered as a complete result")
    rows = load_rows(run_dir / "predictions.jsonl")
    labels = load_rows(labels_path) if labels_path else {}
    if set(labels) - set(rows):
        raise ValueError("Review labels must refer to processed frames")
    output.mkdir(parents=True, exist_ok=False)
    fps = report.get("output_fps", report["source_fps"])
    width, height = report["resolution"]
    writer = imageio_ffmpeg.write_frames(
        str(output / "target_only.mp4"), (width, height), fps=fps,
        pix_fmt_in="bgr24", pix_fmt_out="yuv420p", codec="libx264", quality=7,
        macro_block_size=1, ffmpeg_log_level="error", output_params=["-movflags", "+faststart"])
    cap = cv2.VideoCapture(report["video"])
    tiles, index, written = [], 0, 0
    try:
        writer.send(None)
        while index <= max(rows):
            ok, source = cap.read()
            if not ok:
                raise ValueError(f"Source video ended before prediction frame {index}")
            if index in rows:
                row = rows[index]
                frame = source.copy()
                if row["box"]:
                    coords = [round(v*s) for v, s in zip(row["box"], (width, height, width, height))]
                    cv2.rectangle(frame, coords[:2], coords[2:], (0, 0, 255), 3)
                cv2.rectangle(frame, (0, 0), (width, 31), (18, 18, 18), -1)
                label = (f'{row["time_seconds"]:.1f}s | {row["state"]} | '
                         + (f'TARGET track {row["tracker_id"]}' if row["box"] else "Target unconfirmed"))
                if partial:
                    label = f'PARTIAL ({report["status"]}) | ' + label
                cv2.putText(frame, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 225, 255), 1)
                writer.send(frame.tobytes())
                written += 1
                if index in labels:
                    # Independent draft labels remain visibly distinguished.
                    review = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(review)
                    for box, color in [(labels[index]["target_box"], (220, 40, 255)), (row["box"], (255, 30, 30))]:
                        if box:
                            draw.rectangle([round(v*s) for v, s in zip(box, (width, height, width, height))], outline=color, width=3)
                    review = review.resize((640, 363))
                    tile = Image.new("RGB", (640, 412), (18, 18, 18))
                    tile.paste(review, (0, 31))
                    draw = ImageDraw.Draw(tile)
                    draw.text((8, 8), f'{row["time_seconds"]:.2f}s / frame {index} / {row["state"]}', fill="white")
                    draw.text((8, 397), "RED: model decision     PURPLE: source review draft", fill="white")
                    tiles.append(tile)
            index += 1
    finally:
        cap.release()
        writer.close()
    if written != len(rows):
        raise ValueError("Rendered frame count differs from predictions")
    for offset in range(0, len(tiles), 4):
        sheet = Image.new("RGB", (1280, 824), (18, 18, 18))
        for j, tile in enumerate(tiles[offset:offset+4]):
            sheet.paste(tile, ((j % 2)*640, (j // 2)*412))
        sheet.save(output / f"review_{offset//4+1}.jpg", quality=94)
    (output / "render.json").write_text(json.dumps({
        "source_run": str(run_dir.resolve()), "source_status": report["status"], "rendered_frames": written, "output_fps": fps,
        "labels": str(Path(labels_path).resolve()) if labels_path else None,
        "note": "H.264, no audio; sampled predictions only, original playback speed. Labels never affect video decisions."
    }, indent=2), encoding="utf-8")
    print(f"Rendered: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--labels", help="Optional independent review labels for comparison sheets")
    parser.add_argument("--allow-partial", action="store_true", help="Render available predictions from a failed/interrupted run with a visible PARTIAL banner")
    args = parser.parse_args()
    render(args.run, args.output, args.labels, args.allow_partial)
