"""Run English person-description experiments from the VS Code terminal."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import importlib.metadata
import importlib.util
import json
import gzip
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
# Model downloads/config stay inside the project; credentials remain in the
# normal Hugging Face token location. Never print or copy tokens into reports.
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / ".cache" / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# kernels 0.16 + hub 1.30 otherwise emit an invalid trailing-space header
# when telemetry is disabled. A fixed, non-identifying origin avoids it.
os.environ.setdefault("HF_HUB_USER_AGENT_ORIGIN", "phase1")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
(Path(os.environ["YOLO_CONFIG_DIR"]) / "Ultralytics").mkdir(parents=True, exist_ok=True)

from phase1.identity import Config, IdentityManager


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", help="Local test video; constant-frame-rate RGB video is expected")
    p.add_argument("--query", help='English clothing description, e.g. "person wearing a red jacket"')
    p.add_argument("--backend", choices=["sam3", "yolo-clip"], default="sam3")
    p.add_argument("--output", help="New result directory (existing directories are refused)")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--yolo", default=str(ROOT / "yolo11n.pt"))
    p.add_argument("--sam3-model", default="facebook/sam3", help="Transformers SAM3 checkpoint, not a SAM3.1 .pt file")
    p.add_argument("--sam3-postprocessing", choices=["auto", "native", "portable"], default="auto",
                   help="Auto uses portable NMS/components on Windows/CPU; native requires compatible HF kernels")
    p.add_argument("--sam3-history-frames", type=int, default=32, help="Retained recent non-conditioning SAM3 outputs; all conditioning anchors are preserved")
    p.add_argument("--sam3-mask-cleanup", choices=['off', 'conservative'], default='conservative',
                   help="Remove small remote mask islands from the already-selected person; does not fill occlusion")
    p.add_argument("--reid-encoder", help="Optional local TorchScript encoder; contract in PHASE1.md")
    p.add_argument("--max-objects", type=int, default=8, help="SAM3 object cap, including both prompt groups")
    p.add_argument("--max-frames", type=int, default=900, help="Bound first experiments to 900 frames; 0 reads to end")
    p.add_argument("--frame-stride", type=int, default=1, help="Process every Nth source frame; preserves source timestamps and playback speed")
    p.add_argument("--fps", type=float, help="Explicit source FPS override when metadata is invalid")
    p.add_argument("--acquire-threshold", type=float)
    p.add_argument("--acquire-margin", type=float)
    p.add_argument("--recovery-threshold", type=float, default=0.72)
    p.add_argument("--recovery-margin", type=float, default=0.08)
    p.add_argument("--no-recovery", action="store_true", help="Ablation: stop accepting targets after continuity breaks")
    p.add_argument("--no-video", action="store_true", help="Write only predictions and report")
    p.add_argument("--save-observations", action="store_true", help="Archive candidate embeddings and target masks for reproducible local debugging")
    p.add_argument("--check-env", action="store_true", help="Check local dependencies and CUDA without downloading models")
    return p.parse_args()


def environment():
    versions = {}
    for name in ("torch", "torchvision", "transformers", "ultralytics", "open-clip-torch", "opencv-python", "kernels"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {"python": sys.version.split()[0], "executable": sys.executable, "packages": versions}
    if importlib.util.find_spec("torch"):
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_memory_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    if versions["transformers"]:
        try:
            from transformers import Sam3VideoModel, Sam3VideoProcessor
            report["sam3_streaming_api"] = True
        except (ImportError, RuntimeError, OSError) as exc:
            report["sam3_streaming_api"] = False
            report["sam3_import_error"] = str(exc)
    report["sam3_weights_checked_by_this_command"] = False
    smoke_report = ROOT / "outputs" / "sam3_smoke.json"
    report["previous_sam3_smoke_report"] = str(smoke_report) if smoke_report.is_file() else None
    return report


def draw(frame, candidates, decision):
    import cv2
    h, w = frame.shape[:2]
    for c in candidates:
        box = tuple(int(v*scale) for v, scale in zip(c.box, (w, h, w, h)))
        selected = decision.box is not None and c.track_id == decision.tracker_id
        color = (0, 0, 255) if selected else (255, 150, 0)
        cv2.rectangle(frame, box[:2], box[2:], color, 3 if selected else 1)
        label = f"{'TARGET' if selected else 'candidate'} {c.track_id} S={c.semantic:.2f}"
        cv2.putText(frame, label, (box[0], max(40, box[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(frame, decision.state, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    return frame


def run(args):
    stride = getattr(args, "frame_stride", 1)
    if stride < 1:
        raise ValueError("frame-stride must be positive")
    if not args.video or not args.query or not args.query.strip():
        raise ValueError("Provide --video and --query, or use --check-env")
    video = Path(args.video).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Test video not found: {video}. Put your video in data/videos first.")
    if args.reid_encoder and not Path(args.reid_encoder).is_file():
        raise FileNotFoundError(f"ReID encoder not found: {args.reid_encoder}")
    if args.max_frames < 0 or args.max_objects < 2:
        raise ValueError("max-frames must be >=0, max-objects >=2")
    import cv2
    import numpy as np
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install CUDA PyTorch or explicitly choose --device cpu.")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode video: {video}")
    fps = args.fps if args.fps is not None else capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Invalid video FPS; provide a known source frame rate with --fps")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if min(width, height) <= 0:
        capture.release()
        raise ValueError("Invalid video dimensions")
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    out = Path(args.output) if args.output else ROOT / "outputs" / ("phase1_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    try:
        out.mkdir(parents=True, exist_ok=False)
    except Exception:
        capture.release()
        raise
    config = Config(
        acquire_threshold=args.acquire_threshold if args.acquire_threshold is not None else (0.55 if args.backend == "sam3" else 0.11),
        acquire_margin=args.acquire_margin if args.acquire_margin is not None else (0.10 if args.backend == "sam3" else 0.025),
        recovery_threshold=args.recovery_threshold, recovery_margin=args.recovery_margin,
        enable_recovery=not args.no_recovery)
    manager = IdentityManager(config)
    report = {"schema_version": 1, "status": "initializing", "video": str(video),
              "query": args.query, "backend": args.backend, "source_fps": fps,
              "source_frame_count": source_count, "resolution": [width, height],
              "frame_stride": stride, "output_fps": fps/stride,
              "timestamp_basis": "frame_index / source_fps (CFR assumption)",
              "causal": True, "config": asdict(config), "arguments": vars(args),
              "environment": environment(), "accuracy": None,
              "limitations": ["Accuracy requires manual ground truth; identity scores are not calibrated probabilities.",
                              "Default CLIP part features are not clothes-invariant ReID.",
                              "No robot coordinates, navigation, motor commands, or action-understanding VLM.",
                              "SAM3 backend uses facebook/sam3, not SAM3.1."]}
    writer = log = backend = observations = None
    processed, decoded, inference_ms, loop_ms = 0, 0, [], []
    states, events = Counter(), Counter()
    init_started = time.perf_counter()
    try:
        from phase1.models import Sam3Backend, YoloClipBackend
        backend = Sam3Backend(args) if args.backend == "sam3" else YoloClipBackend(args)
        report["encoder"] = backend.encoder.kind
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        report["initialization_seconds"] = time.perf_counter()-init_started
        if not args.no_video:
            writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps/stride, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Cannot create output video")
        log = (out / "predictions.jsonl").open("w", encoding="utf-8")
        if getattr(args, "save_observations", False):
            observations = gzip.open(out / "observations.jsonl.gz", "wt", encoding="utf-8")
            (out / "mask_audit").mkdir()
        while not args.max_frames or processed < args.max_frames:
            loop_start = time.perf_counter()
            # Decode sequentially; do not seek or use any future observation.
            # Include skipped-frame decode time in the next sampled-frame cost.
            ok, frame = capture.read()
            if not ok:
                break
            source_index = decoded
            decoded += 1
            while source_index % stride:
                ok, frame = capture.read()
                if not ok:
                    break
                source_index = decoded
                decoded += 1
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                raise ValueError("Video dimensions changed during decoding")
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            backend.target_hint = manager.locked_id
            candidates = backend.process(frame)
            decision = manager.update(candidates, source_index/fps)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter()-started)*1000
            inference_ms.append(latency_ms)
            states[decision.state] += 1
            if decision.event:
                events[decision.event] += 1
            row = {"frame_index": source_index, "time_seconds": source_index/fps,
                   **asdict(decision), "inference_and_identity_ms": latency_ms,
                   "candidates": [{"tracker_id": c.track_id, "box": c.box, "confidence": c.confidence,
                                   "semantic": c.semantic, "overlap": c.overlap} for c in candidates]}
            log.write(json.dumps(row, allow_nan=False)+"\n")
            if observations is not None:
                observations.write(json.dumps({"frame_index": source_index, "time_seconds": source_index/fps,
                                               "candidates": [asdict(c) for c in candidates]}, allow_nan=False)+"\n")
                mask = getattr(backend, "last_masks", {}).get(manager.locked_id)
                if mask is not None:
                    np.savez_compressed(out / "mask_audit" / f"{source_index:06d}.npz",
                                        tracker_id=manager.locked_id, shape=mask.shape,
                                        bits=np.packbits(mask))
            if writer:
                writer.write(draw(frame, candidates, decision))
            loop_ms.append((time.perf_counter()-loop_start)*1000)
            processed += 1
            if processed % 30 == 0:
                log.flush()
                print(f"{processed} samples / source {source_index} ({source_index/fps:.1f}s) | {decision.state} | inference {latency_ms:.0f} ms", flush=True)
        if processed == 0:
            raise RuntimeError("Video contains no decodable frames")
        if source_count > 0 and decoded < source_count and (not args.max_frames or processed < args.max_frames):
            report["status"] = "partial_decode"
        else:
            report["status"] = "completed"
        report["stopped_at_frame_limit"] = bool(args.max_frames and processed >= args.max_frames)
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error_type"] = type(exc).__name__
        report["error_stack"] = [{"file": Path(item.filename).name, "line": item.lineno, "function": item.name}
                                 for item in traceback.extract_tb(exc.__traceback__)[-12:]]
        if type(exc).__module__.startswith("torch"):
            # Local CUDA runtime messages help distinguish allocation/driver
            # failures. Network/download exceptions remain unprinted.
            report["accelerator_error"] = str(exc)[:1600]
            print(report["accelerator_error"], file=sys.stderr)
        # Raw network exceptions can contain URLs or signed credentials; do not
        # serialize them. The CLI displays a concise failure classification.
        raise
    finally:
        capture.release()
        if writer:
            writer.release()
        if log:
            log.close()
        if observations:
            observations.close()
        report.update(processed_frames=processed, decoded_source_frames=decoded, state_frames=dict(states), events=dict(events))
        if backend is not None:
            report["postprocessing"] = getattr(backend, "postprocessing", "backend_default")
        if inference_ms:
            steady = inference_ms[5:]  # Exclude explicit warm-up count, never label playback FPS as throughput.
            report["performance"] = {
                "first_frame_ms": inference_ms[0], "warmup_frames_excluded": min(5, processed),
                "mean_inference_ms_all_frames": float(np.mean(inference_ms)),
                "steady_p50_ms": float(np.median(steady)) if steady else None,
                "steady_p95_ms": float(np.percentile(steady, 95)) if steady else None,
                "offline_pipeline_fps": 1000*processed/sum(loop_ms) if loop_ms else None,
                "note": "Includes decode/inference/log/video writing; excludes model load, camera/network/control latency."}
        if args.device.startswith("cuda"):
            report["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
            report["peak_gpu_reserved_gib"] = torch.cuda.max_memory_reserved()/2**30
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(f"Results: {out.resolve()}", flush=True)


def main():
    args = parse_args()
    if args.check_env:
        print(json.dumps(environment(), ensure_ascii=False, indent=2))
        return
    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted; partial report saved.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        name = type(exc).__name__
        if name in ("FileNotFoundError", "ValueError"):
            print(f"{name}: {exc}", file=sys.stderr)
        elif "GatedRepo" in name or "401" in str(exc) or "403" in str(exc):
            print("SAM3 model access is unavailable. Request access to facebook/sam3 and run hf auth login locally. Never put tokens in this command or chat.", file=sys.stderr)
        elif "OutOfMemory" in name:
            print("GPU memory exhausted. Reduce SAM3 object count or use the yolo-clip baseline; inspect the saved report.", file=sys.stderr)
        else:
            print(f"Experiment failed ({name}). Check --check-env, model access and PHASE1.md. A report is saved if initialization started.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
