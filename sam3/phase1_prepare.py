"""Download/check model assets before a video is available. No robot access."""
import argparse
import json
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import phase1_run  # Project-local model/config cache setup.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["sam3", "yolo-clip"], required=True)
    p.add_argument("--config-only", action="store_true", help="Check SAM3 checkpoint access without downloading weights")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--frames", type=int, default=3, help="Repeat the packaged image to check streaming state; not a motion benchmark")
    args = p.parse_args()
    if not 1 <= args.frames <= 30:
        p.error("--frames must be between 1 and 30")
    if args.config_only:
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download("facebook/sam3", "config.json")
        except Exception as exc:
            # Never dump token-bearing request objects or signed URLs.
            print(json.dumps({"sam3_config_access": False, "error_type": type(exc).__name__}))
            raise SystemExit(1)
        print(json.dumps({"sam3_config_access": True, "config_path": path}))
        return
    options = SimpleNamespace(device=args.device, query="person wearing a dark shirt",
                              reid_encoder=None, yolo=str(phase1_run.ROOT / "yolo11n.pt"),
                              sam3_model="facebook/sam3", max_objects=8)
    from phase1.models import Sam3Backend, YoloClipBackend
    try:
        print(f"Loading {args.backend} model assets...", flush=True)
        model = Sam3Backend(options) if args.backend == "sam3" else YoloClipBackend(options)
        import cv2
        import torch
        import ultralytics
        image_path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError("Packaged Ultralytics smoke-test image is unavailable")
        samples = []
        for index in range(args.frames):
            started = time.perf_counter()
            candidates = model.process(frame)
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            samples.append({"frame_index": index, "inference_seconds": elapsed,
                            "candidate_ids": [c.track_id for c in candidates]})
            print(f"Smoke frame {index+1}/{args.frames}: {len(candidates)} candidates, {elapsed:.2f}s", flush=True)
        result = {"backend": model.name, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "source": str(image_path), "candidate_count": len(candidates),
                  "samples": samples,
                  "postprocessing": getattr(model, "postprocessing", "backend_default"),
                  "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated()/2**30 if args.device == "cuda" else None,
                  "candidates": [{"tracker_id": c.track_id, "box": c.box, "semantic": c.semantic} for c in candidates],
                  "note": "Repeated static image for inference/state smoke testing only; not a real-video, clothing-change or accuracy benchmark."}
        output = phase1_run.ROOT / "outputs" / (args.backend+"_smoke.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(json.dumps({"model_load_or_smoke_success": False, "error_type": type(exc).__name__,
                          "stack": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                                    for f in traceback.extract_tb(exc.__traceback__)[-8:]]}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
