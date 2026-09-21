"""Inference adapters. SAM3 means facebook/sam3 in Transformers, NOT SAM3.1."""
from __future__ import annotations

from pathlib import Path

from .identity import Candidate, iou


def add_overlaps(candidates):
    for current in candidates:
        a = current.box
        area = (a[2]-a[0])*(a[3]-a[1])
        current.overlap = max((
            max(0, min(a[2], b.box[2])-max(a[0], b.box[0])) *
            max(0, min(a[3], b.box[3])-max(a[1], b.box[1])) / area
            for b in candidates if b is not current), default=0.0)


class AppearanceEncoder:
    """Whole/lower-body CLIP baseline; optional local TorchScript ReID encoder.

    TorchScript contract: RGB, float32 [N,3,256,128], ImageNet normalization;
    return a finite [N,D] embedding tensor. Export/adapt your trained model to
    this contract first. Arbitrary training .pth files are not compatible.
    """
    def __init__(self, device, query, reid_path=None):
        import torch
        import open_clip
        self.torch, self.device = torch, device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.model = self.model.to(device).eval()
        texts = [query, "a different person", "a generic pedestrian", "a mannequin or robot"]
        tokens = open_clip.get_tokenizer("ViT-B-32")(texts).to(device)
        with torch.inference_mode():
            self.text_features = self._normalize(self.model.encode_text(tokens).float())
        self.reid = None
        if reid_path:
            self.reid = torch.jit.load(str(Path(reid_path)), map_location=device).eval()
        self.kind = "torchscript-reid" if self.reid is not None else "clip-lower-body-appearance-baseline"

    def _normalize(self, value):
        return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def encode(self, frame, candidates, semantic=True, masks=None):
        if not candidates:
            return
        import cv2
        from PIL import Image
        torch = self.torch
        height, width = frame.shape[:2]
        full, lower = [], []
        for c in candidates:
            x1, y1, x2, y2 = c.box
            x1, y1 = max(0, int(x1*width)), max(0, int(y1*height))
            x2, y2 = min(width, max(x1+1, int(x2*width))), min(height, max(y1+1, int(y2*height)))
            pixels = frame[y1:y2, x1:x2].copy()
            if masks is not None:
                # Do not embed an overlapping pedestrian/background as part of
                # the target. The visible silhouette is not an occlusion oracle.
                pixels[~masks[c.track_id][y1:y2, x1:x2]] = 127
            crop = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
            full.append(crop)
            lower.append(crop.crop((0, int(crop.height*0.55), crop.width, crop.height)))
        batch = torch.stack([self.preprocess(im) for im in full+lower]).to(self.device)
        with torch.inference_mode():
            features = self._normalize(self.model.encode_image(batch).float())
            whole, parts = features[:len(full)], features[len(full):]
            sims = whole @ self.text_features.T
            semantics = (sims[:, 0]-sims[:, 1:].mean(dim=1)).cpu().tolist()
            if self.reid is not None:
                from torchvision.transforms import Compose, Resize, ToTensor, Normalize
                transform = Compose([Resize((256, 128)), ToTensor(),
                                     Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                reid_batch = torch.stack([transform(im) for im in full]).to(self.device)
                result = self.reid(reid_batch)
                if not isinstance(result, torch.Tensor) or result.ndim != 2 or result.shape[0] != len(full):
                    raise ValueError("ReID encoder must return a [N,D] tensor")
                if not torch.isfinite(result).all() or (result.norm(dim=1) <= 0).any():
                    raise ValueError("ReID encoder produced invalid/zero embeddings")
                # Use the dedicated identity embedding for both score components.
                whole = parts = self._normalize(result.float())
        for c, feature, part, score in zip(candidates, whole.cpu().tolist(), parts.cpu().tolist(), semantics):
            c.feature, c.stable_feature = tuple(feature), tuple(part)
            if semantic:
                c.semantic = float(score)


class YoloClipBackend:
    name = "yolo11n-bytetrack-openclip"
    def __init__(self, args):
        from ultralytics import YOLO
        if not Path(args.yolo).is_file():
            raise FileNotFoundError(f"YOLO weights not found: {args.yolo}")
        self.model = YOLO(args.yolo)
        self.device = args.device
        self.encoder = AppearanceEncoder(args.device, args.query, args.reid_encoder)

    def process(self, frame):
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                                   conf=0.20, device=self.device, verbose=False)[0]
        h, w = frame.shape[:2]
        candidates = []
        if results.boxes is not None and results.boxes.id is not None:
            for ident, box, score in zip(results.boxes.id.cpu().tolist(), results.boxes.xyxy.cpu().tolist(), results.boxes.conf.cpu().tolist()):
                box = tuple(max(0.0, min(1.0, v/scale)) for v, scale in zip(box, (w, h, w, h)))
                if box[2] > box[0] and box[3] > box[1]:
                    candidates.append(Candidate(int(ident), box, float(score), 0.0))
        add_overlaps(candidates)
        self.encoder.encode(frame, candidates)
        return candidates


class Sam3Backend:
    name = "sam3-transformers-causal-streaming"
    def __init__(self, args):
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor, Sam3VideoConfig
        from .sam3_postprocessing import configure
        self.postprocessing = configure(getattr(args, "sam3_postprocessing", "auto"), args.device)
        self.torch, self.device = torch, args.device
        self.query = args.query.strip()
        self.mask_cleanup = getattr(args, 'sam3_mask_cleanup', 'conservative')
        if self.query.lower() == "person":
            raise ValueError("Supply a clothing description, not only 'person'")
        dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
        config = Sam3VideoConfig.from_pretrained(args.sam3_model)
        config.max_num_objects = args.max_objects
        self.history_window = getattr(args, "sam3_history_frames", 32)
        required_history = max(config.tracker_config.num_maskmem, config.tracker_config.max_object_pointers_in_encoder)
        if self.history_window < required_history:
            raise ValueError(f"SAM3 history must be at least {required_history} frames for this checkpoint")
        self.frame_index = 0
        self.model = Sam3VideoModel.from_pretrained(args.sam3_model, config=config, dtype=dtype).to(args.device).eval()
        self.processor = Sam3VideoProcessor.from_pretrained(args.sam3_model)
        if len(self.processor.tokenizer(self.query).input_ids) > 32:
            raise ValueError("SAM3 uses short concept prompts; shorten the English clothing description to <=32 tokens")
        from .sam3_history import new_streaming_session
        self.session = new_streaming_session(
            inference_device=args.device, inference_state_device="cpu",
            video_storage_device="cpu", dtype=dtype)
        # Generic people remain candidates after clothes stop matching the query.
        self.processor.add_text_prompt(self.session, ["person", self.query])
        self.encoder = AppearanceEncoder(args.device, self.query, args.reid_encoder)
        self.encoder.kind += ":sam3-visible-mask-crops"

    def process(self, frame):
        import cv2
        torch = self.torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model(inference_session=self.session,
                                frame=inputs.pixel_values[0].to(self.model.dtype),
                                frame_idx=self.frame_index, reverse=False)
            result = self.processor.postprocess_outputs(self.session, output, original_sizes=inputs.original_sizes)
        h, w = frame.shape[:2]
        mapping = result.get("prompt_to_obj_ids")
        if mapping is None:
            raise RuntimeError("SAM3 prompt-to-object mapping unavailable; check the Transformers version")
        ids = result["object_ids"].cpu().tolist()
        boxes = result["boxes"].cpu().tolist()
        scores = result["scores"].cpu().tolist()
        masks = result["masks"].bool().cpu().numpy()
        mask_by_id = {int(i): mask for i, mask in zip(ids, masks)}
        self.last_masks = mask_by_id  # Current observation only; optional local audit archive.
        # Only the already-selected identity is refined. Keep the raw masks for
        # diagnostics, and leave SAM3's internal temporal state untouched.
        hint = getattr(self, 'target_hint', None)
        if getattr(self, 'mask_cleanup', 'conservative') == 'conservative' and hint in mask_by_id:
            from .mask_quality import remove_remote_specks
            cleaned, removed = remove_remote_specks(mask_by_id[hint])
            if removed:
                mask_by_id = {**mask_by_id, hint: cleaned}
        records = {}
        for ident, box, score in zip(ids, boxes, scores):
            # Processor boxes precede its per-prompt non-overlap constraint.
            # Recompute from the final visible mask; a suppressed mask may now
            # be empty even when the returned box still has positive area.
            ys, xs = mask_by_id[int(ident)].nonzero()
            if not len(xs):
                continue
            box = (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1)
            norm = tuple(max(0.0, min(1.0, v/scale)) for v, scale in zip(box, (w, h, w, h)))
            if norm[2] > norm[0] and norm[3] > norm[1]:
                records[int(ident)] = (norm, float(score))
        query_ids = [int(i) for i in mapping.get(self.query, []) if int(i) in records]
        candidates = []
        for ident in mapping.get("person", []):
            ident = int(ident)
            if ident not in records:
                continue
            box, score = records[ident]
            mask = mask_by_id[ident]
            semantic = max((records[q][1] for q in query_ids
                            if (mask & mask_by_id[q]).sum() / max(1, (mask | mask_by_id[q]).sum()) >= .5), default=0.0)
            candidates.append(Candidate(ident, box, score, semantic))
        # SAM3 masks are disjoint within the person prompt. Bbox intersection
        # cannot establish which person is occluded. Missing masks, confidence,
        # temporal continuity and appearance govern acceptance here; YOLO still
        # uses its conservative box-overlap gate because it has no masks.
        self.encoder.encode(frame, candidates, semantic=False, masks=mask_by_id)
        from .sam3_history import trim_streaming_history
        trim_streaming_history(self.session, self.frame_index, self.history_window)
        self.frame_index += 1
        return candidates
