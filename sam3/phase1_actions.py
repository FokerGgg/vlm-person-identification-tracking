"""Independent local video-VLM probe; does not change tracking decisions.

Each clip starts a fresh conversation. Expected answers/labels are never sent
to the model. This is an offline diagnostic available only at the clip end,
not evidence of frame-by-frame causal event recognition or robust ReID.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import phase1_run  # Project-local cache configuration, no model initialization.


PROMPT = (
    "The reference image shows the target person near the beginning of this clip. "
    "Follow that person in the video and describe their observed actions in temporal order. "
    "Report changes to worn or carried items only when directly visible. Distinguish an "
    "action from an item merely being present. Do not infer actions during occlusion. "
    "If no change is observed, say so. If the person cannot be followed reliably, state the uncertainty. "
    "Describe ordinary movements too, even if there are no clothing changes. "
    "Use the video's absolute source timestamps in seconds. Return one JSON object with keys: "
    "actions (list of objects with start_seconds, end_seconds, description), "
    "clothing_before (a text description of worn and carried items at the start), "
    "clothing_after (a text description of worn and carried items at the end), "
    "visibility (text), uncertainty (text explaining any limitations)."
)

OPEN_PROMPT = (
    "The reference image identifies the target person at the start of the video clip. "
    "Describe what this person does in the video in temporal order. Look closely at their "
    "movements and what happens to their clothing and carried items. Describe what is "
    "actually visible; if nothing changes or if the action is unclear, say so. "
    "Answer in plain English, at most 150 words."
)


def prepare_case(video, predictions, start, end, sample_fps, target_view=False):
    import cv2
    import numpy as np
    from PIL import Image
    if not all(math.isfinite(x) for x in (start, end, sample_fps)) or start < 0 or end <= start or sample_fps <= 0:
        raise ValueError("Require finite 0 <= start < end and positive sampling FPS")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open video")
        fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or end > count/fps or sample_fps > fps:
            raise ValueError("Clip/sampling rate outside source video")
        first = math.ceil(start*fps)
        past = [r for r in predictions if r['frame_index'] <= first and r.get('box') is not None]
        if not past or first-past[-1]['frame_index'] > fps*.5:
            raise ValueError("No recent confirmed target to use as the reference image")
        reference = past[-1]
        indices = sorted(set(math.ceil(t*fps) for t in np.arange(start, end, 1/sample_fps)
                             if first <= math.ceil(t*fps) < min(count, math.ceil(end*fps))))
        if len(indices) < 4:
            raise ValueError("Use a clip with at least four sampled frames")
        wanted = set(indices)
        frames, crop = [], None
        by_index = {r['frame_index']: r for r in predictions}
        last_observation = None
        for index in range(indices[-1]+1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Unexpected video decode failure")
            if index in by_index:
                last_observation = by_index[index]
            if index == reference['frame_index']:
                h, w = frame.shape[:2]
                box = reference['box']
                x1, y1, x2, y2 = (int(v*s) for v, s in zip(box, (w, h, w, h)))
                crop = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            if index in wanted:
                if target_view:
                    # Current/past tracker boxes only. A missing decision yields
                    # a blank crop; no interpolation across loss or future frame.
                    canvas = np.full((640, 384, 3), 127, dtype=np.uint8)
                    if last_observation and last_observation.get('box') and index-last_observation['frame_index'] <= fps*.5:
                        h, w = frame.shape[:2]
                        x1, y1, x2, y2 = last_observation['box']
                        dx, dy = (x2-x1)*.35, (y2-y1)*.15
                        left, top = max(0, int((x1-dx)*w)), max(0, int((y1-dy)*h))
                        right, bottom = min(w, math.ceil((x2+dx)*w)), min(h, math.ceil((y2+dy)*h))
                        patch = frame[top:bottom, left:right]
                        scale = min(384/patch.shape[1], 640/patch.shape[0])
                        patch = cv2.resize(patch, (max(1, round(patch.shape[1]*scale)), max(1, round(patch.shape[0]*scale))))
                        py, px = (640-patch.shape[0])//2, (384-patch.shape[1])//2
                        canvas[py:py+patch.shape[0], px:px+patch.shape[1]] = patch
                    frame = canvas
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if crop is None or len(frames) != len(indices):
            raise ValueError("Incomplete video/reference sampling")
        metadata = dict(total_num_frames=count, fps=fps, width=frames[0].shape[1], height=frames[0].shape[0],
                        duration=count/fps, frames_indices=indices, video_backend="opencv_cfr_sequential")
        audit = dict(frame_indices=indices, frame_times_seconds=[i/fps for i in indices],
                     reference_frame_index=reference['frame_index'], reference_box=reference['box'],
                     reference_tracker_id=reference.get('tracker_id'),
                     target_view=target_view,
                     information_available_at_seconds=indices[-1]/fps)
        return crop, np.stack(frames), metadata, audit
    finally:
        capture.release()


def make_inputs(processor, reference, frames, metadata, prompt=PROMPT):
    reference_content = [] if reference is None else [{"type": "text", "text": "Reference person:"}, {"type": "image"}]
    messages = [{"role": "user", "content": reference_content + [
        {"type": "text", "text": "Video clip:"}, {"type": "video"},
        {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[reference] if reference is not None else None, videos=[frames],
                       video_metadata=[metadata], do_sample_frames=False,
                       videos_kwargs={"size": {"shortest_edge": 128*32*32,
                                                "longest_edge": len(frames)*640*384},
                                      "cap_pixels_per_frame": True},
                       images_kwargs={"size": {"shortest_edge": 128*32*32, "longest_edge": 256*32*32}},
                       return_tensors="pt")
    return inputs


def parse_response(text):
    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) and isinstance(result.get('actions'), list) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--cases', required=True, help='JSON list of {id,start,end}; other fields never reach model')
    p.add_argument('--output', required=True, help='New directory')
    p.add_argument('--model', default='Qwen/Qwen3-VL-4B-Instruct')
    p.add_argument('--sample-fps', type=float, default=2.)
    p.add_argument('--target-view', action='store_true', help='Past/current tracker crops with context padding; isolates action recognition from scene search')
    p.add_argument('--prompt-mode', choices=['structured', 'open'], default='open')
    p.add_argument('--no-reference-image', action='store_true', help='Target-view only: video without a static appearance anchor')
    p.add_argument('--prepare-only', action='store_true', help='Validate actual processor inputs without loading weights')
    args = p.parse_args()
    if args.no_reference_image and not args.target_view:
        p.error('--no-reference-image requires --target-view so the target remains specified')
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    video = Path(args.video).resolve()
    predictions = [json.loads(line) for line in Path(args.predictions).read_text(encoding='utf-8').splitlines()]
    cases = json.loads(Path(args.cases).read_text(encoding='utf-8'))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = None
    prompt = OPEN_PROMPT if args.prompt_mode == 'open' else PROMPT
    if args.no_reference_image:
        prompt = 'This video is centered on one selected person using a visual tracker. ' + prompt.split('. ', 1)[1]
    report = {"status": "running", "model": args.model, "video": str(video), "prompt": prompt,
              "prompt_mode": args.prompt_mode, "target_view": args.target_view,
              "reference_image_in_model_input": not args.no_reference_image,
              "sample_fps": args.sample_fps, "generation": {"do_sample": False, "max_new_tokens": 512},
              "mode": "independent_offline_clip_probe", "affects_tracking": False,
              "expected_answers_in_model_input": False, "cases": [],
              "limitations": ["One video, selected diagnostic clips, not a held-out benchmark.",
                              "Target reference uses a prior tracker box; target selection is not tested here.",
                              "Answers are available after the final supplied frame; no online timing claim.",
                              "Action descriptions cannot by themselves prove identity after an unseen change."]}
    try:
        if not args.prepare_only:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required for this local BF16 probe')
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
            report['revision'] = model.config._commit_hash
            report['gpu'] = torch.cuda.get_device_name(0)
        with (out / 'responses.jsonl').open('w', encoding='utf-8') as log:
            for case in cases:
                # No expectations, scenario names, or previous model answers enter inputs.
                reference, frames, metadata, audit = prepare_case(
                    video, predictions, float(case['start']), float(case['end']), args.sample_fps, args.target_view)
                number = len(report['cases'])
                reference.save(out / f'reference_{number}.jpg')
                inputs = make_inputs(processor, None if args.no_reference_image else reference, frames, metadata, prompt)
                row = {"id": case['id'], **audit, "input_tokens": int(inputs.input_ids.shape[-1]),
                       "video_grid_thw": inputs.video_grid_thw.tolist()}
                if model is not None:
                    inputs = inputs.to('cuda')
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                    torch.cuda.synchronize()
                    response = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                    row.update(response=response, parsed=parse_response(response),
                               inference_seconds=time.perf_counter()-started,
                               peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                    del generated
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                log.flush()
                report['cases'].append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                del inputs
        report['status'] = 'prepared_only' if args.prepare_only else 'completed'
    except BaseException as exc:
        report['status'], report['error_type'] = 'failed', type(exc).__name__
        raise
    finally:
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
