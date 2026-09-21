"""Replay identity decisions from saved observations; never reruns the detector.

Allows controlled ablation on exactly the same masks/embeddings. Scores from
this development replay are not an independent test set or a speed benchmark.
"""
import argparse
from collections import Counter
from dataclasses import asdict, fields
import gzip
import importlib.util
import json
from pathlib import Path
import sys

from phase1 import identity


def replay(source, output, implementation=None, refine_target_masks=False):
    source, output = Path(source), Path(output)
    original = json.loads((source / 'report.json').read_text(encoding='utf-8'))
    if original['status'] != 'completed':
        raise ValueError('Replay requires a completed observation run')
    module = identity
    if implementation:
        spec = importlib.util.spec_from_file_location('local_identity_ablation', Path(implementation).resolve())
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    names = {f.name for f in fields(module.Config)}
    config = module.Config(**{k: v for k, v in original['config'].items() if k in names})
    manager = module.IdentityManager(config)
    output.mkdir(parents=True, exist_ok=False)
    encoder = capture = None
    refined = []
    source_index = -1
    if refine_target_masks:
        import phase1_run
        import cv2
        import numpy as np
        from phase1.models import AppearanceEncoder
        from phase1.mask_quality import remove_remote_specks
        encoder = AppearanceEncoder('cuda', original['query'], original['arguments'].get('reid_encoder'))
        capture = cv2.VideoCapture(original['video'])
    states, events, count = Counter(), Counter(), 0
    with gzip.open(source / 'observations.jsonl.gz', 'rt', encoding='utf-8') as observations, (output / 'predictions.jsonl').open('w', encoding='utf-8') as log:
        for observation in observations:
            row = json.loads(observation)
            candidates = [module.Candidate(**c) for c in row['candidates']]
            if refine_target_masks:
                while source_index < row['frame_index']:
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError('Source decode failed during mask replay')
                    source_index += 1
                selected = next((c for c in candidates if c.track_id == manager.locked_id), None)
                path = source / 'mask_audit' / f"{row['frame_index']:06d}.npz"
                if selected is not None:
                    if not path.is_file():
                        raise ValueError('Required target mask is missing; cannot reproduce refinement')
                    with np.load(path) as data:
                        if int(data['tracker_id']) != selected.track_id:
                            raise ValueError('Replay target differs from archived mask identity')
                        raw = np.unpackbits(data['bits'], count=int(np.prod(data['shape']))).reshape(tuple(data['shape'])).astype(bool)
                    cleaned, removed = remove_remote_specks(raw)
                    if removed:
                        ys, xs = cleaned.nonzero()
                        h, w = cleaned.shape
                        selected.box = (float(xs.min()/w), float(ys.min()/h), float((xs.max()+1)/w), float((ys.max()+1)/h))
                        encoder.encode(frame, [selected], semantic=False, masks={selected.track_id: cleaned})
                        refined.append({'frame_index': row['frame_index'], 'removed_pixels': removed, 'box': selected.box})
            decision = manager.update(candidates, row['time_seconds'])
            result = {"frame_index": row['frame_index'], "time_seconds": row['time_seconds'], **asdict(decision),
                      "candidates": [{"tracker_id": c.track_id, "box": c.box, "confidence": c.confidence,
                                      "semantic": c.semantic, "overlap": c.overlap} for c in candidates]}
            log.write(json.dumps(result, allow_nan=False)+'\n')
            states[decision.state] += 1
            if decision.event:
                events[decision.event] += 1
            count += 1
    if count != original['processed_frames']:
        raise ValueError('Observation archive is incomplete')
    if capture:
        capture.release()
    report = {**original, "config": asdict(config), "state_frames": dict(states), "events": dict(events),
              "source_observation_run": str(source.resolve()), "model_inference_reused": True,
              "identity_implementation": str(Path(implementation).resolve()) if implementation else str(Path(identity.__file__).resolve())}
    report['refined_target_mask_frames'] = refined
    if refine_target_masks:
        report['arguments'] = {**report['arguments'], 'sam3_mask_cleanup': 'conservative'}
        report['limitations'] = report.get('limitations', []) + ['Cached SAM3 outputs; recomputed target-mask cleanup and CLIP embedding only where pixels changed. No new SAM3 inference.']
    for key in ('performance', 'peak_gpu_allocated_gib', 'peak_gpu_reserved_gib', 'initialization_seconds'):
        report.pop(key, None)
    report['limitations'] = report.get('limitations', []) + ['Identity-only replay on recorded observations; not new model inference or speed validation.']
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({"frames": count, "states": dict(states), "events": dict(events)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--identity-implementation', help='Optional trusted local Python implementation for ablation')
    p.add_argument('--refine-target-masks', action='store_true')
    args = p.parse_args()
    replay(args.run, args.output, args.identity_implementation, args.refine_target_masks)
