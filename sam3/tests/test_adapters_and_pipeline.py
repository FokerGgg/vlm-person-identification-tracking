import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import phase1_run
from phase1.identity import Candidate


class Sam3AdapterTests(unittest.TestCase):
    """Exercise empty/multi-prompt outputs without needing gated SAM3 weights."""
    def run_adapter(self, mapping, ids, boxes, scores, target_hint=None, add_speck=False):
        import numpy as np
        import torch
        from phase1.models import Sam3Backend
        backend = Sam3Backend.__new__(Sam3Backend)
        backend.torch, backend.device, backend.query = torch, "cpu", "person in red"
        backend.session = object()
        backend.frame_index, backend.history_window = 0, 32
        backend.target_hint = target_hint
        class Inputs:
            pixel_values = torch.zeros(1, 3, 8, 8)
            original_sizes = [[100, 100]]
            def to(self, device):
                return self
        class Processor:
            def __call__(self, **kwargs):
                return Inputs()
            def postprocess_outputs(self, *args, **kwargs):
                masks = torch.zeros(len(ids), 100, 100, dtype=torch.bool)
                for index, (x1, y1, x2, y2) in enumerate(boxes):
                    masks[index, y1:y2, x1:x2] = True
                if add_speck:
                    masks[0, :5, :1] = True
                return {"object_ids": torch.tensor(ids), "boxes": torch.tensor(boxes),
                        "scores": torch.tensor(scores), "prompt_to_obj_ids": mapping, "masks": masks}
        class Model:
            dtype = torch.float32
            def __call__(self, **kwargs):
                assert kwargs["reverse"] is False
                assert kwargs["frame"].shape == (3, 8, 8)
                return None
        class Encoder:
            def encode(self, frame, candidates, semantic, masks):
                assert semantic is False
        backend.model, backend.processor, backend.encoder = Model(), Processor(), Encoder()
        return backend.process(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_empty_prompt_mapping_is_valid(self):
        self.assertEqual(self.run_adapter({}, [], [], []), [])

    def test_remote_speck_cleanup_only_uses_already_selected_identity(self):
        raw = self.run_adapter({'person': [1]}, [1], [[50, 20, 65, 80]], [.9], add_speck=True)
        clean = self.run_adapter({'person': [1]}, [1], [[50, 20, 65, 80]], [.9], target_hint=1, add_speck=True)
        self.assertEqual(raw[0].box, (0., 0., .65, .8))
        self.assertEqual(clean[0].box, (.5, .2, .65, .8))

    def test_generic_person_survives_missing_clothing_match(self):
        people = self.run_adapter({"person": [7]}, [7], [[10, 10, 30, 90]], [0.9])
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0].semantic, 0)

    def test_query_duplicates_do_not_become_extra_people(self):
        people = self.run_adapter({"person": [7], "person in red": [9]}, [7, 9],
                                  [[10, 10, 30, 90], [10, 10, 30, 90]], [0.9, 0.95])
        self.assertEqual([c.track_id for c in people], [7])
        self.assertAlmostEqual(people[0].semantic, 0.95)

    def test_invalid_empty_masks_are_dropped(self):
        self.assertEqual(self.run_adapter({"person": [1]}, [1], [[0, 0, 0, 0]], [0.9]), [])


class VideoPipelineTests(unittest.TestCase):
    def test_decode_decisions_logs_video_and_metrics(self):
        import cv2
        import numpy as np
        from phase1_evaluate import evaluate, load_rows
        from phase1.models import YoloClipBackend
        root = phase1_run.ROOT / "outputs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="pipeline_test_", dir=root) as directory:
            base = Path(directory)
            source = base / "fixture.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30., (320, 240))
            self.assertTrue(writer.isOpened())
            for _ in range(20):
                writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            writer.release()
            class Backend:
                encoder = SimpleNamespace(kind="synthetic-observations-for-plumbing-test")
                def __init__(self, args):
                    self.index = -1
                def process(self, frame):
                    self.index += 1
                    if 8 <= self.index < 10:
                        return []
                    return [Candidate(1 if self.index < 8 else 9, (0.2, 0.1, 0.5, 0.9),
                                      0.9, 0.3 if self.index < 8 else -0.5, (1., 0.), (1., 0.))]
            args = SimpleNamespace(video=str(source), query="person in red", backend="yolo-clip", output=str(base / "run"),
                                   device="cpu", yolo="unused", reid_encoder=None, max_frames=0, max_objects=8,
                                   fps=None, acquire_threshold=None, acquire_margin=None,
                                   recovery_threshold=0.72, recovery_margin=0.08, no_recovery=False, no_video=False,
                                   save_observations=True)
            with patch("phase1.models.YoloClipBackend", Backend), patch("phase1_run.environment", return_value={"synthetic_test": True}):
                phase1_run.run(args)
            report = json.loads((base / "run/report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["processed_frames"], 20)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["events"]["acquired"], 1)
            self.assertEqual(report["events"]["recovered"], 1)
            rows = load_rows(base / "run/predictions.jsonl")
            from phase1_replay import replay
            replay(base / 'run', base / 'replay')
            replayed = load_rows(base / 'replay/predictions.jsonl')
            for index, original in rows.items():
                for field in ('state', 'box', 'tracker_id', 'event', 'identity_score'):
                    self.assertEqual(replayed[index][field], original[field])
            self.assertIsNone(rows[8]["box"])
            self.assertEqual(rows[19]["target_id"], "target-1")
            labels = {i: {"target_box": None if 8 <= i < 10 else [0.2, 0.1, 0.5, 0.9]} for i in range(20)}
            evaluation = evaluate(rows, labels)
            self.assertEqual(evaluation["false_follow_when_absent_rate"], 0.)
            cap = cv2.VideoCapture(str(base / "run/annotated.mp4"))
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 30., places=2)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 20)
            ok, frame = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[:2], (240, 320))
            with patch("phase1.models.YoloClipBackend", Backend):
                with self.assertRaises(FileExistsError):
                    phase1_run.run(args)
            args.output, args.frame_stride = str(base / "strided"), 3
            with patch("phase1.models.YoloClipBackend", Backend), patch("phase1_run.environment", return_value={}):
                phase1_run.run(args)
            sampled = load_rows(base / "strided/predictions.jsonl")
            self.assertEqual(sorted(sampled), [0, 3, 6, 9, 12, 15, 18])
            self.assertAlmostEqual(sampled[18]["time_seconds"], .6)
            cap = cv2.VideoCapture(str(base / "strided/annotated.mp4"))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 7)
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 10.)
            cap.release()
            info = json.loads((base / "strided/report.json").read_text())
            self.assertEqual(info["decoded_source_frames"], 20)
            self.assertEqual(info["status"], "completed")
            recovery_predictions = {i: {"box": None if i < 6 else [.2, .1, .5, .9], "time_seconds": i/30}
                                    for i in (0, 3, 6, 9)}
            recovery_labels = {i: {"target_box": None if i == 0 else [.2, .1, .5, .9]} for i in (0, 3, 6, 9)}
            scored = evaluate(recovery_predictions, recovery_labels, frame_step=3)
            self.assertAlmostEqual(scored["reappearance_episodes"][0]["recovery_seconds"], .1)
            self.assertEqual(evaluate(recovery_predictions, recovery_labels)["reappearance_episodes"], [])


if __name__ == "__main__":
    unittest.main()
