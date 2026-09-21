import json
from pathlib import Path
import tempfile
import unittest

from phase1_actions import prepare_case, parse_response, PROMPT


class ActionsProbeTests(unittest.TestCase):
    def test_sampling_timestamps_and_reference_never_use_future(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / 'clip.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 64))
            for i in range(90):
                writer.write(np.full((64, 64, 3), i, dtype=np.uint8))
            writer.release()
            predictions = [{'frame_index': 6, 'box': [.1, .1, .8, .8], 'tracker_id': 1},
                           {'frame_index': 60, 'box': [.5, .5, .9, .9], 'tracker_id': 99}]
            ref, frames, metadata, audit = prepare_case(video, predictions, .21, 2.8, 2)
            self.assertEqual(audit['reference_tracker_id'], 1)
            self.assertEqual(audit['frame_indices'][0], 7)
            self.assertEqual(len(frames), len(metadata['frames_indices']))
            self.assertTrue(all(.21 <= t < 2.8 for t in audit['frame_times_seconds']))
            self.assertAlmostEqual(audit['information_available_at_seconds'], audit['frame_indices'][-1]/30)
            _, focused, _, _ = prepare_case(video, predictions, .21, 2.8, 2, target_view=True)
            self.assertEqual(focused.shape[1:], (640, 384, 3))
            self.assertTrue(np.all(focused[1] == 127))  # Stale target expires; no future box interpolation.
            with self.assertRaises(ValueError):
                prepare_case(video, predictions, 2.8, 4., 2)

    def test_raw_response_parsing_does_not_fabricate_answers(self):
        self.assertIsNone(parse_response('I cannot tell.'))
        self.assertEqual(parse_response('```json\n{"actions": []}\n```'), {'actions': []})
        self.assertIsNone(parse_response('{"answer": "walking"}'))
        self.assertNotIn('remov', PROMPT.lower())
        self.assertNotIn('jacket', PROMPT.lower())


if __name__ == '__main__':
    unittest.main()
