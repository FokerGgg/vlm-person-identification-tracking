from types import SimpleNamespace
import unittest

from phase1.sam3_history import trim_streaming_history, new_streaming_session


class HistoryTests(unittest.TestCase):
    def test_preserves_anchors_and_exact_recent_window(self):
        session = SimpleNamespace(
            processed_frames=dict.fromkeys(range(100)),
            output_dict_per_obj={0: {"cond_frame_outputs": {0: "anchor", 8: "correction"},
                                    "non_cond_frame_outputs": dict.fromkeys(range(100))}},
            frames_tracked_per_obj={0: dict.fromkeys(range(100))},
            point_inputs_per_obj={0: {0: "old-input"}}, mask_inputs_per_obj={0: {99: "new-input"}},
            obj_id_to_tracker_score_frame_wise=dict.fromkeys(range(100)),
            suppressed_obj_ids=dict.fromkeys(range(100)),
            unmatched_frame_inds={5: list(range(100))}, overlap_pair_to_frame_inds={(0, 5): [0, 60, 90]})
        trim_streaming_history(session, 99, 32)
        self.assertEqual(list(session.processed_frames), list(range(68, 100)))
        self.assertEqual(session.output_dict_per_obj[0]["cond_frame_outputs"], {0: "anchor", 8: "correction"})
        self.assertEqual(len(session.output_dict_per_obj[0]["non_cond_frame_outputs"]), 32)
        self.assertEqual(session.mask_inputs_per_obj[0], {99: "new-input"})
        self.assertEqual(session.overlap_pair_to_frame_inds[(0, 5)], [90])

    def test_unfilled_window_is_unchanged(self):
        trim_streaming_history(object(), 30, 32)

    def test_logical_frame_count_and_upstream_pointers_survive_eviction(self):
        import torch
        from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import Sam3TrackerVideoModel
        session = new_streaming_session(dtype=torch.float32)
        session.obj_id_to_idx(10)
        for i in range(100):
            session.add_new_frame(torch.zeros(3, 2, 2), frame_idx=i)
            key = "cond_frame_outputs" if i == 0 else "non_cond_frame_outputs"
            session.output_dict_per_obj[0][key][i] = {"object_pointer": torch.tensor([float(i)])}
        model = SimpleNamespace(config=SimpleNamespace(max_object_pointers_in_encoder=16), training=False)
        method = Sam3TrackerVideoModel._get_object_pointers
        before = method(model, session, 0, 100, session.num_frames, torch.device("cpu"), streaming=False)
        trim_streaming_history(session, 99, 32)
        self.assertEqual(len(session.processed_frames), 32)
        self.assertEqual(session.num_frames, 100)
        after = method(model, session, 0, 100, session.num_frames, torch.device("cpu"), streaming=False)
        self.assertEqual(before[0], after[0])
        self.assertTrue(torch.equal(torch.stack(before[1]), torch.stack(after[1])))
        session.add_new_frame(torch.zeros(3, 2, 2), frame_idx=100)
        self.assertEqual(session.num_frames, 101)


if __name__ == "__main__":
    unittest.main()
