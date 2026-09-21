import copy
import math
import unittest

from phase1.identity import Candidate, Config, IdentityManager
from phase1_evaluate import evaluate


def person(ident=1, semantic=0.3, feature=(1., 0.), part=(1., 0.), x=0.2, overlap=0.):
    return Candidate(ident, (x, 0.1, x+0.15, 0.9), 0.95, semantic, feature, part, overlap)


class IdentityTests(unittest.TestCase):
    def manager(self, **kwargs):
        return IdentityManager(Config(confirm_frames=2, confirm_seconds=0.1,
                                      recovery_frames=2, recovery_seconds=0.1, **kwargs))

    def acquired(self, **kwargs):
        manager = self.manager(**kwargs)
        self.assertIsNone(manager.update([person()], 0.).box)
        self.assertEqual(manager.update([person()], 0.1).event, "acquired")
        return manager

    def test_confirmation_requires_same_person(self):
        m = self.manager()
        for i in range(10):
            self.assertIsNone(m.update([person(ident=1+i%2)], i*0.1).box)

    def test_absence_resets_acquisition_confirmation(self):
        m = self.manager()
        m.update([person()], 0.)
        m.update([], 0.1)
        self.assertEqual(m.update([person()], 0.2).state, "CONFIRMING")

    def test_multiple_matching_people_not_arbitrarily_selected(self):
        m = self.manager()
        self.assertEqual(m.update([person(), person(ident=2, x=0.6)], 0).state, "AMBIGUOUS")

    def test_changed_clothes_do_not_need_original_text(self):
        m = self.acquired()
        # Whole appearance changes while a local cue remains consistent.
        d = m.update([person(semantic=-0.8, feature=(0.4, math.sqrt(0.84)))], 0.2)
        self.assertEqual(d.state, "TRACKING")
        self.assertEqual(d.event, "appearance_shift_possible")

    def test_occlusion_freezes_memory_and_emits_no_stale_box(self):
        m = self.acquired()
        before = copy.deepcopy(m.gallery)
        d = m.update([person(overlap=0.6, feature=(0., 1.), part=(0., 1.))], 0.2)
        self.assertEqual(d.state, "OCCLUDED")
        self.assertIsNone(d.box)
        self.assertEqual(m.gallery, before)

    def test_reappearance_new_id_keeps_target_identity(self):
        m = self.acquired()
        m.update([], 0.2)
        self.assertEqual(m.update([person(ident=9, semantic=-1.)], 1.).state, "VERIFYING")
        d = m.update([person(ident=9, semantic=-1.)], 1.1)
        self.assertEqual((d.event, d.target_id, d.tracker_id), ("recovered", "target-1", 9))

    def test_clothes_matching_distractor_not_reacquired(self):
        m = self.acquired()
        for i in range(2, 10):
            d = m.update([person(ident=2, semantic=0.99, feature=(0., 1.), part=(0., 1.))], i*0.1)
            self.assertIsNone(d.box)

    def test_reused_tracker_id_does_not_bypass_identity(self):
        m = self.acquired()
        d = m.update([person(feature=(0., 1.), part=(0., 1.))], 0.2)
        self.assertIsNone(d.box)

    def test_ambiguous_recovery_does_not_poison_memory(self):
        m = self.acquired()
        m.update([], 0.2)
        before = copy.deepcopy(m.gallery)
        d = m.update([person(ident=2), person(ident=3, x=0.6)], 0.3)
        self.assertEqual(d.state, "AMBIGUOUS")
        self.assertEqual(m.gallery, before)

    def test_long_loss_never_returns_to_clothing_acquisition(self):
        m = self.acquired()
        d = m.update([], 10.)
        self.assertEqual(d.state, "SEARCHING_IDENTITY")
        self.assertEqual(d.target_id, "target-1")

    def test_clock_gaps_break_confirmation(self):
        m = self.manager()
        m.update([person()], 0)
        self.assertEqual(m.update([person()], 5).state, "CONFIRMING")
        with self.assertRaises(ValueError):
            m.update([], 5)

    def test_gallery_bounded_anchor_preserved(self):
        m = self.acquired(gallery_size=3, memory_interval=0)
        first = m.gallery[0]
        for i in range(2, 20):
            m.update([person()], i*0.1)
        self.assertEqual(len(m.gallery), 3)
        self.assertEqual(m.gallery[0], first)

    def test_no_recovery_ablation(self):
        m = self.acquired(enable_recovery=False)
        m.update([], 0.2)
        for i in range(3, 10):
            self.assertIsNone(m.update([person()], i*0.1).box)

    def test_invalid_observations_rejected(self):
        c = person()
        c.feature = (float("nan"), 1.)
        with self.assertRaises(ValueError):
            self.manager().update([c], 0)


class EvaluationTests(unittest.TestCase):
    def test_absence_wrong_person_and_recovery_are_separate(self):
        box = [0.1, 0.1, 0.3, 0.9]
        other = [0.6, 0.1, 0.8, 0.9]
        labels = {i: {"target_box": b} for i, b in enumerate([box, None, None, box, box, box])}
        predictions = {i: {"box": b, "time_seconds": i/10} for i, b in enumerate([box, box, None, other, None, box])}
        result = evaluate(predictions, labels)
        self.assertEqual(result["counts"]["wrong_box_frames"], 1)
        self.assertEqual(result["false_follow_when_absent_rate"], 0.5)
        self.assertEqual(result["visible_target_success_rate"], 0.5)
        self.assertAlmostEqual(result["reappearance_episodes"][0]["recovery_seconds"], 0.2)

    def test_empty_predictions_cannot_count_as_success(self):
        result = evaluate({0: {"box": None, "time_seconds": 0}}, {0: {"target_box": [0., 0., 1., 1.]}})
        self.assertEqual(result["visible_target_success_rate"], 0.)
        self.assertIsNone(result["first_emission_on_labeled_frames"])

    def test_missing_predicted_label_frame_is_an_error(self):
        with self.assertRaises(ValueError):
            evaluate({}, {0: {"target_box": None}})


if __name__ == "__main__":
    unittest.main()
