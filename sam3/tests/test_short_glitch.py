"""Regression cases: don't turn a mask glitch into long loss or an ID swap."""
import copy
import unittest

from phase1.identity import Config, IdentityManager
from tests.test_phase1 import person


class ShortGlitchTests(unittest.TestCase):
    def acquired(self, **kwargs):
        m = IdentityManager(Config(confirm_frames=1, confirm_seconds=0, **kwargs))
        m.update([person()], 0.)
        return m

    def test_bad_mask_never_emitted_and_local_continuity_beats_distant_lookalike(self):
        m = self.acquired()
        before = copy.deepcopy(m.gallery)
        bad = person()
        bad.box = (0., 0., .95, 1.)
        self.assertIsNone(m.update([bad], .1).box)
        self.assertEqual(m.last_seen, 0.)
        self.assertEqual(m.gallery, before)
        # Appearance-only global ranking is tied, but the tracks are far apart.
        candidates = [person(), person(ident=2, x=.7)]
        self.assertIsNone(m.update(candidates, .2).box)
        d = m.update(candidates, .3)
        self.assertEqual((d.event, d.tracker_id), ("short_gap_recovered", 1))
        self.assertEqual(m.gallery, before)

    def test_same_id_wrong_appearance_does_not_resume(self):
        m = self.acquired()
        m.update([], .1)
        for t in (.2, .3):
            self.assertIsNone(m.update([person(feature=(0., 1.), part=(0., 1.))], t).box)

    def test_crossing_lookalike_blocks_local_recovery(self):
        m = self.acquired()
        m.update([], .1)
        for t in (.2, .3):
            self.assertIsNone(m.update([person(), person(ident=2, x=.21)], t).box)

    def test_long_absence_still_requires_unique_global_identity(self):
        m = self.acquired()
        m.update([], .1)
        for t in (1., 1.1, 1.2, 1.3):
            self.assertIsNone(m.update([person(), person(ident=2, x=.7)], t).box)

    def test_confirmation_is_consecutive_and_no_recovery_ablation_preserved(self):
        m = self.acquired()
        m.update([], .1)
        m.update([person()], .2)
        m.update([], .3)
        self.assertIsNone(m.update([person()], .4).box)
        n = self.acquired(enable_recovery=False)
        n.update([], .1)
        for t in (.2, .3, .4):
            self.assertIsNone(n.update([person()], t).box)


if __name__ == "__main__":
    unittest.main()
