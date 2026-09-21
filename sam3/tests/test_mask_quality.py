import unittest
import numpy as np
from phase1.mask_quality import remove_remote_specks


class MaskQualityTests(unittest.TestCase):
    def test_remote_noise_removed_without_altering_body(self):
        mask = np.zeros((300, 500), bool)
        mask[50:250, 200:280] = True
        mask[0:20, 0:10] = True
        cleaned, removed = remove_remote_specks(mask)
        self.assertEqual(removed, 200)
        self.assertTrue(cleaned[50:250, 200:280].all())
        self.assertTrue(mask[:20, :10].all())  # Input preserved for the audit.

    def test_nearby_disconnected_hand_or_bag_preserved(self):
        mask = np.zeros((300, 500), bool)
        mask[50:250, 200:280] = True
        mask[130:150, 290:310] = True
        cleaned, removed = remove_remote_specks(mask)
        self.assertEqual(removed, 0)
        np.testing.assert_array_equal(cleaned, mask)

    def test_two_substantial_pieces_not_arbitrarily_reduced_to_largest(self):
        mask = np.zeros((300, 500), bool)
        mask[50:200, 300:380] = True
        mask[100:150, 20:80] = True
        cleaned, removed = remove_remote_specks(mask)
        self.assertEqual(removed, 0)
        np.testing.assert_array_equal(cleaned, mask)

    def test_empty_mask_remains_empty(self):
        cleaned, removed = remove_remote_specks(np.zeros((8, 8), bool))
        self.assertEqual(removed, 0)
        self.assertFalse(cleaned.any())
