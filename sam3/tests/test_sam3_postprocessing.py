import unittest
import torch

from phase1.sam3_postprocessing import PortableCVUtils


class PortablePostprocessingTests(unittest.TestCase):
    def test_mask_nms_suppresses_duplicate_keeps_disjoint_and_ties(self):
        overlaps = torch.tensor([[1., .9, 0.], [.9, 1., 0.], [0., 0., 1.]])
        scores = torch.tensor([.8, .9, .9])
        self.assertEqual(PortableCVUtils.generic_nms(overlaps, scores, .5, True).tolist(), [1, 2])
        self.assertEqual(PortableCVUtils.generic_nms(torch.empty(0, 0), torch.empty(0), .5, True).tolist(), [])

    def test_components_are_eight_connected_and_batch_independent(self):
        mask = torch.zeros(2, 1, 7, 7, dtype=torch.uint8)
        mask[0, 0, 1, 1] = mask[0, 0, 2, 2] = mask[0, 0, 5, 5] = 1
        mask[1] = 1
        labels, counts = PortableCVUtils.cc_2d(mask)
        self.assertEqual(counts[0, 0, 1, 1], 2)
        self.assertEqual(labels[0, 0, 1, 1], labels[0, 0, 2, 2])
        self.assertEqual(counts[0, 0, 5, 5], 1)
        self.assertEqual(counts[0, 0, 0, 0], 0)
        self.assertTrue((counts[1] == 49).all())

    def test_upstream_postprocessor_fills_holes_and_removes_specks(self):
        from unittest.mock import patch
        from transformers.models.sam3_video import modeling_sam3_video as module
        scores = torch.full((1, 1, 9, 9), -1.)
        scores[:, :, 2:7, 2:7] = 1.
        scores[:, :, 4, 4] = -1.
        scores[:, :, 0, 0] = 1.
        with patch.object(module, "cv_utils_kernel", PortableCVUtils()):
            result = module.fill_holes_in_mask_scores(scores, max_area=2)
        self.assertGreater(result[0, 0, 4, 4], 0)
        self.assertLess(result[0, 0, 0, 0], 0)
        self.assertGreater(result[0, 0, 2, 2], 0)


if __name__ == "__main__":
    unittest.main()
