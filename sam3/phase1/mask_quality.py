"""Conservative removal of small disconnected mask islands far from a body.

Does not choose between two substantial regions, fill occlusions, or smooth
motion. These heuristics require held-out validation, especially carried items.
"""
import math


def remove_remote_specks(mask):
    import cv2
    import numpy as np
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError('Expected a two-dimensional binary mask')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count < 3:
        return mask, 0
    main = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[main]
    if area / max(1, binary.sum()) < .9:
        return mask, 0  # Multiple substantial pieces may be a real occlusion.
    discard = []
    for index in range(1, count):
        left, top, width, height, size = stats[index]
        gap = math.hypot(max(x-left-width, left-x-w, 0), max(y-top-height, top-y-h, 0))
        if index != main and size <= .05*area and gap > .5*max(w, h):
            discard.append(index)
    if not discard:
        return mask, 0
    cleaned = binary.astype(bool) & ~np.isin(labels, discard)
    return cleaned, int(binary.sum()-cleaned.sum())
