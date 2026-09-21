"""Model-independent target identity state, using only past/current observations.

Scores are heuristic and require calibration on held-out videos. In the default
encoder, `stable_feature` is a lower-body CLIP crop, NOT biometric identity or
a clothes-invariant ReID model. An exported ReID encoder can replace it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


def cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    denominator = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    return sum(x*y for x, y in zip(a, b)) / denominator if denominator else 0.0


def iou(a, b) -> float:
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


@dataclass
class Candidate:
    track_id: int
    box: tuple[float, float, float, float]  # normalized xyxy
    confidence: float
    semantic: float
    feature: tuple[float, ...] = ()
    stable_feature: tuple[float, ...] = ()
    overlap: float = 0.0


@dataclass
class Config:
    acquire_threshold: float = 0.11
    acquire_margin: float = 0.025
    confirm_seconds: float = 0.15
    confirm_frames: int = 3
    recovery_seconds: float = 0.25
    recovery_frames: int = 4
    recovery_threshold: float = 0.72
    recovery_margin: float = 0.08
    keep_threshold: float = 0.55
    update_threshold: float = 0.75
    clean_overlap: float = 0.10
    detection_threshold: float = 0.35
    continuity_seconds: float = 0.5
    short_recovery_frames: int = 2
    short_recovery_iou: float = 0.45
    max_continuous_area_ratio: float = 2.5
    search_after_seconds: float = 2.0
    memory_interval: float = 0.5
    gallery_size: int = 16
    enable_recovery: bool = True

    def __post_init__(self):
        if self.gallery_size < 2 or min(self.confirm_frames, self.recovery_frames, self.short_recovery_frames) < 1:
            raise ValueError("gallery_size must be >=2 and confirmation counts positive")
        for name in ("confirm_seconds", "recovery_seconds", "memory_interval", "search_after_seconds"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.continuity_seconds <= 0:
            raise ValueError("continuity_seconds must be positive")
        if not 0 < self.short_recovery_iou <= 1 or self.max_continuous_area_ratio <= 1:
            raise ValueError("Invalid continuity geometry thresholds")


@dataclass
class Decision:
    state: str
    target_id: str | None = None
    tracker_id: int | None = None
    box: tuple[float, ...] | None = None
    identity_score: float | None = None
    event: str | None = None
    last_seen_seconds: float | None = None
    reason: str = ""


class IdentityManager:
    def __init__(self, config: Config):
        self.config = config
        self.locked_id: int | None = None
        self.acquired = False
        self.gallery: list[tuple[tuple, tuple]] = []
        self.last_box = None
        self.last_seen = None
        self.last_update = -math.inf
        self.previous_time = -math.inf
        self.pending_id = None
        self.pending_since = 0.0
        self.pending_hits = 0
        self.pending_time = -math.inf
        self.state = "SEARCHING"
        self.freeze_until = 0.0
        self.short_hits = 0

    def _reset_pending(self):
        self.pending_id = None
        self.pending_hits = 0
        self.pending_time = -math.inf

    def _confirm(self, track_id, time_s, recovery=False):
        if self.pending_id != track_id or time_s-self.pending_time > self.config.continuity_seconds:
            self.pending_id, self.pending_since, self.pending_hits = track_id, time_s, 0
        self.pending_hits += 1
        self.pending_time = time_s
        duration = self.config.recovery_seconds if recovery else self.config.confirm_seconds
        count = self.config.recovery_frames if recovery else self.config.confirm_frames
        return self.pending_hits >= count and time_s-self.pending_since + 1e-9 >= duration

    def _clean(self, candidate):
        return (candidate.track_id >= 0 and candidate.confidence >= self.config.detection_threshold
                and candidate.overlap <= self.config.clean_overlap)

    def _similarities(self, candidate):
        whole = max((cosine(candidate.feature, full) for full, _ in self.gallery), default=0.0)
        stable = max((cosine(candidate.stable_feature, part) for _, part in self.gallery), default=0.0)
        # A part can support an upper-body appearance change, but cannot alone
        # establish identity. Both components contribute to the recovery score.
        score = max(0.75*whole + 0.25*stable, 0.35*whole + 0.65*stable)
        return whole, stable, score

    def _remember(self, candidate, time_s):
        if time_s-self.last_update < self.config.memory_interval:
            return
        self.gallery.append((candidate.feature, candidate.stable_feature))
        if len(self.gallery) > self.config.gallery_size:
            del self.gallery[1]  # Preserve the acquisition anchor.
        self.last_update = time_s

    def _accept(self, candidate, time_s, score, event=None):
        self.locked_id = candidate.track_id
        self.last_box, self.last_seen = candidate.box, time_s
        self.state = "TRACKING"
        self._reset_pending()
        self.short_hits = 0
        return Decision("TRACKING", "target-1", candidate.track_id, candidate.box,
                        score, event, time_s, "Current observation passed identity checks")

    def _geometry(self, candidate, elapsed):
        box, previous = candidate.box, self.last_box
        distance = math.hypot((box[0]+box[2]-previous[0]-previous[2])/2,
                              (box[1]+box[3]-previous[1]-previous[3])/2)
        area = (box[2]-box[0])*(box[3]-box[1])
        prior_area = (previous[2]-previous[0])*(previous[3]-previous[1])
        ratio = max(area/prior_area, prior_area/area)
        return distance <= 0.08+0.5*elapsed and ratio <= self.config.max_continuous_area_ratio

    def update(self, candidates: list[Candidate], time_s: float) -> Decision:
        if not math.isfinite(time_s) or time_s <= self.previous_time:
            raise ValueError("Timestamps must be finite and strictly increasing")
        self.previous_time = time_s
        if len({c.track_id for c in candidates}) != len(candidates):
            raise ValueError("Duplicate candidate tracker IDs")
        for c in candidates:
            if len(c.box) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in c.box):
                raise ValueError("Boxes must be finite normalized xyxy coordinates")
            if c.box[2] <= c.box[0] or c.box[3] <= c.box[1]:
                raise ValueError("Candidate boxes must have positive area")
            values = (c.confidence, c.semantic, c.overlap, *c.feature, *c.stable_feature)
            if not all(math.isfinite(x) for x in values):
                raise ValueError("Candidate scores and features must be finite")

        clean = [c for c in candidates if self._clean(c)]
        if not self.acquired:
            ranked = sorted(clean, key=lambda c: c.semantic, reverse=True)
            best = ranked[0] if ranked else None
            ambiguous = len(ranked) > 1 and best.semantic-ranked[1].semantic < self.config.acquire_margin
            if best is None or best.semantic < self.config.acquire_threshold or ambiguous:
                self._reset_pending()
                self.state = "AMBIGUOUS" if ambiguous else "SEARCHING"
                return Decision(self.state, reason="No unique, clean language match")
            if not self._confirm(best.track_id, time_s):
                self.state = "CONFIRMING"
                return Decision(self.state, reason="Confirming the same language-matched track")
            self.acquired = True
            self._remember(best, time_s)
            return self._accept(best, time_s, None, "acquired")

        # The original clothing/semantic score is deliberately absent below.
        elapsed = time_s-self.last_seen
        same = next((c for c in clean if c.track_id == self.locked_id), None)
        recovering = self.state != "TRACKING"
        nearby_time = elapsed <= self.config.continuity_seconds
        # A one-frame mask explosion is not evidence that identity changed.
        # Keep the last trusted observation, never emit/extrapolate its old box.
        # After a glitch require repeated current evidence from the original ID,
        # matching geometry and appearance. Similar people elsewhere do not
        # create local ambiguity; overlapping challengers still veto recovery.
        if same is not None and recovering and nearby_time and self.config.enable_recovery:
            _, _, short_score = self._similarities(same)
            spatial_match = self._geometry(same, elapsed) and iou(same.box, self.last_box) >= self.config.short_recovery_iou
            challenger = any(c.track_id != self.locked_id
                             and iou(c.box, self.last_box) >= self.config.short_recovery_iou
                             and self._similarities(c)[2] >= short_score-self.config.recovery_margin
                             for c in clean)
            if spatial_match and short_score >= self.config.update_threshold and not challenger:
                self.short_hits += 1
                self._reset_pending()
                if self.short_hits >= self.config.short_recovery_frames:
                    self.freeze_until = time_s+self.config.continuity_seconds
                    return self._accept(same, time_s, short_score, "short_gap_recovered")
                self.state = "VERIFYING_CONTINUITY"
                return Decision(self.state, "target-1", identity_score=short_score,
                                last_seen_seconds=self.last_seen,
                                reason="Confirming local continuity after a brief rejected observation")
        self.short_hits = 0
        if same is not None and not recovering and elapsed <= self.config.continuity_seconds:
            whole, stable, score = self._similarities(same)
            plausible_motion = self._geometry(same, elapsed)
            if score >= self.config.keep_threshold and plausible_motion:
                event = "appearance_shift_possible" if whole < self.config.keep_threshold and stable >= self.config.update_threshold else None
                if time_s >= self.freeze_until and (score >= self.config.update_threshold or event):
                    self._remember(same, time_s)
                return self._accept(same, time_s, score, event)

        # Do not feed a rejected short-term geometry outlier straight back into
        # global recovery, where high CLIP similarity could accept the bad mask.
        eligible = [c for c in clean if not (nearby_time and c.track_id == self.locked_id
                                             and not self._geometry(c, elapsed))]
        ranked = sorted(((self._similarities(c)[2], c) for c in eligible), key=lambda x: x[0], reverse=True)
        best_score, best = ranked[0] if ranked else (0.0, None)
        ambiguous = len(ranked) > 1 and best_score-ranked[1][0] < self.config.recovery_margin
        if (self.config.enable_recovery and best is not None
                and best_score >= self.config.recovery_threshold and not ambiguous):
            if self._confirm(best.track_id, time_s, recovery=True):
                self.freeze_until = time_s + self.config.continuity_seconds
                return self._accept(best, time_s, best_score, "recovered")
            self.state = "VERIFYING"
            return Decision(self.state, "target-1", identity_score=best_score,
                            last_seen_seconds=self.last_seen, reason="Reappearance requires multiple observations")

        self._reset_pending()
        covered = any(c.track_id == self.locked_id and c.overlap > self.config.clean_overlap for c in candidates)
        self.state = ("AMBIGUOUS" if ambiguous and best_score >= self.config.recovery_threshold else
                      "OCCLUDED" if covered else
                      "LOST" if elapsed <= self.config.search_after_seconds else "SEARCHING_IDENTITY")
        return Decision(self.state, "target-1", last_seen_seconds=self.last_seen,
                        reason="Identity unconfirmed; no target box is emitted")
