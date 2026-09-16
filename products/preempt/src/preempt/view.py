"""Honesty rails: when the camera cannot see, the product says so and stops.

A monitoring system that keeps producing a green light with a bin bag over the
lens is worse than no system, because the ward stops checking. So every frame is
graded before anything is inferred from it, and four specific failures are named
rather than averaged away:

* **too dark** - a side room at 03:00 with the lights off. Mean V plus the shape
  of the V histogram, because a dark room and an underexposed bright room look
  different in the histogram and only one of them is recoverable.
* **blocked** - a curtain, a visitor's coat, a fogged lens. Whole-frame Laplacian
  variance, which collapses when there is nothing in focus anywhere.
* **camera moved** - the single most dangerous failure, because every zone a
  nurse drew is now pointing at the wrong furniture while the system carries on
  confidently. ORB against the first usable frame, median match displacement.
* **out of frame** - no usable pose for a while. Not a fault, but not a state we
  may report anything about either.
* **paused for personal care** - staff turned the camera off, deliberately. Not a
  fault at all, and the most important of the five to record honestly: falls
  cluster around bathing, toileting and dressing, which is exactly when a ward
  will pause. Every run reports how long it was blind by choice, and says that
  anything that happened in that window is unknowable rather than absent.

All four are computed with OpenCV: `calcHist`, `Laplacian`, `ORB` +
`BFMatcher`. None of them retain an image; the reference ORB descriptors live in
memory for the run and are not a picture (they are 32-byte binary strings per
keypoint) and are never written anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import Thresholds

USABLE = "usable"
TOO_DARK = "too dark"
BLOCKED = "blocked"
CAMERA_MOVED = "camera moved"
NO_PERSON = "person out of frame"
PAUSED = "paused for personal care"

VIEW_STATES = (USABLE, TOO_DARK, BLOCKED, CAMERA_MOVED, NO_PERSON, PAUSED)

FAULTS = (TOO_DARK, BLOCKED, CAMERA_MOVED)
"""The states somebody has to go and fix. `PAUSED` is not one of them."""

REMEDY = {
    TOO_DARK: "turn on the night light or move the camera off the window",
    BLOCKED: "check the lens for a curtain, a coat or condensation",
    CAMERA_MOVED: "the zones no longer match the room: re-draw bed, chair and door",
    NO_PERSON: "nobody is in view, so nothing is being assessed",
    PAUSED: (
        "staff paused the camera for personal care. Nothing is being assessed and "
        "anything that happens in this window is unknowable"
    ),
}


@dataclass(frozen=True)
class ViewReport:
    """What the camera could see at one instant, with the numbers behind it."""

    state: str
    time_s: float
    mean_v: float
    dark_fraction: float
    laplacian_var: float
    shift_px: float | None
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.state == USABLE

    @property
    def remedy(self) -> str:
        return REMEDY.get(self.state, "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "time_s": round(self.time_s, 3),
            "mean_v": round(self.mean_v, 1),
            "dark_fraction": round(self.dark_fraction, 3),
            "laplacian_var": round(self.laplacian_var, 1),
            "shift_px": None if self.shift_px is None else round(self.shift_px, 1),
            "detail": self.detail,
            "remedy": self.remedy,
        }


def exposure(image: np.ndarray) -> tuple[float, float]:
    """Mean V and the fraction of pixels in the darkest 16 V bins."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if image.ndim == 3 else None
    v = hsv[:, :, 2] if hsv is not None else image
    hist = cv2.calcHist([v], [0], None, [256], [0, 256]).flatten()
    total = float(hist.sum()) or 1.0
    return float(v.mean()), float(hist[:16].sum() / total)


def focus(image: np.ndarray) -> float:
    """Whole-frame Laplacian variance. Low means nothing is in focus anywhere."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class ViewMonitor:
    """Grades frames and remembers just enough to notice the camera being knocked."""

    def __init__(self, thresholds: Thresholds, *, max_features: int = 600) -> None:
        self.t = thresholds
        self._orb = cv2.ORB.create(nfeatures=max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._ref_kp: tuple[cv2.KeyPoint, ...] | None = None
        self._ref_desc: np.ndarray | None = None
        self._last_person_s: float | None = None
        self.moved_since_s: float | None = None

    # -- camera movement ----------------------------------------------------
    def _shift_px(self, gray: np.ndarray) -> float | None:
        kp, desc = self._orb.detectAndCompute(gray, None)
        if desc is None or len(kp) < 12:
            return None
        if self._ref_desc is None:
            self._ref_kp, self._ref_desc = tuple(kp), desc
            return 0.0
        matches = self._matcher.match(self._ref_desc, desc)
        if len(matches) < 12:
            return None
        matches = sorted(matches, key=lambda m: m.distance)[: max(12, len(matches) // 2)]
        src = np.array([self._ref_kp[m.queryIdx].pt for m in matches], dtype=np.float64)
        dst = np.array([kp[m.trainIdx].pt for m in matches], dtype=np.float64)
        return float(np.median(np.linalg.norm(dst - src, axis=1)))

    # -- the grade ----------------------------------------------------------
    def grade(self, image: np.ndarray, time_s: float, *, person_seen: bool) -> ViewReport:
        mean_v, dark_fraction = exposure(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        shift = self._shift_px(gray)
        if person_seen:
            self._last_person_s = time_s

        if mean_v < self.t.dark_mean_v or dark_fraction > self.t.dark_low_bin_fraction:
            return ViewReport(
                TOO_DARK, time_s, mean_v, dark_fraction, lap, shift,
                f"mean brightness {mean_v:.0f} of 255, "
                f"{dark_fraction * 100:.0f}% of the frame is near black",
            )
        if lap < self.t.blocked_laplacian_var:
            return ViewReport(
                BLOCKED, time_s, mean_v, dark_fraction, lap, shift,
                f"no detail anywhere in the frame (focus measure {lap:.1f})",
            )
        if shift is not None and shift > self.t.camera_moved_px:
            if self.moved_since_s is None:
                self.moved_since_s = time_s
            return ViewReport(
                CAMERA_MOVED, time_s, mean_v, dark_fraction, lap, shift,
                f"the view has shifted {shift:.0f} px since setup, "
                "so the zones no longer line up with the room",
            )
        if not person_seen:
            last = self._last_person_s
            gone_for = time_s - last if last is not None else time_s
            if gone_for >= self.t.absent_s:
                return ViewReport(
                    NO_PERSON, time_s, mean_v, dark_fraction, lap, shift,
                    f"nobody in view for {gone_for:.0f} s",
                )
        return ViewReport(USABLE, time_s, mean_v, dark_fraction, lap, shift)


    def grade_recorded(
        self, time_s: float, *, person_seen: bool, hint: str = USABLE
    ) -> ViewReport:
        """Grade an instant when there is no image to look at.

        A recorded pose track has no pixels by construction, so the three
        image-based rails cannot be re-derived from it. They were evaluated on
        the device at capture time, and the track carries their verdict as a
        hint. The fourth rail, nobody in view, is a property of the pose stream
        itself and is still computed here.
        """
        if person_seen:
            self._last_person_s = time_s
        if hint == PAUSED:
            return ViewReport(PAUSED, time_s, 0.0, 0.0, 0.0, None, REMEDY[PAUSED])
        if hint != USABLE:
            return ViewReport(hint, time_s, 0.0, 0.0, 0.0, None, REMEDY.get(hint, ""))
        last = self._last_person_s
        if not person_seen:
            gone_for = time_s - last if last is not None else time_s
            if gone_for >= self.t.absent_s:
                return ViewReport(
                    NO_PERSON, time_s, 0.0, 0.0, 0.0, None,
                    f"nobody in view for {gone_for:.0f} s",
                )
        return ViewReport(USABLE, time_s, 0.0, 0.0, 0.0, None)


def summarise(reports: list[ViewReport]) -> dict[str, Any]:
    """Seconds spent in each view state, for the run record and the report."""
    counts: dict[str, int] = dict.fromkeys(VIEW_STATES, 0)
    for r in reports:
        counts[r.state] = counts.get(r.state, 0) + 1
    total = max(1, len(reports))
    span = (reports[-1].time_s - reports[0].time_s) if len(reports) > 1 else 0.0
    per_sample = span / max(1, total - 1) if total > 1 else 0.0
    return {
        "frames": len(reports),
        "usable_fraction": round(counts[USABLE] / total, 4),
        "by_state": {k: v for k, v in counts.items() if v},
        "blind_by_choice_s": round(counts[PAUSED] * per_sample, 2),
        "faulty_s": round(sum(counts[s] for s in FAULTS) * per_sample, 2),
    }
