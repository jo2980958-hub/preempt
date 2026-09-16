"""Unsteady gait, scored over seconds rather than read off one frame.

Nobody can tell from a still whether a person is steady. What a physiotherapist
watches for is a pattern over a few strides: the body swinging wider than the
path, steps that arrive at uneven times, a pause mid-stride, a hand going out to
a wall that is not there yet. All four are computed here over a sliding window,
and the score is the fusion of them with the individual reasons kept, because
"gait score 0.71" is useless to a nurse and "swaying 11 cm and reaching for the
wall" is not.

The measures
------------
**Sway.** `cv2.fitLine` gives the walking line through the centre-of-mass floor
track over the window, robustly (DIST_L2 with iterative reweighting), and sway is
the RMS perpendicular distance from it in centimetres. Fitting the line means
walking round a corner is not scored as sway, which a fixed-axis measure would
get wrong.

**Step timing, measured and then deliberately not scored.** Footfalls are the
peaks of the distance between the two feet on the floor plane: the gap closes as
one foot swings past the other and opens at each double support, so one peak is
one step. The coefficient of variation of those intervals is the standard
clinical gait-variability statistic and it was in the score until the evaluation
was run properly.

It had to come out. At the fifteen hertz this pipeline samples at, a 1.9 Hz
cadence gives about eight samples a step, and the peak of a smoothed gap signal
lands one sample either side at random. Measured on synthetic walks that are
steady by construction, the CV came out between 0.26 and 0.52, which is the same
magnitude as the clinical effect it was supposed to detect, and it produced two
false alarms in twenty quiet sequences on its own. So step timing is reported for
a clinician to look at and contributes nothing to the score, and
`docs/evaluation.md` says why. Sway separates the same cases cleanly: 2.3 to
3.8 cm steady against 10.8 to 12.7 cm unsteady.

**Reaching for support.** A wrist whose floor projection sits within
`wall_reach_m` of a wall polygon and stays there. Held, not instantaneous, so
brushing past a wall is not a reason.

**Sway frequency.** `cv2.dft` on the lateral deviation. Reported, not scored: the
literature on frequency bands in this setting is not strong enough to hang a call
on, and saying so is cheaper than pretending.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .config import Thresholds, Zone
from .geometry import FloorFrame, distance_to_zone_m
from .kinematics import BodyState


@dataclass
class GaitReport:
    """A gait assessment, or an explicit statement that there is not enough to go on."""

    scored: bool
    score: float = 0.0
    sway_rms_cm: float | None = None
    step_count: int = 0
    step_time_cv: float | None = None
    longest_step_ratio: float | None = None
    reaching_for_support: bool = False
    dominant_sway_hz: float | None = None
    path_length_m: float = 0.0
    reasons: list[str] = field(default_factory=list)
    insufficient: str = ""

    @property
    def unsteady_at(self) -> float:
        return 0.0 if not self.scored else self.score

    def to_dict(self) -> dict[str, Any]:
        def num(v: float | None, places: int = 2) -> float | None:
            return None if v is None else round(float(v), places)

        return {
            "scored": self.scored,
            "score": round(self.score, 3),
            "sway_rms_cm": num(self.sway_rms_cm),
            "step_count": self.step_count,
            "step_time_cv": num(self.step_time_cv, 3),
            "longest_step_ratio": num(self.longest_step_ratio),
            "reaching_for_support": self.reaching_for_support,
            "dominant_sway_hz": num(self.dominant_sway_hz),
            "path_length_m": num(self.path_length_m),
            "reasons": list(self.reasons),
            "insufficient": self.insufficient,
        }


def smooth(series: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Gaussian smoothing with an OpenCV kernel, edges replicated."""
    if series.size < 3 or sigma_samples <= 0:
        return series.astype(np.float64)
    ksize = int(max(3, round(sigma_samples * 6) | 1))
    kernel = cv2.getGaussianKernel(ksize, sigma_samples)
    column = series.astype(np.float64).reshape(-1, 1)
    return cv2.filter2D(column, -1, kernel, borderType=cv2.BORDER_REPLICATE).flatten()


def perpendicular_deviation(track: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Signed distance of each point from the fitted walking line, in metres."""
    if track.shape[0] < 4:
        return None
    line = cv2.fitLine(track.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    vx, vy, x0, y0 = (float(v) for v in line)
    normal = np.array([-vy, vx], dtype=np.float64)
    offsets = track - np.array([x0, y0])
    return offsets @ normal, np.array([vx, vy])


def dominant_frequency(deviation: np.ndarray, fps: float) -> float | None:
    """Peak of the lateral-deviation spectrum, via `cv2.dft`. Reported, not scored."""
    n = deviation.size
    if n < 16 or fps <= 0:
        return None
    centred = (deviation - deviation.mean()).astype(np.float32)
    spectrum = cv2.dft(centred.reshape(-1, 1), flags=cv2.DFT_COMPLEX_OUTPUT)
    magnitude = cv2.magnitude(spectrum[:, 0, 0], spectrum[:, 0, 1])
    half = magnitude[1 : n // 2]
    if half.size == 0:
        return None
    peak = int(np.argmax(half)) + 1
    return float(peak * fps / n)


def stance_onsets(gaps: np.ndarray, times: np.ndarray) -> list[float]:
    """Times at which a foot plants, from the peaks of the distance between the feet.

    The first version of this looked for minima in the speed of the ankle
    midpoint. It found almost no footfalls, for a reason that is obvious in
    hindsight: averaging the two ankles cancels most of the swing, and at a ten
    hertz sample rate a 1.7 Hz cadence leaves six samples a step to find a minimum
    in.

    The separation between the two feet is a much better signal. It goes to nearly
    zero as the swinging foot passes the stance foot and peaks at each double
    support, so one peak is one step. It is one dimensional, it does not care
    which foot is which, and it survives the sample rate.
    """
    if gaps.size < 5:
        return []
    smoothed = smooth(gaps, sigma_samples=1.2)
    threshold = float(np.median(smoothed))
    onsets: list[float] = []
    for i in range(1, smoothed.size - 1):
        if smoothed[i] >= smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            if smoothed[i] >= threshold:
                if not onsets or times[i] - onsets[-1] > 0.25:
                    onsets.append(float(times[i]))
    return onsets


class GaitWindow:
    """A sliding window of body states, scored on demand."""

    def __init__(self, thresholds: Thresholds, frame: FloorFrame, walls: tuple[Zone, ...] = ()) -> None:
        self.t = thresholds
        self.frame = frame
        self.walls = walls
        self.states: deque[BodyState] = deque()
        self._reach_since: float | None = None

    def clear(self) -> None:
        """Forget the window. Called the moment the person stops being on their feet."""
        self.states.clear()
        self._reach_since = None

    def push(self, state: BodyState) -> None:
        self.states.append(state)
        cutoff = state.time_s - self.t.gait_window_s
        while self.states and self.states[0].time_s < cutoff:
            self.states.popleft()

    # -- reaching for support ----------------------------------------------
    def _reaching(self, state: BodyState) -> bool:
        if not self.walls:
            return False
        near = False
        for wrist in state.wrists_floor:
            if wrist is None:
                continue
            for wall in self.walls:
                d = distance_to_zone_m(self.frame, wall, wrist)
                if d is not None and d <= self.t.wall_reach_m:
                    near = True
        if not near:
            self._reach_since = None
            return False
        if self._reach_since is None:
            self._reach_since = state.time_s
        return (state.time_s - self._reach_since) >= self.t.wall_reach_hold_s

    # -- the score ----------------------------------------------------------
    def score(self, fps: float) -> GaitReport:
        usable = [s for s in self.states if s.torso_floor is not None and s.floor_xy is not None]
        if len(usable) < 8:
            return GaitReport(
                scored=False,
                insufficient="fewer than 8 frames with a usable floor position in the window",
            )
        times = np.array([s.time_s for s in usable])
        span = float(times[-1] - times[0])
        if span < self.t.gait_window_s * 0.6:
            return GaitReport(
                scored=False,
                insufficient=f"only {span:.1f} s of continuous view, "
                f"{self.t.gait_window_s:.0f} s needed",
            )

        track = np.stack([s.torso_floor for s in usable])
        path_length = float(np.sum(np.linalg.norm(np.diff(track, axis=0), axis=1)))

        deviation_result = perpendicular_deviation(track)
        if deviation_result is None:
            return GaitReport(scored=False, insufficient="the walking line could not be fitted")
        deviation, _ = deviation_result
        sway_cm = float(np.sqrt(np.mean(deviation**2)) * 100.0)

        gaps = np.array([s.ankle_gap_m for s in usable if s.ankle_gap_m is not None])
        gap_times = np.array([s.time_s for s in usable if s.ankle_gap_m is not None])
        onsets = stance_onsets(gaps, gap_times) if gaps.size == times.size else (
            stance_onsets(gaps, gap_times) if gaps.size >= 5 else []
        )
        intervals = np.diff(np.array(onsets)) if len(onsets) >= 2 else np.array([])

        step_cv: float | None = None
        longest_ratio: float | None = None
        if intervals.size >= 2 and float(np.mean(intervals)) > 1e-6:
            step_cv = float(np.std(intervals) / np.mean(intervals))
            longest_ratio = float(np.max(intervals) / np.median(intervals))

        reaching = self._reaching(usable[-1])
        report = GaitReport(
            scored=True,
            sway_rms_cm=sway_cm,
            step_count=len(onsets),
            step_time_cv=step_cv,
            longest_step_ratio=longest_ratio,
            reaching_for_support=reaching,
            dominant_sway_hz=dominant_frequency(deviation, fps),
            path_length_m=path_length,
        )

        if path_length < 0.30 and not reaching:
            report.scored = False
            report.insufficient = (
                f"the person moved {path_length * 100:.0f} cm in the window, "
                "which is standing still rather than walking"
            )
            return report
        if len(onsets) < self.t.gait_min_steps and not reaching:
            report.scored = False
            report.insufficient = (
                f"{len(onsets)} footfalls seen, {self.t.gait_min_steps} needed to score gait"
            )
            return report

        report.score, report.reasons = self._fuse(report)
        return report

    def _fuse(self, r: GaitReport) -> tuple[float, list[str]]:
        """Each measure contributes its own ramp between the warn and high thresholds.

        Deliberately a weighted sum of named ramps rather than a classifier: a
        clinical lead can see why the number moved and can move a threshold
        without retraining anything.
        """
        t = self.t
        parts: list[tuple[float, float, str]] = []

        if r.sway_rms_cm is not None:
            ramp = _ramp(r.sway_rms_cm, t.sway_rms_warn_cm, t.sway_rms_high_cm)
            parts.append((ramp, 0.72, f"swaying {r.sway_rms_cm:.0f} cm about the walking line"))
        if r.reaching_for_support:
            parts.append((1.0, 0.28, "a hand is out to the wall for support"))

        weight = sum(w for _, w, _ in parts) or 1.0
        blended = sum(value * w for value, w, _ in parts) / weight
        # Each threshold pair is defined so that reaching the upper one is enough
        # on its own -- sway of 9 cm about the walking line is not something a
        # steady gait does, whatever the step timing looks like. A pure weighted
        # mean cannot express that: it dilutes one maximal signal with three
        # quiet ones and lands just under the line. So the score is the greater
        # of the blend and the strongest single measure, scaled so that a measure
        # at its upper threshold clears the unsteady line on its own.
        strongest = max((value for value, _, _ in parts), default=0.0)
        score = max(blended, 0.75 * strongest)
        reasons = [text for value, _, text in parts if value > 0.15]
        return float(np.clip(score, 0.0, 1.0)), reasons


def _ramp(value: float, low: float, high: float) -> float:
    """0 below `low`, 1 at or above `high`, linear between. No cliff edges."""
    if high <= low:
        return 1.0 if value >= high else 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))
