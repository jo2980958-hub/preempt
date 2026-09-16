"""Bed and chair exit: the movement that comes *before* standing.

The clinical fact this module is built on is that standing up from a seat is not
one event, it is four phases, and the first two happen while the person is still
supported. Phase I is flexion momentum: the trunk leans forward and the centre of
mass travels toward the feet. Phase II is momentum transfer, the moment the
weight leaves the seat. Phase III is extension. Phase IV is stabilisation.

Every commercial bed sensor fires at the end of Phase II or later, because a
pressure pad cannot know anything until the weight has gone. A camera watching
the centre of mass can see Phase I. That gap is the product.

So the call is raised when the centre of mass starts travelling toward the feet
while the knees are still bent and the hips are still over the bed, and the lead
time reported in `docs/evaluation.md` is measured from that instant to the
instant the person is upright on their feet.

Posture, robustly
-----------------
Posture comes from the knee angle and the trunk angle rather than from a height,
because a height needs a calibrated vertical vanishing point and a knee angle
does not. Heights are used when they are available and are reported as extra
evidence, never as the thing the decision hangs on.

States, in the order a night shift would recognise them:

    settled -> stirring -> sitting up -> on the edge -> preparing to rise
            -> rising -> standing -> walking

with `on the floor` reachable from anywhere and `away` when the person has left
the room.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Thresholds, Zone
from .geometry import FloorFrame, distance_to_zone_m, inside
from .kinematics import BodyState

SETTLED = "settled"
STIRRING = "stirring"
SITTING_UP = "sitting up"
ON_EDGE = "on the edge"
PREPARING = "preparing to rise"
RISING = "rising"
STANDING = "standing"
WALKING = "walking"
ON_FLOOR = "on the floor"
AWAY = "away"

ORDER = (
    SETTLED,
    STIRRING,
    SITTING_UP,
    ON_EDGE,
    PREPARING,
    RISING,
    STANDING,
    WALKING,
    ON_FLOOR,
)

KNEE_SEATED_MAX = 130.0
KNEE_STANDING_MIN = 150.0
BODY_AXIS_LYING_MIN = 62.0
BODY_AXIS_UPRIGHT_MAX = 16.0
TRUNK_LYING_MIN = 60.0
STIR_SPEED_MPS = 0.05
DESCENT_WINDOW_S = 0.5
"""Hip-height slope window for telling a sit-down from a rise: long enough that
seated keypoint noise averages out, short enough to follow a one-second sit."""
KNEE_CLOSING_DEG_S = 90.0
"""Without metric heights, a sit-down is the knees closing at least this fast. On
the CDC footage they close at 150 to 220 degrees a second."""
SAT_DOWN_REPORT_S = 1.5
"""How long after the descent the reading keeps saying the person just sat down."""


@dataclass
class ExitReading:
    """One frame's posture verdict and the measurements that produced it."""

    time_s: float
    state: str
    support: str = ""
    lean_offset: float | None = None
    flexion_deg_s: float | None = None
    knee_deg: float | None = None
    trunk_deg: float | None = None
    hip_height_m: float | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def num(v: float | None, places: int = 3) -> float | None:
            return None if v is None else round(float(v), places)

        return {
            "time_s": round(self.time_s, 3),
            "state": self.state,
            "support": self.support,
            "lean_offset": num(self.lean_offset),
            "flexion_deg_s": num(self.flexion_deg_s, 1),
            "knee_deg": num(self.knee_deg, 1),
            "trunk_deg": num(self.trunk_deg, 1),
            "hip_height_m": num(self.hip_height_m),
            "reasons": list(self.reasons),
        }


class ExitDetector:
    """The state machine. Feed it BodyStates in time order; ask it for a reading.

    It keeps a short history because the two quantities that matter -- how fast
    the centre of mass is closing on the feet, and how long that has held -- are
    both differences over time, and reading them off a single frame is how a
    system ends up calling every time someone rolls over.
    """

    def __init__(
        self,
        thresholds: Thresholds,
        frame: FloorFrame,
        *,
        bed_zones: tuple[Zone, ...] = (),
        chair_zones: tuple[Zone, ...] = (),
        history_s: float = 2.0,
    ) -> None:
        self.t = thresholds
        self.frame = frame
        self.bed_zones = bed_zones
        self.chair_zones = chair_zones
        self.history_s = history_s
        self.history: deque[BodyState] = deque()
        self.state = SETTLED
        self.entered_at: float = 0.0
        self._prepare_since: float | None = None
        self._lapsed_since: float | None = None
        self._transfer_delta: float = 0.0
        self._floor_since: float | None = None
        self._floor_lapsed: float | None = None
        self._descent_at: float | None = None
        self._descent_speed: float = 0.0
        self.timeline: list[ExitReading] = []

    # -- geometry helpers ---------------------------------------------------
    def _support(self, state: BodyState) -> str:
        """Which piece of furniture the person is on, by their hip position."""
        point = state.com_image
        if point is None:
            return ""
        for zone in self.bed_zones:
            if inside(zone, tuple(point)):
                return zone.name
        for zone in self.chair_zones:
            if inside(zone, tuple(point)):
                return zone.name
        return ""

    def _transfer_rate(self, window_s: float = 0.9) -> float | None:
        """How fast the upper body is arriving over the feet, per second.

        `lean_offset` is the shoulders' horizontal displacement from the foot
        contact, as a fraction of their height above it. Sitting on the edge of a
        bed it is strongly negative, because the feet are out in front and the
        shoulders are back over the mattress. Standing up drives it to zero: that
        *is* the definition of the momentum-transfer phase, the centre of mass
        arriving over the base of support, and it happens before the weight
        leaves the bed.

        So the rate of that quantity is Phase I measured directly. Fitted over a
        window rather than differenced frame to frame, because keypoint noise
        would otherwise dominate; a straight-line fit over about a second is
        stable and still fast enough to leave seconds of warning.

        Leaning back onto a pillow drives the same quantity the other way, so the
        sign does the work that a separate "is this a lie-down" check would
        otherwise have to.
        """
        # Lean gathered while the person was coming down onto the seat is evidence
        # of sitting down, not of getting up; see `_note_descent`.
        since = self._descent_at if self._descent_at is not None else -np.inf
        samples = [
            (s.time_s, s.lean_offset)
            for s in self.history
            if s.lean_offset is not None and s.time_s > since
        ]
        if len(samples) < 4:
            return None
        latest = samples[-1][0]
        window = [(t, a) for t, a in samples if latest - t <= window_s]
        if len(window) < 4:
            return None
        times = np.array([t for t, _ in window])
        values = np.array([a for _, a in window])
        if float(times[-1] - times[0]) < 1e-3:
            return None
        self._transfer_delta = float(np.median(values[-3:]) - np.median(values[:3]))
        return float(np.polyfit(times - times[0], values, 1)[0])

    def _lying_on_floor(self, state: BodyState) -> bool:
        """Geometric, not guessed: the whole body back-projects onto the floor.

        This is the third version of this test and the first one that survived
        real footage. It is worth recording what the other two got wrong, because
        both looked right and both were silently useless.

        The first asked whether the head was low, in metres. That needs a height
        measured from a base point the head is above, and a person lying down has
        their head about a body length *sideways* from their feet, so the height
        came out between three and five metres. On twelve real falls it fired
        zero times.

        The second asked how long the body was when flattened onto the floor.
        That saturates: standing and lying both gave about 2.2 m.

        What separates them is where the head lands when it is back-projected as
        if it were on the floor. Someone on the floor: 0.9 to 1.7 m from their
        feet, which is a person. Someone standing: 2.5 to 5.3 m, or past the
        horizon and therefore nowhere at all. Measured on UR Fall, cleanly
        separated, with the fraction of joints that back-project sensibly as a
        second condition.
        """
        if not self._feet_on_floor(state) or self._support(state):
            return False  # lying on a bed is lying on a bed
        if state.floor_consistency < self.t.floor_consistency_min:
            return False
        distance = state.head_floor_distance_m
        if distance is None:
            return False
        return self.t.floor_body_min_m <= distance <= self.t.floor_body_max_m

    # -- posture ------------------------------------------------------------
    def _feet_on_floor(self, state: BodyState) -> bool:
        """Are the feet actually on the floor, or on a mattress?

        Every height this product measures is a height *above the floor plane*,
        computed from the point where the body meets the floor. If the feet are on
        a bed, that point is not on the floor and every height derived from it is
        inflated -- measurably so: a patient lying in bed reads as a hip height of
        0.92 m, which is standing. So heights are only trusted when the foot
        contact is on the floor, and the furniture a nurse drew is what decides.
        """
        if state.foot_contact is None or state.floor_xy is None:
            return False
        if np.isnan(state.floor_xy).any():
            return False
        point = tuple(state.foot_contact)
        return not any(inside(zone, point) for zone in self.bed_zones + self.chair_zones)

    def _posture(self, state: BodyState) -> str:
        """Supported, lying, seated or upright, in that order of reliability.

        Two earlier versions of this were wrong in ways worth recording, because
        both failed silently and both suppressed real calls.

        The first used the **image knee angle**. A thigh pointing along the viewing
        ray foreshortens to nothing, and a patient in a chair read as standing.

        The second used the **body axis against the projective vertical**, ankles
        to head. A body lying on the floor and pointing away from the camera also
        images as close to the vertical: measured at 43 degrees where 90 was
        expected. Foreshortening again.

        What survives is metric height above the floor plane, which is a real
        physical quantity and not an appearance: on the floor the head is at 0.1 to
        0.45 m, on the edge of a bed at about 1.1 m, standing at about 1.6 m.
        Measured, in this room, in metres, and it only needs the feet to be on the
        floor -- which is the first thing checked.
        """
        t = self.t
        if not self._feet_on_floor(state):
            return "supported"
        if state.head_height is not None and state.hip_height is not None:
            if state.head_height < t.floor_head_height_m:
                return "lying"
            if state.hip_height >= t.stand_hip_height_m:
                return "upright"
            return "seated"
        knee, trunk = state.knee_deg, state.trunk_deg
        if trunk is not None and abs(trunk) >= TRUNK_LYING_MIN:
            return "lying"
        if knee is None:
            return "unknown"
        if knee >= KNEE_STANDING_MIN:
            return "upright"
        if knee <= KNEE_SEATED_MAX:
            return "seated"
        return "unknown"

    # -- the step -----------------------------------------------------------
    def update(self, state: BodyState) -> ExitReading:
        self.history.append(state)
        while self.history and state.time_s - self.history[0].time_s > self.history_s:
            self.history.popleft()

        support = self._support(state)
        posture = self._posture(state)
        if posture != "seated":
            # The evidence for standing up is only meaningful while the person is
            # sitting. Leaving it accumulated across a walk meant one noisy frame
            # that read as seated could satisfy a hold that had started minutes
            # earlier, which is how the last false alarm in the evaluation got in.
            self._prepare_since = None
            self._lapsed_since = None
        descending = self._note_descent(state)
        if descending:
            # The same lean, travelling the other way. Whatever evidence for a rise
            # had built up belongs to the sit-down and is thrown away with it.
            self._prepare_since = None
            self._lapsed_since = None
        distance = state.lean_offset
        closing = self._transfer_rate()
        speed = (
            float(np.linalg.norm(state.com_velocity)) if state.com_velocity is not None else None
        )
        reasons: list[str] = []

        if self._lying_on_floor(state):
            self._floor_since = self._floor_since or state.time_s
            self._floor_lapsed = None
            if state.time_s - self._floor_since >= self.t.floor_hold_s:
                reasons.append(
                    f"the whole body maps onto the floor over {state.floor_spread:.1f} m"
                )
                if state.head_height is not None:
                    reasons.append(f"the head is {state.head_height * 100:.0f} cm off the floor")
                return self._settle(ON_FLOOR, state, support, distance, closing, reasons)
        elif self._floor_since is not None:
            if self._floor_lapsed is None:
                self._floor_lapsed = state.time_s
            if state.time_s - self._floor_lapsed > self.t.floor_grace_s:
                self._floor_since = None
                self._floor_lapsed = None

        if state.visible_joints < self.t.pose_keypoints_min:
            return self._settle(self.state, state, support, distance, closing, reasons)

        if posture == "upright":
            walking = speed is not None and speed >= 0.25
            reasons.append(
                f"knees are straight at {state.knee_deg:.0f} degrees"
                if state.knee_deg is not None
                else "the hips are at standing height"
            )
            if walking:
                reasons.append(f"moving at {speed:.1f} m/s")
            return self._settle(
                WALKING if walking else STANDING, state, support, distance, closing, reasons
            )

        # Still supported. This is where the product earns its keep.
        if posture in ("lying", "supported"):
            if support or posture == "supported":
                stirring = speed is not None and speed >= STIR_SPEED_MPS
                if stirring:
                    reasons.append(f"moving in {support}")
                return self._settle(
                    STIRRING if stirring else SETTLED, state, support, distance, closing, reasons
                )
            return self._settle(self.state, state, support, distance, closing, reasons)

        rising = self._hip_rise_speed()
        if rising is not None and rising >= self.t.rise_hip_speed_mps:
            reasons.append(f"the hips are rising at {rising:.2f} m/s")
            return self._settle(RISING, state, support, distance, closing, reasons)

        prepared = self._preparing(state, closing)
        if prepared:
            reasons.append(
                f"the upper body is moving over the feet at {closing:.2f} body widths a second"
                if closing is not None
                else "the upper body is moving over the feet"
            )
            if distance is not None:
                reasons.append(
                    "the shoulders have arrived over the feet"
                    if distance >= -0.05
                    else (
                        f"the shoulders are {abs(distance):.2f} back from over the feet and closing"
                    )
                )
            if state.knee_deg is not None:
                reasons.append(
                    f"the knees are still bent at {state.knee_deg:.0f} degrees, "
                    "so the weight is still on the bed"
                )
            return self._settle(PREPARING, state, support, distance, closing, reasons)

        if self._descent_at is not None and state.time_s - self._descent_at <= SAT_DOWN_REPORT_S:
            reasons.append(
                f"sitting down: the hips came down from standing at {self._descent_speed:.2f} m/s"
                if self._descent_speed > 0
                else "sitting down: the knees closed from standing"
            )
        on_edge = self._on_edge(state)
        if on_edge:
            reasons.append("sitting on the edge with the feet on the floor")
            return self._settle(ON_EDGE, state, support, distance, closing, reasons)

        if support:
            reasons.append(f"sitting up in {support}")
            return self._settle(SITTING_UP, state, support, distance, closing, reasons)
        return self._settle(SITTING_UP, state, support, distance, closing, reasons)

    # -- transition helpers -------------------------------------------------
    def _note_descent(self, state: BodyState) -> bool:
        """Is this person on their way *down* onto the seat? Remember when, if so.

        Sitting down and getting up share a shape. To lower yourself onto a chair
        you lean forward over your feet with your knees bent, and from one camera
        that is the lean `_preparing` calls on. Real footage (the CDC chair-stand
        clips, `docs/evaluation.md`) produced a call on every sit-down because of
        it. Thresholds cannot separate the two, because the postures really are
        the same; the direction of travel can.

        A sit-down is the hips going down fast -- as fast as a rise goes up, which
        is `rise_hip_speed_mps` -- or, without metric heights, the knees closing,
        within the detector's history of the person having been upright. A rise
        starts with the hips seated and still, or going up. The fit is over half a
        second rather than a frame difference so that keypoint noise on a seated
        hip cannot pass for a descent.

        While it holds, `_descent_at` moves forward, and `_transfer_rate` only
        uses lean samples from after it, so the evidence for standing up has to
        be gathered from scratch once the person has actually sat.
        """
        upright_recently = False
        for reading in reversed(self.timeline):
            if state.time_s - reading.time_s > self.history_s:
                break
            if reading.state in (STANDING, WALKING):
                upright_recently = True
                break
        if not upright_recently:
            return False
        window = [s for s in self.history if state.time_s - s.time_s <= DESCENT_WINDOW_S]
        hips = [(s.time_s, s.hip_height) for s in window if s.hip_height is not None]
        if len(hips) >= 3 and hips[-1][0] - hips[0][0] >= DESCENT_WINDOW_S / 2:
            slope = float(np.polyfit([t for t, _ in hips], [h for _, h in hips], 1)[0])
            descending = slope <= -self.t.rise_hip_speed_mps
            speed = -slope
        else:
            knees = [(s.time_s, s.knee_deg) for s in window if s.knee_deg is not None]
            if len(knees) < 3 or knees[-1][0] - knees[0][0] < DESCENT_WINDOW_S / 2:
                return False
            slope = float(np.polyfit([t for t, _ in knees], [k for _, k in knees], 1)[0])
            descending = slope <= -KNEE_CLOSING_DEG_S
            speed = 0.0
        if descending:
            self._descent_at = state.time_s
            self._descent_speed = speed
        return descending

    def _hip_rise_speed(self) -> float | None:
        """Upward hip speed in m/s, when a metric height is available."""
        heights = [(s.time_s, s.hip_height) for s in self.history if s.hip_height is not None]
        if len(heights) < 3:
            return None
        span = heights[-1][0] - heights[0][0]
        if span < 0.15:
            return None
        return float((heights[-1][1] - heights[0][1]) / span)

    def _preparing(self, state: BodyState, flexion: float | None) -> bool:
        """Folding forward, far enough out over the feet, knees still bent.

        All three, because each alone has a common innocent explanation: a person
        reaching for a cup flexes, a person sitting slouched is already out over
        their feet, and a person standing still has straight knees. Together they
        are the movement that precedes standing and very little else.
        """
        # The feet have to be on the floor beside the furniture. Sitting up in
        # bed drives the same lean measure the same way, and without this gate the
        # product calls every time somebody props themselves up on an elbow.
        leaning_out = state.lean_offset is not None and state.lean_offset >= self.t.lean_over_feet
        # A rate alone is not enough. Somebody shifting in a chair produces the
        # same rate for a moment and then goes back; somebody standing up keeps
        # going. So the total travel across the window has to be real as well.
        folding = (
            flexion is not None
            and flexion >= self.t.transfer_rate_per_s
            and self._transfer_delta >= self.t.transfer_delta
        )
        # The feet have to be on the floor beside the furniture. Sitting up in bed
        # drives the same lean measure the same way, and without this gate the
        # product calls every time somebody props themselves up on an elbow.
        # You cannot begin to stand up if you were walking a second ago. Without
        # this, someone crossing the room past the foot of the bed satisfies the
        # edge test and the lean test at the same instant and gets called. It was
        # the only false alarm left in the synthetic evaluation.
        was_seated = self._seated_recently(state.time_s)
        holds = was_seated and self._on_edge(state) and folding and leaning_out
        if holds:
            self._lapsed_since = None
            if self._prepare_since is None:
                self._prepare_since = state.time_s
        else:
            # One bad frame is keypoint noise, not a change of mind. Without this
            # grace the evidence restarts from zero several times during a real
            # transfer and the call arrives a second late, which is a second of
            # the lead time this whole product exists to buy.
            if self._prepare_since is None:
                return False
            if self._lapsed_since is None:
                self._lapsed_since = state.time_s
            if (state.time_s - self._lapsed_since) > self.t.prepare_grace_s:
                self._prepare_since = None
                self._lapsed_since = None
                return False
        return (state.time_s - self._prepare_since) >= self.t.prepare_hold_s

    def _seated_recently(self, time_s: float, window_s: float = 2.5, need_s: float = 0.8) -> bool:
        """Was the person actually sitting somewhere in the recent past?"""
        seated = [
            r.time_s
            for r in self.timeline
            if time_s - r.time_s <= window_s and r.state in (SITTING_UP, ON_EDGE, SETTLED, STIRRING)
        ]
        if len(seated) < 2:
            return False
        return (max(seated) - min(seated)) >= need_s

    def _on_edge(self, state: BodyState) -> bool:
        """Hips still on the furniture, feet already on the floor next to it."""
        if state.floor_xy is None:
            return False
        for zone in self.bed_zones + self.chair_zones:
            distance = distance_to_zone_m(self.frame, zone, state.floor_xy)
            if distance is None:
                continue
            if -self.t.edge_margin_m <= distance <= self.t.edge_margin_m * 3:
                return True
        return False

    def _settle(
        self,
        new_state: str,
        state: BodyState,
        support: str,
        distance: float | None,
        closing: float | None,
        reasons: list[str],
    ) -> ExitReading:
        if new_state != self.state:
            self.state = new_state
            self.entered_at = state.time_s
        reading = ExitReading(
            time_s=state.time_s,
            state=new_state,
            support=support,
            lean_offset=distance,
            flexion_deg_s=closing,
            knee_deg=state.knee_deg,
            trunk_deg=state.trunk_deg,
            hip_height_m=state.hip_height,
            reasons=reasons,
        )
        self.timeline.append(reading)
        return reading

    # -- results ------------------------------------------------------------
    def first_time_in(self, state: str) -> float | None:
        return next((r.time_s for r in self.timeline if r.state == state), None)

    def lead_time_s(self) -> float | None:
        """Seconds between the first call and the person actually being upright.

        Measured to the *first upright frame after the call*, not to the first in
        the sequence: someone who was already standing when the recording began
        is not evidence of anything. None when either end did not happen, which is
        the honest answer for a sequence where nobody stood up, or where the call
        was never raised, and it is counted as a miss rather than quietly dropped.
        """
        called = self.first_time_in(PREPARING)
        if called is None:
            return None
        upright = next(
            (
                r.time_s
                for r in self.timeline
                if r.state in (STANDING, WALKING) and r.time_s > called
            ),
            None,
        )
        if upright is None:
            return None
        return float(upright - called)
