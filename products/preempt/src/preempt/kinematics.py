"""Pose over time turned into physical quantities on the floor plane.

This is where the product stops being a pose demo. A single frame of keypoints
says almost nothing useful about whether someone is about to get out of bed. A
few seconds of them, projected onto a metric floor and differentiated properly,
says a great deal.

Quantities produced per frame, all with units:

| name | unit | how |
|---|---|---|
| `floor_xy` | m | ankle midpoint back-projected through the floor homography |
| `torso_floor` | m | the hips' shadow on the floor, given their measured height |
| `hip_height` | m | single-view metrology from the foot contact point |
| `head_height` | m | same, to the head |
| `body_axis_deg` | deg from vertical | ankles to head, the strongest posture cue |
| `trunk_deg` | deg from vertical | trunk vector against the *projective* vertical |
| `lean_offset` | ratio | how far the shoulders sit out over the feet |
| `floor_consistency` | 0..1 | fraction of joints that back-project onto the floor |
| `floor_spread` | m | how long the body is when flattened onto the floor |
| `ankle_gap_m` | m | how far apart the feet are: the gait signal |
| `head_floor_distance_m` | m | the head back-projected onto the floor, from the feet |
| `com_velocity` | m/s | Kalman-smoothed, from `cv2.KalmanFilter` |

Two decisions worth defending:

**The projective vertical.** "Is the trunk upright" is not the image angle. A
camera high in the corner of a room images a vertical line as a line through the
vertical vanishing point, so the trunk is compared against the direction to that
point, not against the image's y axis. Without a configured vanishing point the
code falls back to the image vertical and records `trunk_reference: "image"` so
the limitation travels with the number.

**Segment weights.** The centre of mass uses Winter's segment mass fractions
(head 8.1%, trunk 49.7%, upper arm 2.8% each, forearm+hand 2.2% each, thigh
10.0% each, shank+foot 6.1% each) redistributed over whichever joints are
actually visible, so an occluded arm shifts the estimate rather than silently
biasing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .config import Thresholds
from .geometry import FloorFrame
from .pose import K, PoseFrame

# Winter, Biomechanics and Motor Control of Human Movement, table 4.1.
SEGMENT_MASS: dict[str, float] = {
    "nose": 0.081,
    "left shoulder": 0.124,
    "right shoulder": 0.124,
    "left elbow": 0.016,
    "right elbow": 0.016,
    "left wrist": 0.012,
    "right wrist": 0.012,
    "left hip": 0.124,
    "right hip": 0.124,
    "left knee": 0.050,
    "right knee": 0.050,
    "left ankle": 0.031,
    "right ankle": 0.031,
}


@dataclass
class BodyState:
    """Everything measured about one instant, with explicit gaps where unmeasurable."""

    time_s: float
    index: int
    floor_xy: np.ndarray | None = None
    com_image: np.ndarray | None = None
    torso_floor: np.ndarray | None = None
    com_velocity: np.ndarray | None = None
    hip_height: float | None = None
    head_height: float | None = None
    trunk_deg: float | None = None
    trunk_reference: str = "image"
    knee_deg: float | None = None
    body_axis_deg: float | None = None
    lean_offset: float | None = None
    floor_consistency: float = 0.0
    floor_spread: float | None = None
    visible_joints: int = 0
    ankle_gap_m: float | None = None
    head_floor_distance_m: float | None = None
    foot_contact: np.ndarray | None = None
    wrists_floor: tuple[np.ndarray | None, np.ndarray | None] = (None, None)

    def to_dict(self) -> dict[str, Any]:
        def pair(v: np.ndarray | None) -> list[float] | None:
            return None if v is None else [round(float(v[0]), 3), round(float(v[1]), 3)]

        def num(v: float | None, places: int = 3) -> float | None:
            return None if v is None else round(float(v), places)

        return {
            "time_s": round(self.time_s, 3),
            "floor_xy": pair(self.floor_xy),
            "torso_floor": pair(self.torso_floor),
            "com_speed_mps": num(
                None if self.com_velocity is None else float(np.linalg.norm(self.com_velocity))
            ),
            "hip_height_m": num(self.hip_height),
            "head_height_m": num(self.head_height),
            "trunk_deg": num(self.trunk_deg, 1),
            "trunk_reference": self.trunk_reference,
            "knee_deg": num(self.knee_deg, 1),
            "body_axis_deg": num(self.body_axis_deg, 1),
            "lean_offset": num(self.lean_offset),
            "ankle_gap_m": num(self.ankle_gap_m),
            "head_floor_distance_m": num(self.head_floor_distance_m),
            "floor_consistency": num(self.floor_consistency),
            "floor_spread_m": num(self.floor_spread),
            "visible_joints": self.visible_joints,
        }


def centre_of_mass(pose: PoseFrame, min_score: float) -> tuple[np.ndarray | None, int]:
    """Segment-weighted COM in image pixels, renormalised over visible joints."""
    total, acc = 0.0, np.zeros(2, dtype=np.float64)
    seen = 0
    for name, weight in SEGMENT_MASS.items():
        i = K[name]
        if pose.scores[i] >= min_score:
            acc += pose.xy[i] * weight
            total += weight
            seen += 1
    if total < 1e-6:
        return None, seen
    return acc / total, seen


def vertical_direction(
    frame: FloorFrame, at: np.ndarray, floor_xy: np.ndarray | None = None
) -> tuple[np.ndarray, str]:
    """The image direction of 'up' at a point, and which reference it used."""
    return frame.up_direction(at, floor_xy=floor_xy)


def _toward_feet(pose: PoseFrame, up: np.ndarray, min_score: float) -> np.ndarray | None:
    """The image direction 'the way the feet are', horizontally, at this instant.

    The sign that separates leaning forward to stand up from lying back down.
    Both increase the trunk's angle from the vertical; only one of them moves the
    shoulders toward the feet, and getting that backwards would make the product
    call every time somebody settled back onto a pillow.
    """
    hip = pose.midpoint("left hip", "right hip", min_score)
    ankle = pose.midpoint("left ankle", "right ankle", min_score)
    if hip is None or ankle is None:
        knee = pose.midpoint("left knee", "right knee", min_score)
        if hip is None or knee is None:
            return None
        ankle = knee
    offset = ankle - hip
    perpendicular = offset - float(np.dot(offset, up)) * up
    norm = float(np.linalg.norm(perpendicular))
    if norm < 4.0:  # feet are directly under the hips; there is no forward yet
        return None
    return perpendicular / norm


def trunk_angle_deg(
    pose: PoseFrame,
    frame: FloorFrame,
    min_score: float,
    floor_xy: np.ndarray | None = None,
) -> tuple[float | None, str]:
    """Trunk angle from the projective vertical. Positive is leaning over the feet."""
    hip = pose.midpoint("left hip", "right hip", min_score)
    shoulder = pose.midpoint("left shoulder", "right shoulder", min_score)
    if hip is None or shoulder is None:
        return None, "image"
    trunk = shoulder - hip
    norm = float(np.linalg.norm(trunk))
    if norm < 1e-6:
        return None, "image"
    up, reference = vertical_direction(frame, hip, floor_xy)
    unit = trunk / norm
    cosine = float(np.clip(np.dot(unit, up), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosine)))
    forward = _toward_feet(pose, up, min_score)
    if forward is not None and float(np.dot(unit, forward)) < 0:
        angle = -angle
    return angle, reference


def lean_offset(
    pose: PoseFrame,
    frame: FloorFrame,
    min_score: float,
    contact: np.ndarray | None,
    floor_xy: np.ndarray | None = None,
) -> float | None:
    """How far the shoulders have travelled out over the feet, as a ratio.

    The shoulders' perpendicular distance from the vertical line through the foot
    contact, divided by their distance along it. It is dimensionless, so it needs
    no scale and survives any camera position, and it is the tangent of the angle
    the whole body makes with the vertical at the feet -- which is the quantity a
    physiotherapist means by "leaning out over their feet".

    Signed, positive when the shoulders are out on the same side as the feet.
    Chosen over a centre-of-mass position in metres for a blunt reason: a single
    view cannot place a point of unknown height, so a centre of mass in metres
    would have to be fabricated. This cannot.
    """
    shoulder = pose.midpoint("left shoulder", "right shoulder", min_score)
    if shoulder is None or contact is None:
        return None
    up, _ = vertical_direction(frame, contact, floor_xy)
    offset = shoulder - contact
    along = float(np.dot(offset, up))
    if abs(along) < 1e-6:
        return None
    lateral = offset - along * up
    magnitude = float(np.linalg.norm(lateral)) / abs(along)
    forward = _toward_feet(pose, up, min_score)
    if forward is not None and float(np.dot(lateral, forward)) < 0:
        return -magnitude
    return magnitude


def body_axis_deg(
    pose: PoseFrame,
    frame: FloorFrame,
    min_score: float,
    contact: np.ndarray | None,
    floor_xy: np.ndarray | None = None,
) -> float | None:
    """Angle of the whole body, ankles to head, away from the projective vertical.

    Roughly 0 standing, 20 to 40 sitting with the feet out in front, and near 90
    lying down. It is the single most reliable posture cue available from one
    camera, because it uses the longest segment in the body (so keypoint noise
    matters least) and it is measured against the vertical that this camera
    actually images rather than against the image's y axis.

    It replaced an earlier test that asked how much of the body back-projected
    onto the floor plane. That test was wrong in a way worth recording: a person
    sitting on the edge of a bed and leaning forward, seen from a high corner, has
    most of their joints landing on the floor plane within a body length, and read
    as lying down. The failure was silent and it suppressed every bed-exit call.
    """
    head = pose.midpoint("left ear", "right ear", min_score)
    if head is None:
        head = pose.point("nose", min_score)
    if head is None:
        head = pose.midpoint("left shoulder", "right shoulder", min_score)
    if head is None or contact is None:
        return None
    axis = head - contact
    norm = float(np.linalg.norm(axis))
    if norm < 8.0:
        return None
    up, _ = vertical_direction(frame, contact, floor_xy)
    cosine = float(np.clip(np.dot(axis / norm, up), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def knee_angle_deg(pose: PoseFrame, min_score: float) -> float | None:
    """Hip-knee-ankle angle, averaged over whichever legs are visible. 180 is straight.

    The one posture cue that survives a bad camera angle. Projection distorts an
    angle, but the difference between a knee at roughly 90 degrees (seated, or on
    the edge of the bed) and one at roughly 175 degrees (standing) survives almost
    any viewpoint, so the seated-to-standing decision does not depend on having
    calibrated the camera's vertical.
    """
    angles: list[float] = []
    for side in ("left", "right"):
        hip = pose.point(f"{side} hip", min_score)
        knee = pose.point(f"{side} knee", min_score)
        ankle = pose.point(f"{side} ankle", min_score)
        if hip is None or knee is None or ankle is None:
            continue
        upper, lower = hip - knee, ankle - knee
        n1, n2 = float(np.linalg.norm(upper)), float(np.linalg.norm(lower))
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cosine = float(np.clip(np.dot(upper / n1, lower / n2), -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cosine))))
    return float(np.mean(angles)) if angles else None


def floor_consistency(
    pose: PoseFrame,
    frame: FloorFrame,
    min_score: float,
    anchor: np.ndarray | None = None,
) -> tuple[float, float | None]:
    """How much of the body really is on the floor, measured from the feet.

    Back-project every visible joint as if it were on the floor. A joint that
    genuinely is lands within a body length of where the feet are. A joint on a
    person who is upright lands metres away, or past the horizon, where it cannot
    be back-projected at all.

    The anchor is the feet, not the middle of the back-projected cloud. Anchoring
    on the cloud's own median was the earlier version and it scored a person
    sitting on the edge of a bed and leaning forward at 0.88, because their
    scattered projections are still self-consistent around their own centre. The
    feet are the one point that is known to be on the floor, so they are the one
    point worth measuring from.
    """
    visible = pose.visible(min_score)
    if visible.sum() < 4:
        return 0.0, None
    pts = frame.to_floor(pose.xy[visible])
    good = pts[~np.isnan(pts).any(axis=1)]
    if good.shape[0] < 3:
        return 0.0, None
    if anchor is None or np.isnan(anchor).any():
        anchor = np.median(good, axis=0)
    radius = np.linalg.norm(good - anchor, axis=1)
    near = radius <= 2.00  # a body length from the feet, plus slack
    consistency = float(near.sum() / float(visible.sum()))
    spread = float(radius[near].max()) if near.any() else None
    return consistency, spread


def foot_contact_point(pose: PoseFrame, min_score: float) -> np.ndarray | None:
    """The image point where the body meets the floor, as best as can be told."""
    ankles = [
        pose.point("left ankle", min_score),
        pose.point("right ankle", min_score),
    ]
    present = [a for a in ankles if a is not None]
    if present:
        return np.mean(np.stack(present), axis=0)
    # No ankles: the lowest visible joint is the best available contact guess.
    visible = pose.visible(min_score)
    if not visible.any():
        return None
    idx = int(np.argmax(np.where(visible, pose.xy[:, 1], -np.inf)))
    return pose.xy[idx]


class ComTracker:
    """A constant-velocity Kalman filter on the floor-plane centre of mass.

    `cv2.KalmanFilter` rather than a hand-rolled difference, for two reasons that
    matter here: keypoints are noisy at the few-centimetre level and a raw
    difference of two noisy positions at 30 fps is almost pure noise; and pose is
    dropped on some frames, which a filter handles by predicting while a
    difference handles by producing a spike.

    State is [x, y, vx, vy] in metres and metres per second.
    """

    def __init__(self, process_noise: float = 1e-2, measurement_noise: float = 4e-3) -> None:
        self.kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_64F)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float64) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float64) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float64)
        self._started = False
        self._last_t: float | None = None

    def _transition(self, dt: float) -> np.ndarray:
        return np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )

    def update(self, position: np.ndarray | None, time_s: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Returns (position, velocity) in metres and m/s, or None before the first fix."""
        dt = 1.0 / 30.0 if self._last_t is None else max(1e-3, time_s - self._last_t)
        self._last_t = time_s
        if position is None or np.isnan(position).any():
            if not self._started:
                return None
            self.kf.transitionMatrix = self._transition(dt)
            predicted = self.kf.predict()
            return predicted[:2, 0].copy(), predicted[2:, 0].copy()
        if not self._started:
            self.kf.statePost = np.array(
                [[position[0]], [position[1]], [0.0], [0.0]], dtype=np.float64
            )
            self._started = True
            return position.copy(), np.zeros(2)
        self.kf.transitionMatrix = self._transition(dt)
        self.kf.predict()
        corrected = self.kf.correct(
            np.array([[position[0]], [position[1]]], dtype=np.float64)
        )
        return corrected[:2, 0].copy(), corrected[2:, 0].copy()


@dataclass
class Kinematics:
    """Turns a stream of PoseFrames into a stream of BodyStates."""

    frame: FloorFrame
    thresholds: Thresholds
    tracker: ComTracker = field(default_factory=ComTracker)

    def state(self, pose: PoseFrame) -> BodyState:
        t = self.thresholds
        min_score = t.keypoint_score_min
        out = BodyState(time_s=pose.time_s, index=pose.index)
        out.visible_joints = int(pose.visible(min_score).sum())
        if out.visible_joints < t.pose_keypoints_min:
            return out

        com_image, _ = centre_of_mass(pose, min_score)
        out.com_image = com_image
        contact = foot_contact_point(pose, min_score)
        out.foot_contact = contact
        if contact is not None:
            mapped = self.frame.to_floor(contact.reshape(1, 2))[0]
            out.floor_xy = None if np.isnan(mapped).any() else mapped

        hip = pose.midpoint("left hip", "right hip", min_score)
        head = pose.midpoint("left ear", "right ear", min_score)
        if head is None:
            head = pose.point("nose", min_score)

        if contact is not None:
            if hip is not None:
                out.hip_height = self.frame.height_above_floor(tuple(contact), tuple(hip)).metres
            if head is not None:
                out.head_height = self.frame.height_above_floor(tuple(contact), tuple(head)).metres

        # The torso's position on the floor: its shadow, given the height just
        # measured. Exact once the height is, and the height is only approximate
        # because it assumed the hips were above the feet -- true to a few
        # centimetres when standing, which is the only time this is used.
        if hip is not None and out.hip_height is not None:
            shadow = self.frame.floor_shadow(hip, out.hip_height)
            if shadow is not None and not np.isnan(shadow).any():
                out.torso_floor = shadow
        if out.torso_floor is None:
            out.torso_floor = out.floor_xy

        tracked = self.tracker.update(out.torso_floor, pose.time_s)
        if tracked is not None:
            _, velocity = tracked
            out.com_velocity = velocity

        out.trunk_deg, out.trunk_reference = trunk_angle_deg(
            pose, self.frame, min_score, out.floor_xy
        )
        out.knee_deg = knee_angle_deg(pose, min_score)
        out.body_axis_deg = body_axis_deg(
            pose, self.frame, min_score, contact, out.floor_xy
        )
        out.lean_offset = lean_offset(pose, self.frame, min_score, contact, out.floor_xy)
        out.floor_consistency, out.floor_spread = floor_consistency(
            pose, self.frame, min_score, out.floor_xy
        )

        # Back-project the head as if it were lying on the floor and see how far
        # from the feet it lands. For someone on the floor that is a body length.
        # For someone standing it is three to five metres, or past the horizon and
        # therefore nowhere at all. It is the measure that actually separated the
        # two on real footage, where heights did not, because a height computed
        # from a base point the head is *not* above is meaningless.
        if contact is not None and head is not None:
            mapped = self.frame.to_floor(np.stack([contact, head]))
            if not np.isnan(mapped).any():
                out.head_floor_distance_m = float(np.linalg.norm(mapped[0] - mapped[1]))

        left_ankle = pose.point("left ankle", min_score)
        right_ankle = pose.point("right ankle", min_score)
        if left_ankle is not None and right_ankle is not None:
            feet = self.frame.to_floor(np.stack([left_ankle, right_ankle]))
            if not np.isnan(feet).any():
                out.ankle_gap_m = float(np.linalg.norm(feet[0] - feet[1]))

        for side, index in (("left", 0), ("right", 1)):
            wrist = pose.point(f"{side} wrist", min_score)
            if wrist is None:
                continue
            mapped = self.frame.to_floor(wrist.reshape(1, 2))[0]
            wrists = list(out.wrists_floor)
            wrists[index] = None if np.isnan(mapped).any() else mapped
            out.wrists_floor = (wrists[0], wrists[1])
        return out
