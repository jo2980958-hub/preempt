"""Set a room up by walking through it, instead of measuring it with a tape.

The setup Preempt needs is four floor points with known metric spacing, plus a
vertical reference. Asking a ward to measure a rectangle on the floor of a side
room with a tape measure and then click four pixels is a real barrier, and it is
also the step most likely to be done badly and produce confidently wrong metres.

There is a better way, and it is a known result: a person of roughly known height
walking around a room calibrates the camera. Each upright stance gives a vertical
segment in the image. All those segments meet at the vertical vanishing point.
Any two stances at different depths give a point on the floor's horizon, because
the line through two heads and the line through the two matching feet meet on it.
With those two and an assumed stature, a camera with square pixels and its
principal point near the centre is fully determined.

    vz  = vertical vanishing point      (from the body axes)
    l   = the floor's vanishing line    (from head-line and foot-line crossings)
    K^-1 vz  is parallel to  K^T l      (the vertical is the floor's normal)

which gives the focal length in closed form, and from there the rotation, the
camera height and a metric floor frame.

What this is good for, and what it is not
-----------------------------------------
It is good for a room where nobody will measure anything: a research dataset with
no published extrinsics, or a ward that wants the camera working this afternoon.
The heights it produces are as good as the stature you assumed, so a 1.75 m
assumption for a 1.60 m patient reads every height about 9 per cent high, and the
product says so rather than hiding it.

It is not as good as four measured floor points. When a ward can measure, they
should, and `RoomConfig` takes those directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import FloorPlane
from .pose import PoseFrame

DEFAULT_STATURE_M = 1.75
"""Floor to crown of the person used for scale. Every height the room reports
scales with this number, so it is the one thing worth measuring properly."""

ANKLE_ABOVE_FLOOR_M = 0.07
"""The ankle keypoint is not the floor. It sits at about the malleolus."""

EAR_BELOW_CROWN_M = 0.08
"""Nor is the ear the top of the head. Between them these two shorten the
measurable segment by about 15 cm, which is 9 per cent of an adult and therefore
9 per cent of every height in the room if it is ignored."""


@dataclass
class Calibration:
    """A camera recovered from people, with the evidence and the caveats."""

    ok: bool
    plane: FloorPlane | None = None
    focal_px: float | None = None
    camera_height_m: float | None = None
    stances_used: int = 0
    vz: tuple[float, float] | None = None
    horizon: tuple[float, float, float] | None = None
    residual_px: float | None = None
    reason: str = ""
    assumptions: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.assumptions is None:
            self.assumptions = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "focal_px": None if self.focal_px is None else round(self.focal_px, 1),
            "camera_height_m": (
                None if self.camera_height_m is None else round(self.camera_height_m, 2)
            ),
            "stances_used": self.stances_used,
            "vz": None if self.vz is None else [round(v, 1) for v in self.vz],
            "residual_px": None if self.residual_px is None else round(self.residual_px, 2),
            "reason": self.reason,
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class Stance:
    """One upright person: where their feet are and where the top of them is."""

    foot: np.ndarray
    head: np.ndarray

    @property
    def image_height(self) -> float:
        return float(np.linalg.norm(self.head - self.foot))


def upright_stances(
    poses: list[PoseFrame],
    *,
    min_score: float = 0.5,
    min_height_px: float = 90.0,
) -> list[Stance]:
    """Pick the frames where somebody is clearly standing up, and measure them.

    'Clearly standing' is decided without any camera knowledge, because there is
    none yet: the body must be much taller than it is wide, both ankles must be
    seen and be close together, and the head must be above the hips which must be
    above the ankles. It is a crude test and it is meant to be, because a wrong
    stance here poisons the whole calibration.
    """
    stances: list[Stance] = []
    for pose in poses:
        ankles = [pose.point(f"{s} ankle", min_score) for s in ("left", "right")]
        if any(a is None for a in ankles):
            continue
        hips = pose.midpoint("left hip", "right hip", min_score)
        head = pose.midpoint("left ear", "right ear", min_score)
        if head is None:
            head = pose.point("nose", min_score)
        shoulders = pose.midpoint("left shoulder", "right shoulder", min_score)
        if hips is None or head is None or shoulders is None:
            continue
        foot = (ankles[0] + ankles[1]) / 2.0
        height = float(np.linalg.norm(head - foot))
        if height < min_height_px:
            continue
        width = float(np.linalg.norm(ankles[0] - ankles[1]))
        if width > 0.35 * height:
            continue  # feet wide apart: mid-stride or not standing
        if not (head[1] < shoulders[1] < hips[1] < foot[1]):
            continue  # not stacked vertically in the image at all
        if abs(float(head[0] - foot[0])) > 0.28 * height:
            continue  # leaning
        stances.append(Stance(foot=foot, head=head))
    return stances


def _line(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.cross(np.array([*p, 1.0]), np.array([*q, 1.0]))


def _intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    point = np.cross(a, b)
    if abs(point[2]) < 1e-9:
        return None
    return point[:2] / point[2]


def vertical_vanishing_point(stances: list[Stance]) -> np.ndarray | None:
    """Least-squares meeting point of every body axis.

    Each stance contributes the line through its feet and its head. Those lines
    are the images of parallel vertical world lines, so they all pass through one
    point; solving `l_i . v = 0` for all of them in least squares is the standard
    way to find it and it degrades gracefully when the lines are nearly parallel.
    """
    if len(stances) < 2:
        return None
    lines = np.stack([_line(s.foot, s.head) for s in stances])
    norms = np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    lines = lines / np.maximum(norms, 1e-9)
    _, _, vt = np.linalg.svd(lines)
    v = vt[-1]
    if abs(v[2]) < 1e-9:
        return None
    return v[:2] / v[2]


def horizon_from_stances(stances: list[Stance]) -> np.ndarray | None:
    """The floor's vanishing line, from pairs of people of the same height.

    Two people the same height standing at different depths: the line through
    their heads and the line through their feet are images of two parallel world
    lines, so they cross on the horizon. Enough pairs and the horizon is the line
    those crossings lie on.
    """
    if len(stances) < 3:
        return None
    points: list[np.ndarray] = []
    order = sorted(stances, key=lambda s: s.image_height)
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = order[i], order[j]
            if b.image_height < a.image_height * 1.12:
                continue  # too similar in depth; the crossing is ill-conditioned
            crossing = _intersect(_line(a.head, b.head), _line(a.foot, b.foot))
            if crossing is not None and np.all(np.isfinite(crossing)):
                points.append(crossing)
    if len(points) < 2:
        return None
    array = np.asarray(points)
    keep = np.abs(array[:, 1] - np.median(array[:, 1])) < 6.0 * (
        np.median(np.abs(array[:, 1] - np.median(array[:, 1]))) + 1.0
    )
    array = array[keep] if keep.sum() >= 2 else array
    homogeneous = np.concatenate([array, np.ones((array.shape[0], 1))], axis=1)
    _, _, vt = np.linalg.svd(homogeneous)
    line = vt[-1]
    norm = math.hypot(float(line[0]), float(line[1]))
    return line / norm if norm > 1e-9 else None


def _height_spread(stances: list[Stance], vz: np.ndarray, horizon: np.ndarray) -> float:
    """How much a set of same-height people disagree about their height, given a horizon.

    Zero is perfect agreement. This is the objective the level-horizon search
    minimises, and it is well behaved because the quantity inside is the standard
    affine height ratio: for people who really are the same height it is constant
    when the horizon is right and drifts with depth when it is wrong.
    """
    vz_h = np.array([*vz, 1.0])
    ratios = []
    for stance in stances:
        b = np.array([*stance.foot, 1.0])
        t = np.array([*stance.head, 1.0])
        denominator = abs(float(horizon @ b)) * float(np.linalg.norm(np.cross(t, vz_h)))
        if denominator < 1e-9:
            return float("inf")
        ratios.append(float(np.linalg.norm(np.cross(b, t)) / denominator))
    array = np.asarray(ratios)
    mean = float(array.mean())
    return float(array.std() / mean) if mean > 1e-12 else float("inf")


def level_horizon(stances: list[Stance], vz: np.ndarray, height_px: int) -> np.ndarray | None:
    """The horizon of a camera with no roll, found by a one-dimensional search.

    The general two-point construction -- cross the head line with the foot line
    for every pair and fit a line to the crossings -- is correct and, on real
    footage, useless. When two people are at similar depths their crossing runs
    off to a hundred thousand pixels, a few pixels of keypoint noise swings it,
    and the fitted line came out tilted 24 degrees on the UR Fall camera, which
    then produced no real solution for the focal length at all.

    A camera that is not deliberately rolled has a level horizon, which is one
    unknown instead of three. Searching for it directly, by asking which height
    makes a set of same-height people agree about their height, is stable: the
    objective is smooth, it has one minimum, and a coarse-to-fine scan over the
    plausible range finds it in a few hundred evaluations.
    """
    if len(stances) < 3:
        return None
    best: tuple[float, float] | None = None
    low, high, step = -6.0 * height_px, 6.0 * height_px, height_px / 8.0
    for _ in range(4):
        y = low
        while y <= high:
            spread = _height_spread(stances, vz, np.array([0.0, 1.0, -y]))
            if best is None or spread < best[0]:
                best = (spread, y)
            y += step
        if best is None:
            return None
        low, high = best[1] - step * 2, best[1] + step * 2
        step /= 8.0
    if best is None or not math.isfinite(best[0]) or best[0] > 0.25:
        return None
    return np.array([0.0, 1.0, -best[1]])


def _focal_from(vz: np.ndarray, horizon: np.ndarray, centre: tuple[float, float]) -> float | None:
    """Solve `K^-1 vz || K^T l` for the focal length.

    Both image axes give an equation and only one of them is worth using. For a
    camera with no roll the horizon is nearly level, so its x coefficient is near
    zero and dividing by it amplifies a tenth of a degree of noise into hundreds
    of pixels of focal length: on a test camera of 760 px the two branches gave
    756 and 329, and averaging them produced 582. So the branch with the larger
    coefficient wins, which is the well-conditioned one by construction.
    """
    cx, cy = centre
    a, b, c = (float(v) for v in horizon)
    depth = a * cx + b * cy + c
    numerator, coefficient = (
        (float(vz[0]) - cx, a) if abs(a) > abs(b) else (float(vz[1]) - cy, b)
    )
    if abs(coefficient) < 1e-6:
        return None
    value = numerator * depth / coefficient
    return math.sqrt(value) if value > 0 else None


def calibrate_from_people(
    poses: list[PoseFrame],
    image_size: tuple[int, int],
    *,
    stature_m: float = DEFAULT_STATURE_M,
    floor_extent_m: float = 4.0,
    focal_px: float | None = None,
) -> Calibration:
    """Recover a metric floor frame from a person walking about the room.

    `focal_px` short-circuits the hardest part. When the camera's focal length is
    known -- and for a fixed installation or a published dataset it usually is --
    the floor's normal is simply the vertical direction in camera coordinates,
    `K^-1 vz`, and no horizon has to be estimated at all. That removes the one
    step in this procedure that is genuinely fragile on real footage, so supply
    it whenever it is available.
    """
    width, height = image_size
    centre = (width / 2.0, height / 2.0)
    stances = upright_stances(poses)
    if len(stances) < 4:
        return Calibration(
            ok=False,
            stances_used=len(stances),
            reason=(
                f"only {len(stances)} clean standing frames were found; "
                "walk someone slowly across the room in view of the camera"
            ),
        )

    vz = vertical_vanishing_point(stances)
    if vz is None:
        return Calibration(
            ok=False,
            stances_used=len(stances),
            reason="the body axes were too close to parallel to fix the vertical",
        )

    horizon: np.ndarray | None = None
    if focal_px is not None:
        focal = float(focal_px)
    else:
        horizon = horizon_from_stances(stances)
        # A horizon tilted by more than about eight degrees means the pair
        # construction has been eaten by noise rather than that the camera is
        # rolled.
        if horizon is None or abs(float(horizon[0])) > 0.15:
            horizon = level_horizon(stances, vz, height)
        if horizon is None:
            return Calibration(
                ok=False,
                stances_used=len(stances),
                vz=(float(vz[0]), float(vz[1])),
                reason=(
                    "the floor's horizon could not be fixed from these frames; "
                    "supply the camera's focal length instead"
                ),
            )
        focal = _focal_from(vz, horizon, centre)
    if focal is None or not (0.2 * width < focal < 8.0 * width):
        return Calibration(
            ok=False,
            stances_used=len(stances),
            vz=(float(vz[0]), float(vz[1])),
            reason=(
                "the recovered focal length is not physically plausible, which "
                "usually means the standing frames were all at the same depth"
            ),
        )

    k = np.array([[focal, 0.0, centre[0]], [0.0, focal, centre[1]], [0.0, 0.0, 1.0]])
    k_inv = np.linalg.inv(k)

    # The floor's normal in camera coordinates is the vertical direction, which
    # is exactly what the vertical vanishing point back-projects to. Orient it so
    # that it points from the floor toward the ceiling in the image, which is the
    # sign convention everything downstream assumes.
    normal = k_inv @ np.array([*vz, 1.0])
    normal = normal / np.linalg.norm(normal)
    if normal[1] < 0:
        normal = -normal

    # Camera height: a stance of known stature pins the scale.
    reference = max(stances, key=lambda s: s.image_height)
    low = ANKLE_ABOVE_FLOOR_M
    high = stature_m - EAR_BELOW_CROWN_M
    camera_height = _camera_height(reference, k_inv, normal, low, high)
    if camera_height is None or not (0.6 < camera_height < 6.0):
        return Calibration(
            ok=False,
            focal_px=focal,
            stances_used=len(stances),
            reason="the recovered camera height is not plausible for a room",
        )

    plane = _build_plane(
        k, normal, camera_height, reference, k_inv, floor_extent_m, low, high
    )
    if plane is None:
        return Calibration(
            ok=False, focal_px=focal, stances_used=len(stances),
            reason="the floor frame came out degenerate",
        )

    return Calibration(
        ok=True,
        plane=plane,
        focal_px=focal,
        camera_height_m=camera_height,
        stances_used=len(stances),
        vz=(float(vz[0]), float(vz[1])),
        horizon=None if horizon is None else tuple(float(v) for v in horizon),
        assumptions=[
            f"the person used for scale is {stature_m:.2f} m tall from the floor to the "
            "crown; every height scales with that number",
            "square pixels, no skew, and the principal point at the image centre",
            "the floor is flat and level",
            "the camera is not rolled, if the level-horizon search was used",
        ],
    )


def _camera_height(
    stance: Stance, k_inv: np.ndarray, normal: np.ndarray, low_m: float, high_m: float
) -> float | None:
    """How far the camera is above the floor, from one person of known stature.

    The two endpoints of the measured segment lie on the same vertical world
    line, so their rays satisfy `X_head = X_foot - n (high - low)`. Splitting
    that into the component along the floor normal and the component across it
    gives two relations, and eliminating the two unknown depths leaves the camera
    height in closed form:

        A = |r_f x n| (n . r_h) / (|r_h x n| (n . r_f))
        H = (high - A low) / (1 - A)

    The earlier version of this dropped the across-normal terms, which is only
    valid for a camera pointing exactly at the horizon and was wrong by a metre
    for one pointing slightly down.
    """
    foot_ray = k_inv @ np.array([*stance.foot, 1.0])
    head_ray = k_inv @ np.array([*stance.head, 1.0])
    foot_ray = foot_ray / np.linalg.norm(foot_ray)
    head_ray = head_ray / np.linalg.norm(head_ray)
    foot_along = float(np.dot(normal, foot_ray))
    head_along = float(np.dot(normal, head_ray))
    if abs(foot_along) < 1e-9 or abs(head_along) < 1e-9:
        return None
    foot_across = math.sqrt(max(0.0, 1.0 - foot_along**2))
    head_across = math.sqrt(max(0.0, 1.0 - head_along**2))
    if head_across < 1e-9:
        return None
    a = foot_across * head_along / (head_across * foot_along)
    if abs(1.0 - a) < 1e-9:
        return None
    height = (high_m - a * low_m) / (1.0 - a)
    return float(height) if math.isfinite(height) else None


def _build_plane(
    k: np.ndarray,
    normal: np.ndarray,
    camera_height: float,
    reference: Stance,
    k_inv: np.ndarray,
    extent_m: float,
    low_m: float,
    high_m: float,
) -> FloorPlane | None:
    """Turn the recovered camera into the four points and vertical `RoomConfig` wants."""
    # Two in-plane axes. Any pair will do; the room's own axes are unknown and do
    # not matter, because every quantity the engine reports is a distance.
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, normal))) > 0.9:
        seed = np.array([0.0, 0.0, 1.0])
    axis_u = seed - float(np.dot(seed, normal)) * normal
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)

    # Put the origin under the reference person's feet.
    foot_ray = k_inv @ np.array([*reference.foot, 1.0])
    foot_ray = foot_ray / np.linalg.norm(foot_ray)
    depth = (camera_height - low_m) / float(np.dot(normal, foot_ray))
    ankle = foot_ray * depth
    origin = ankle + normal * low_m  # the floor directly under the ankle

    half = extent_m / 2.0
    corners_world = [
        origin + axis_u * -half + axis_v * -half,
        origin + axis_u * half + axis_v * -half,
        origin + axis_u * half + axis_v * half,
        origin + axis_u * -half + axis_v * half,
    ]
    image_points = []
    for point in corners_world:
        projected = k @ point
        if projected[2] <= 1e-6:
            return None
        image_points.append((float(projected[0] / projected[2]), float(projected[1] / projected[2])))
    world_points = [(-half, -half), (half, -half), (half, half), (-half, half)]

    top_world = origin - normal * (high_m - low_m + low_m)
    projected = k @ top_world
    if projected[2] <= 1e-6:
        return None
    top_px = (float(projected[0] / projected[2]), float(projected[1] / projected[2]))

    return FloorPlane(
        image_points=tuple(image_points),
        world_points=tuple(world_points),
        vertical_vanishing_point=tuple(_vanishing(k, normal)),
        reference_height_m=high_m,
        reference_base_px=_project(k, origin),
        reference_top_px=top_px,
    )


def _project(k: np.ndarray, point: np.ndarray) -> tuple[float, float]:
    projected = k @ point
    return (float(projected[0] / projected[2]), float(projected[1] / projected[2]))


def _vanishing(k: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    point = k @ (-normal)
    if abs(point[2]) < 1e-9:
        return (float(point[0]) * 1e6, float(point[1]) * 1e6)
    return (float(point[0] / point[2]), float(point[1] / point[2]))
