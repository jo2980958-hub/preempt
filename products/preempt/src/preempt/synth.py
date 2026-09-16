"""Synthetic rooms and synthetic people, so some answers are known exactly.

Real fall footage tells you whether a system noticed. It cannot tell you how many
seconds of warning it gave, because nobody labelled the frame where the patient
decided to get up. These sequences can, because the decision is written into them.

How they are built
------------------
A pinhole camera in a corner of a room of known size. A body model of seventeen
joints in metres. Postures - lying, sitting, on the edge, leaning, standing - and
smooth interpolation between them. The floor homography, the vertical vanishing
point and the reference height are then *derived from the same camera*, not fitted,
so the geometry the engine uses is exactly the geometry that generated the scene
and any error in the result is the engine's.

Then the honest part: keypoints get Gaussian noise at the level a real pose
estimator produces (measured against RTMPose's own reported accuracy, about 4 px
at this image size), joints drop out at random, and scores vary. A synthetic
sequence with clean keypoints would flatter every threshold in the product.

What is and is not proved by these
----------------------------------
They prove the decision layer: given pose, does the state machine call at the
right moment, how many seconds ahead, and how often does it call when it should
not. They prove nothing about pose estimation on a real patient in a real bed
with a real blanket, which is what `docs/evaluation.md` uses UR Fall for and is
explicit about.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import FloorPlane, RoomConfig, Thresholds, Zone
from .pose import KEYPOINT_NAMES, PoseFrame

RNG_SEED = 20261026

# Body-frame joint layout for a 1.70 m adult, standing, facing +x, up is +z.
STANDING_BODY: dict[str, tuple[float, float, float]] = {
    "nose": (0.09, 0.0, 1.62),
    "left eye": (0.07, 0.035, 1.645),
    "right eye": (0.07, -0.035, 1.645),
    "left ear": (0.0, 0.075, 1.63),
    "right ear": (0.0, -0.075, 1.63),
    "left shoulder": (0.0, 0.185, 1.40),
    "right shoulder": (0.0, -0.185, 1.40),
    "left elbow": (0.02, 0.205, 1.11),
    "right elbow": (0.02, -0.205, 1.11),
    "left wrist": (0.05, 0.215, 0.86),
    "right wrist": (0.05, -0.215, 0.86),
    "left hip": (0.0, 0.10, 0.93),
    "right hip": (0.0, -0.10, 0.93),
    "left knee": (0.01, 0.105, 0.49),
    "right knee": (0.01, -0.105, 0.49),
    "left ankle": (0.0, 0.105, 0.07),
    "right ankle": (0.0, -0.105, 0.07),
}

UPPER = (
    "nose", "left eye", "right eye", "left ear", "right ear",
    "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist",
)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SynthCamera:
    """A pinhole camera in a corner of the room, and everything derived from it."""

    width: int = 1280
    height: int = 720
    focal: float = 760.0
    position: tuple[float, float, float] = (0.35, 0.35, 2.55)
    target: tuple[float, float, float] = (2.0, 1.9, 0.85)

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.focal, 0.0, self.width / 2.0], [0.0, self.focal, self.height / 2.0], [0.0, 0.0, 1.0]]
        )

    @property
    def Rt(self) -> tuple[np.ndarray, np.ndarray]:
        eye = np.asarray(self.position, dtype=np.float64)
        forward = np.asarray(self.target, dtype=np.float64) - eye
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward])  # rows: camera x, y, z axes
        translation = -rotation @ eye
        return rotation, translation

    def project(self, points_world: np.ndarray) -> np.ndarray:
        rotation, translation = self.Rt
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        camera = pts @ rotation.T + translation
        z = np.maximum(camera[:, 2], 1e-6)
        image = (self.K @ (camera / z[:, None]).T).T
        return image[:, :2]

    def floor_plane(self, corners_m: np.ndarray, reference_xy: tuple[float, float] = (2.0, 1.2)) -> FloorPlane:
        """The FloorPlane the engine will be given, derived from this exact camera."""
        world3 = np.concatenate([corners_m, np.zeros((corners_m.shape[0], 1))], axis=1)
        image_points = self.project(world3)
        rotation, _ = self.Rt
        vertical = self.K @ rotation @ np.array([0.0, 0.0, 1.0])
        vz = (vertical[0] / vertical[2], vertical[1] / vertical[2])
        base = self.project(np.array([[reference_xy[0], reference_xy[1], 0.0]]))[0]
        top = self.project(np.array([[reference_xy[0], reference_xy[1], 1.70]]))[0]
        return FloorPlane(
            image_points=tuple((float(x), float(y)) for x, y in image_points),
            world_points=tuple((float(x), float(y)) for x, y in corners_m),
            vertical_vanishing_point=(float(vz[0]), float(vz[1])),
            reference_height_m=1.70,
            reference_base_px=(float(base[0]), float(base[1])),
            reference_top_px=(float(top[0]), float(top[1])),
        )


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------


def _body_matrix() -> np.ndarray:
    return np.array([STANDING_BODY[name] for name in KEYPOINT_NAMES], dtype=np.float64)


def posture(
    *,
    seat_height: float = 0.0,
    knee_bend: float = 0.0,
    trunk_lean_deg: float = 0.0,
    lying: float = 0.0,
) -> np.ndarray:
    """Body-frame joints for a blend of the postures the product cares about.

    `knee_bend` 0 is standing and 1 is a right angle at the knee with the shank
    vertical and the thigh horizontal, which is sitting. `seat_height` lifts the
    hips onto a bed or chair. `trunk_lean_deg` rotates everything above the hips
    forward about the hip axis, which is Phase I of standing up. `lying` rotates
    the whole body about the ankles onto the horizontal.
    """
    joints = _body_matrix().copy()
    hip_z = 0.93

    if knee_bend > 0:
        thigh = 0.44
        for name in ("left knee", "right knee"):
            i = KEYPOINT_NAMES.index(name)
            joints[i, 0] += thigh * knee_bend
            joints[i, 2] = 0.07 + (0.49 - 0.07) * (1.0 - knee_bend * 0.10)
        for name in ("left ankle", "right ankle"):
            i = KEYPOINT_NAMES.index(name)
            joints[i, 0] += thigh * knee_bend
        drop = (0.93 - 0.55) * knee_bend
        for name in ("left hip", "right hip", *UPPER):
            i = KEYPOINT_NAMES.index(name)
            joints[i, 2] -= drop
        hip_z -= drop

    if seat_height > 0:
        lift = seat_height - (hip_z - 0.55 * 0 - 0.0)
        lift = seat_height - hip_z
        for name in ("left hip", "right hip", *UPPER):
            i = KEYPOINT_NAMES.index(name)
            joints[i, 2] += lift
        for name in ("left knee", "right knee"):
            i = KEYPOINT_NAMES.index(name)
            joints[i, 2] += lift * 0.55
        hip_z = seat_height

    if abs(trunk_lean_deg) > 1e-6:
        angle = math.radians(trunk_lean_deg)
        pivot = np.array([0.0, 0.0, hip_z])
        c, s = math.cos(angle), math.sin(angle)
        for name in UPPER:
            i = KEYPOINT_NAMES.index(name)
            v = joints[i] - pivot
            joints[i] = pivot + np.array([v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c])

    if lying > 0:
        angle = math.radians(90.0 * lying)
        pivot = np.array([0.0, 0.0, 0.10])
        c, s = math.cos(angle), math.sin(angle)
        for i in range(joints.shape[0]):
            v = joints[i] - pivot
            joints[i] = pivot + np.array([v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c])
    return joints


def place(joints: np.ndarray, floor_xy: tuple[float, float], yaw_deg: float, lift_z: float = 0.0) -> np.ndarray:
    """Body frame to world: rotate about the vertical, translate onto the floor."""
    angle = math.radians(yaw_deg)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out = joints @ rotation.T
    out[:, 0] += floor_xy[0]
    out[:, 1] += floor_xy[1]
    out[:, 2] += lift_z
    return out


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass
class Sequence:
    """A synthetic sequence, its ground truth, and the room it happened in."""

    name: str
    description: str
    fps: float
    frames: list[PoseFrame]
    room: RoomConfig
    truth: dict[str, Any] = field(default_factory=dict)
    view_hints: dict[int, str] = field(default_factory=dict)
    """Frame index to the view verdict the device recorded at capture time."""

    @property
    def duration_s(self) -> float:
        return self.frames[-1].time_s if self.frames else 0.0

    def to_track(self) -> dict[str, Any]:
        """The on-disk form: a pose track, which is all a real deployment keeps."""
        return {
            "kind": "preempt-pose-track",
            "version": "1",
            "name": self.name,
            "description": self.description,
            "fps": self.fps,
            "room": self.room.to_dict(),
            "truth": self.truth,
            "view_hints": {str(k): v for k, v in self.view_hints.items()},
            "keypoint_names": list(KEYPOINT_NAMES),
            "frames": [
                {
                    "index": f.index,
                    "time_s": round(f.time_s, 4),
                    "xy": [[round(float(x), 2), round(float(y), 2)] for x, y in f.xy],
                    "scores": [round(float(s), 3) for s in f.scores],
                }
                for f in self.frames
            ],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_track()), encoding="utf-8")
        return target


def load_track(path: str | Path) -> tuple[list[PoseFrame], RoomConfig, dict[str, Any]]:
    """Read a pose track back. The inverse of `Sequence.save`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("kind") != "preempt-pose-track":
        raise ValueError(f"{path} is not a Preempt pose track")
    frames = [
        PoseFrame(
            index=int(f["index"]),
            time_s=float(f["time_s"]),
            xy=np.asarray(f["xy"], dtype=np.float64),
            scores=np.asarray(f["scores"], dtype=np.float64),
        )
        for f in data["frames"]
    ]
    meta = {
        "fps": float(data.get("fps", 30.0)),
        "name": data.get("name", ""),
        "description": data.get("description", ""),
        "truth": data.get("truth", {}),
        "view_hints": {int(k): v for k, v in (data.get("view_hints") or {}).items()},
    }
    return frames, RoomConfig.from_dict(data["room"]), meta


def default_room(camera: SynthCamera | None = None) -> tuple[RoomConfig, SynthCamera]:
    """A single side room: a bed against one wall, a chair, a door, a walking frame."""
    cam = camera or SynthCamera()
    corners = np.array([[0.2, 0.2], [4.0, 0.2], [4.0, 3.4], [0.2, 3.4]])
    plane = cam.floor_plane(corners)

    def zone(name: str, kind: str, rect: tuple[float, float, float, float]) -> Zone:
        x0, y0, x1, y1 = rect
        world = np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])
        pts = cam.project(world)
        return Zone(name=name, kind=kind, points=tuple((float(x), float(y)) for x, y in pts))

    zones = (
        zone("bed 4", "bed", (0.75, 0.30, 3.60, 1.45)),
        zone("chair", "chair", (3.2, 1.0, 3.9, 1.7)),
        zone("door", "door", (0.3, 2.9, 1.2, 3.4)),
        zone("wall", "wall", (0.2, 0.2, 0.45, 3.4)),
        zone("walking frame", "aid", (3.3, 2.2, 3.9, 2.8)),
    )
    room = RoomConfig(
        room="side room 4",
        bed_name="bed 4",
        floor=plane,
        zones=zones,
        thresholds=Thresholds(),
        notes="Synthetic room used for the bundled samples and the evaluation harness.",
    )
    return room, cam


# --- animation helpers -----------------------------------------------------


def _ease(t: float) -> float:
    return float(np.clip(t, 0.0, 1.0) ** 2 * (3 - 2 * np.clip(t, 0.0, 1.0)))


class _Builder:
    """Accumulates world-space joint sets frame by frame, then projects them once."""

    def __init__(self, camera: SynthCamera, fps: float, rng: np.random.Generator) -> None:
        self.camera = camera
        self.fps = fps
        self.rng = rng
        self.world: list[np.ndarray] = []
        self.events: dict[str, float] = {}

    @property
    def t(self) -> float:
        return len(self.world) / self.fps

    def mark(self, name: str) -> None:
        self.events.setdefault(name, self.t)

    def hold(self, joints_world: np.ndarray, seconds: float) -> None:
        for _ in range(int(round(seconds * self.fps))):
            self.world.append(joints_world.copy())

    def ramp(self, make, seconds: float) -> None:
        n = max(1, int(round(seconds * self.fps)))
        for i in range(n):
            self.world.append(make(_ease((i + 1) / n)))

    def build(
        self, name: str, description: str, room: RoomConfig, truth: dict[str, Any],
        *, noise_px: float = 4.0, dropout: float = 0.04,
    ) -> Sequence:
        frames: list[PoseFrame] = []
        for i, joints in enumerate(self.world):
            xy = self.camera.project(joints)
            xy = xy + self.rng.normal(0.0, noise_px, size=xy.shape)
            scores = np.clip(self.rng.normal(0.82, 0.08, size=len(KEYPOINT_NAMES)), 0.0, 0.99)
            missing = self.rng.random(len(KEYPOINT_NAMES)) < dropout
            scores[missing] = self.rng.uniform(0.02, 0.18, size=int(missing.sum()))
            frames.append(PoseFrame(index=i, time_s=i / self.fps, xy=xy, scores=scores))
        truth = {**truth, "events": {k: round(v, 3) for k, v in self.events.items()}}
        return Sequence(name, description, self.fps, frames, room, truth)


def _walk(
    builder: _Builder,
    start: tuple[float, float],
    end: tuple[float, float],
    seconds: float,
    *,
    sway_cm: float = 2.0,
    step_hz: float = 1.7,
    jitter: float = 0.0,
    yaw: float | None = None,
) -> None:
    """A walk across the floor with a sway amplitude and a step cadence."""
    start_v, end_v = np.array(start), np.array(end)
    direction = end_v - start_v
    heading = math.degrees(math.atan2(direction[1], direction[0])) if np.linalg.norm(direction) > 1e-6 else 0.0
    normal = np.array([-direction[1], direction[0]])
    normal = normal / max(1e-6, float(np.linalg.norm(normal)))
    n = max(1, int(round(seconds * builder.fps)))
    phase = 0.0
    for i in range(n):
        u = (i + 1) / n
        base = start_v + direction * u
        sway = math.sin(2 * math.pi * step_hz * (i / builder.fps) * 0.5) * (sway_cm / 100.0)
        wobble = builder.rng.normal(0.0, jitter) if jitter else 0.0
        position = base + normal * (sway + wobble)
        phase += (step_hz + (builder.rng.normal(0.0, jitter * 6) if jitter else 0.0)) / builder.fps
        joints = posture()
        swing = math.sin(2 * math.pi * phase) * 0.22
        for side, sign in (("left", 1.0), ("right", -1.0)):
            for part, factor in (("ankle", 1.0), ("knee", 0.5)):
                idx = KEYPOINT_NAMES.index(f"{side} {part}")
                joints[idx, 0] += swing * sign * factor
                joints[idx, 2] += max(0.0, swing * sign * factor) * 0.35
        builder.world.append(
            place(joints, (float(position[0]), float(position[1])), yaw if yaw is not None else heading)
        )


# --- the scenarios ---------------------------------------------------------

BED_XY = (1.9, 0.9)
BED_EDGE_XY = (1.9, 1.55)
BED_SURFACE_Z = 0.62
DOOR_XY = (0.75, 3.1)


def _bed_exit_core(builder: _Builder, *, lean_seconds: float = 1.6, settle_s: float = 3.0) -> None:
    """Lying in bed, stirring, sitting up, on the edge, leaning, standing."""
    lying = place(posture(lying=1.0), BED_XY, 0.0, lift_z=BED_SURFACE_Z)
    builder.mark("in_bed")
    builder.hold(lying, settle_s)

    builder.ramp(
        lambda u: place(posture(lying=1.0 - 0.35 * u), BED_XY, 0.0, lift_z=BED_SURFACE_Z), 1.2
    )
    builder.mark("stirring")
    builder.ramp(
        lambda u: place(
            posture(lying=0.65 - 0.65 * u, knee_bend=0.9 * u, seat_height=BED_SURFACE_Z * u),
            BED_XY, 0.0, lift_z=BED_SURFACE_Z * (1 - u),
        ),
        1.6,
    )
    builder.mark("sitting_up")
    builder.ramp(
        lambda u: place(
            posture(knee_bend=0.95, seat_height=BED_SURFACE_Z),
            (BED_XY[0], BED_XY[1] + (BED_EDGE_XY[1] - BED_XY[1]) * u), 90.0,
        ),
        1.4,
    )
    builder.mark("on_edge")
    builder.hold(place(posture(knee_bend=0.95, seat_height=BED_SURFACE_Z), BED_EDGE_XY, 90.0), 1.0)

    # Phase I: the trunk goes forward over the feet. This is what Preempt calls on.
    builder.mark("lean_starts")
    builder.ramp(
        lambda u: place(
            posture(knee_bend=0.95, seat_height=BED_SURFACE_Z, trunk_lean_deg=42.0 * u),
            BED_EDGE_XY, 90.0,
        ),
        lean_seconds,
    )
    # Phases II and III: weight off the seat, knees extend.
    builder.ramp(
        lambda u: place(
            posture(
                knee_bend=0.95 * (1 - u),
                seat_height=BED_SURFACE_Z + (0.93 - BED_SURFACE_Z) * u,
                trunk_lean_deg=42.0 * (1 - u),
            ),
            (BED_EDGE_XY[0], BED_EDGE_XY[1] + 0.12 * u), 90.0,
        ),
        1.3,
    )
    builder.mark("standing")
    builder.hold(place(posture(), (BED_EDGE_XY[0], BED_EDGE_XY[1] + 0.12), 90.0), 0.6)


def scenario_bed_exit_steady(camera: SynthCamera, room: RoomConfig, fps: float, rng) -> Sequence:
    b = _Builder(camera, fps, rng)
    _bed_exit_core(b)
    _walk(b, (BED_EDGE_XY[0], BED_EDGE_XY[1] + 0.12), DOOR_XY, 4.0, sway_cm=2.0, step_hz=1.8)
    return b.build(
        "bed-exit-steady",
        "A patient gets out of bed unaided and walks steadily to the door.",
        room,
        {"should_call": True, "expect_state": "rising soon", "expect_unsteady": False},
    )


def scenario_bed_exit_unsteady(camera: SynthCamera, room: RoomConfig, fps: float, rng) -> Sequence:
    b = _Builder(camera, fps, rng)
    _bed_exit_core(b, lean_seconds=2.0)
    _walk(b, (BED_EDGE_XY[0], BED_EDGE_XY[1] + 0.12), DOOR_XY, 6.5, sway_cm=11.0, step_hz=1.1, jitter=0.05)
    return b.build(
        "bed-exit-unsteady",
        "The same exit, but the walk that follows is wide and irregular.",
        room,
        {"should_call": True, "expect_state": "rising soon", "expect_unsteady": True},
    )


def scenario_bed_exit_to_floor(camera: SynthCamera, room: RoomConfig, fps: float, rng) -> Sequence:
    b = _Builder(camera, fps, rng)
    _bed_exit_core(b)
    _walk(b, (BED_EDGE_XY[0], BED_EDGE_XY[1] + 0.12), (1.6, 2.2), 2.4, sway_cm=9.0, step_hz=1.2, jitter=0.05)
    b.mark("fall_starts")
    b.ramp(lambda u: place(posture(lying=u), (1.6, 2.2), 90.0), 0.8)
    b.mark("on_floor")
    b.hold(place(posture(lying=1.0), (1.6, 2.2), 90.0), 4.0)
    return b.build(
        "bed-exit-to-floor",
        "The exit is not caught in time and the patient ends up on the floor.",
        room,
        {"should_call": True, "expect_state": "on the floor", "expect_unsteady": True},
    )


def scenario_settled(camera: SynthCamera, room: RoomConfig, fps: float, rng, loops: int = 1) -> Sequence:
    b = _Builder(camera, fps, rng)
    lying = place(posture(lying=1.0), BED_XY, 0.0, lift_z=BED_SURFACE_Z)
    for _ in range(loops):
        b.hold(lying, 6.0)
        b.ramp(lambda u: place(posture(lying=1.0), BED_XY, 18.0 * u, lift_z=BED_SURFACE_Z), 1.5)
        b.hold(place(posture(lying=1.0), BED_XY, 18.0, lift_z=BED_SURFACE_Z), 6.0)
        b.ramp(lambda u: place(posture(lying=1.0), BED_XY, 18.0 * (1 - u), lift_z=BED_SURFACE_Z), 1.5)
    b.hold(lying, 5.0)
    return b.build(
        "settled-turning-over",
        "A patient asleep in bed who turns over twice. No call should be raised.",
        room,
        {"should_call": False, "expect_state": "settled", "expect_unsteady": False},
    )


def scenario_sit_up_and_lie_back(camera: SynthCamera, room: RoomConfig, fps: float, rng, loops: int = 1) -> Sequence:
    b = _Builder(camera, fps, rng)
    for _ in range(loops):
        _sit_up_cycle(b)
    return b.build(
        "sat-up-and-lay-back",
        "A patient sits up in bed to drink and lies back down. No call should be raised.",
        room,
        {"should_call": False, "expect_state": "watch", "expect_unsteady": False},
    )


def _sit_up_cycle(b: _Builder) -> None:
    b.hold(place(posture(lying=1.0), BED_XY, 0.0, lift_z=BED_SURFACE_Z), 3.0)
    b.ramp(
        lambda u: place(
            posture(lying=1.0 - u, knee_bend=0.9 * u, seat_height=BED_SURFACE_Z * u),
            BED_XY, 0.0, lift_z=BED_SURFACE_Z * (1 - u),
        ),
        1.8,
    )
    b.hold(place(posture(knee_bend=0.9, seat_height=BED_SURFACE_Z), BED_XY, 0.0), 5.0)
    b.ramp(
        lambda u: place(
            posture(lying=u, knee_bend=0.9 * (1 - u), seat_height=BED_SURFACE_Z * (1 - u)),
            BED_XY, 0.0, lift_z=BED_SURFACE_Z * u,
        ),
        1.8,
    )
    b.hold(place(posture(lying=1.0), BED_XY, 0.0, lift_z=BED_SURFACE_Z), 4.0)


def scenario_visitor_in_chair(camera: SynthCamera, room: RoomConfig, fps: float, rng, loops: int = 1) -> Sequence:
    b = _Builder(camera, fps, rng)
    seated = place(posture(knee_bend=0.95, seat_height=0.46), (3.5, 1.35), 200.0)
    for _ in range(loops):
        b.hold(seated, 8.0)
        b.ramp(
            lambda u: place(
                posture(knee_bend=0.95, seat_height=0.46, trunk_lean_deg=12.0 * math.sin(u * math.pi)),
                (3.5, 1.35), 200.0,
            ),
            3.0,
        )
    b.hold(seated, 6.0)
    return b.build(
        "visitor-in-chair",
        "Someone sits in the chair and shifts position without getting up.",
        room,
        {"should_call": False, "expect_state": "watch", "expect_unsteady": False},
    )


def scenario_steady_walk(camera: SynthCamera, room: RoomConfig, fps: float, rng, loops: int = 1) -> Sequence:
    b = _Builder(camera, fps, rng)
    for _ in range(loops):
        _walk(b, (3.4, 2.6), (1.0, 1.0), 5.0, sway_cm=1.8, step_hz=1.9)
        _walk(b, (1.0, 1.0), (3.4, 2.6), 5.0, sway_cm=2.1, step_hz=1.9)
    return b.build(
        "steady-walk",
        "A member of staff walks across the room steadily. Gait should not be called unsteady.",
        room,
        {"should_call": False, "expect_state": "watch", "expect_unsteady": False},
    )


def scenario_unsteady_walk(camera: SynthCamera, room: RoomConfig, fps: float, rng) -> Sequence:
    b = _Builder(camera, fps, rng)
    _walk(b, (3.4, 2.6), (1.0, 1.0), 8.0, sway_cm=13.0, step_hz=0.95, jitter=0.07)
    return b.build(
        "unsteady-walk",
        "A patient crosses the room swaying widely with uneven steps.",
        room,
        {"should_call": True, "expect_state": "unsteady", "expect_unsteady": True},
    )


def scenario_curtain_drawn(camera: SynthCamera, room: RoomConfig, fps: float, rng) -> Sequence:
    """The honest failure: a curtain goes across the lens mid-sequence.

    Every monitoring product needs one of these and most of them do not have one.
    Here the patient starts to get up and the view is lost first, so the product
    has to say it cannot see rather than carry on producing a calm green state.
    """
    b = _Builder(camera, fps, rng)
    b.hold(place(posture(lying=1.0), BED_XY, 0.0, lift_z=BED_SURFACE_Z), 4.0)
    b.ramp(
        lambda u: place(
            posture(lying=1.0 - u, knee_bend=0.9 * u, seat_height=BED_SURFACE_Z * u),
            BED_XY, 0.0, lift_z=BED_SURFACE_Z * (1 - u),
        ),
        2.0,
    )
    blocked_from = len(b.world)
    b.hold(place(posture(knee_bend=0.9, seat_height=BED_SURFACE_Z), BED_XY, 0.0), 10.0)
    sequence = b.build(
        "curtain-drawn",
        "A curtain is pulled across the lens while the patient is sitting up. "
        "The product must say it cannot see.",
        room,
        {"should_call": False, "expect_state": "view unusable", "expect_unsteady": False},
        noise_px=9.0,
        dropout=0.45,
    )
    sequence.view_hints = {i: "blocked" for i in range(blocked_from, len(sequence.frames))}
    return sequence


def scenario_personal_care(camera: SynthCamera, room: RoomConfig, fps: float, rng) -> Sequence:
    """The cost of privacy, made visible: staff pause the camera for washing.

    Falls cluster around bathing, toileting and dressing, and those are exactly
    the activities a ward will pause the camera for. A product that quietly showed
    a calm green light through that window would be lying by omission. This
    sequence exists so that the interface, the run record and the report all have
    to say how long the system was blind by choice, and that what happened then is
    unknowable rather than nothing.
    """
    b = _Builder(camera, fps, rng)
    b.hold(place(posture(lying=1.0), BED_XY, 0.0, lift_z=BED_SURFACE_Z), 4.0)
    b.ramp(
        lambda u: place(
            posture(lying=1.0 - u, knee_bend=0.9 * u, seat_height=BED_SURFACE_Z * u),
            BED_XY, 0.0, lift_z=BED_SURFACE_Z * (1 - u),
        ),
        1.5,
    )
    paused_from = len(b.world)
    b.hold(place(posture(knee_bend=0.9, seat_height=BED_SURFACE_Z), BED_XY, 0.0), 9.0)
    paused_to = len(b.world)
    b.hold(place(posture(lying=1.0), BED_XY, 0.0, lift_z=BED_SURFACE_Z), 4.0)
    sequence = b.build(
        "personal-care-pause",
        "Staff pause the camera while a patient is washed. The product must say how "
        "long it was blind and that the window is unknowable.",
        room,
        {"should_call": False, "expect_state": "paused", "expect_unsteady": False},
    )
    sequence.view_hints = {
        i: "paused for personal care" for i in range(paused_from, paused_to)
    }
    return sequence


SCENARIOS = {
    "bed-exit-steady": scenario_bed_exit_steady,
    "curtain-drawn": scenario_curtain_drawn,
    "personal-care-pause": scenario_personal_care,
    "bed-exit-unsteady": scenario_bed_exit_unsteady,
    "bed-exit-to-floor": scenario_bed_exit_to_floor,
    "settled-turning-over": scenario_settled,
    "sat-up-and-lay-back": scenario_sit_up_and_lie_back,
    "visitor-in-chair": scenario_visitor_in_chair,
    "steady-walk": scenario_steady_walk,
    "unsteady-walk": scenario_unsteady_walk,
}


QUIET = ("settled-turning-over", "sat-up-and-lay-back", "visitor-in-chair", "steady-walk")
"""The negatives. Their length is what makes a false-alarm rate mean anything."""


def make(name: str, *, fps: float = 30.0, seed: int = RNG_SEED, loops: int = 1) -> Sequence:
    """One sequence. `loops` repeats a quiet scenario to buy observed quiet hours.

    A false-alarm rate computed over twenty sequences of fifteen seconds each is
    arithmetic, not evidence: five minutes of quiet extrapolated to a bed-night
    multiplies any single stray call by 144. So the quiet scenarios loop, and the
    evaluation runs them long enough that the denominator is worth dividing by.
    """
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; known: {sorted(SCENARIOS)}")
    room, camera = default_room()
    rng = np.random.default_rng(seed)
    builder = SCENARIOS[name]
    if name in QUIET and loops > 1:
        return builder(camera, room, fps, rng, loops)
    return builder(camera, room, fps, rng)


def make_all(*, fps: float = 30.0, seed: int = RNG_SEED) -> list[Sequence]:
    return [make(name, fps=fps, seed=seed + i) for i, name in enumerate(sorted(SCENARIOS))]
