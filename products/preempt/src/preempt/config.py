"""Room setup and the thresholds the ward can change.

Everything a ward configures once lives here: where the bed is, where the chair
is, where the door is, which four floor points define the plane, and how patient
or twitchy the escalation should be. Nothing in the engine reads a magic number
that is not on one of these dataclasses.

Units, stated once so they are never ambiguous downstream:
  * image space is pixels, origin top-left, y down;
  * floor space is **metres**, a right-handed 2D frame on the floor plane with an
    origin the ward picks (normally a room corner);
  * heights are metres above the floor plane;
  * times are seconds of video time, not wall clock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ZONE_KINDS = ("bed", "chair", "door", "wall", "aid", "exclude")


@dataclass(frozen=True)
class Zone:
    """A polygon a member of staff drew once, in image pixels.

    `kind` decides what the engine does with it. `name` is what appears in a
    reason string, so it is written the way a nurse would say it: "bed 4", not
    "zone_0".
    """

    name: str
    kind: str
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if self.kind not in ZONE_KINDS:
            raise ValueError(f"unknown zone kind {self.kind!r}; known: {ZONE_KINDS}")
        if len(self.points) < 3:
            raise ValueError(f"zone {self.name!r} needs at least 3 points, got {len(self.points)}")

    @property
    def polygon(self) -> np.ndarray:
        return np.asarray(self.points, dtype=np.float32).reshape(-1, 2)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "points": [list(p) for p in self.points]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Zone:
        return cls(
            name=data["name"],
            kind=data["kind"],
            points=tuple((float(x), float(y)) for x, y in data["points"]),
        )


@dataclass(frozen=True)
class FloorPlane:
    """Four image points that are known to lie on the floor, and their real spacing.

    `image_points` are in the order the ward clicked them; `world_points` are the
    same four positions in floor metres. The usual setup is the four corners of a
    floor rectangle whose sides the ward measured with a tape.

    `vertical_vanishing_point` is optional. With it, the engine can state a
    height above the floor in metres (single-view metrology, Criminisi et al.).
    Without it, it refuses to state a height and says so rather than guessing.
    """

    image_points: tuple[tuple[float, float], ...]
    world_points: tuple[tuple[float, float], ...]
    vertical_vanishing_point: tuple[float, float] | None = None
    reference_height_m: float = 1.70
    reference_base_px: tuple[float, float] | None = None
    reference_top_px: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if len(self.image_points) != 4 or len(self.world_points) != 4:
            raise ValueError("a floor plane needs exactly 4 image points and 4 world points")

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_points": [list(p) for p in self.image_points],
            "world_points": [list(p) for p in self.world_points],
            "vertical_vanishing_point": (
                list(self.vertical_vanishing_point) if self.vertical_vanishing_point else None
            ),
            "reference_height_m": self.reference_height_m,
            "reference_base_px": (
                list(self.reference_base_px) if self.reference_base_px else None
            ),
            "reference_top_px": list(self.reference_top_px) if self.reference_top_px else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FloorPlane:
        def pair(value: Any) -> tuple[float, float] | None:
            return (float(value[0]), float(value[1])) if value else None

        return cls(
            image_points=tuple((float(x), float(y)) for x, y in data["image_points"]),
            world_points=tuple((float(x), float(y)) for x, y in data["world_points"]),
            vertical_vanishing_point=pair(data.get("vertical_vanishing_point")),
            reference_height_m=float(data.get("reference_height_m", 1.70)),
            reference_base_px=pair(data.get("reference_base_px")),
            reference_top_px=pair(data.get("reference_top_px")),
        )


@dataclass(frozen=True)
class Thresholds:
    """The tuning surface. Defaults are argued for in `docs/evaluation.md`.

    None of these are learned. They are geometric or kinematic quantities with
    units, so a clinical lead can reason about them and change them, which is the
    point of not hiding the decision inside a classifier.
    """

    # --- pose quality -------------------------------------------------------
    keypoint_score_min: float = 0.30
    """A keypoint below this is treated as absent, not as a position."""
    pose_keypoints_min: int = 6
    """Fewer usable keypoints than this and the frame contributes nothing."""

    # --- bed and chair exit -------------------------------------------------
    trunk_upright_deg: float = 45.0
    """Trunk angle from horizontal above which the person counts as sitting up."""
    edge_margin_m: float = 0.25
    """How close to a zone's boundary the hips must be to count as on the edge."""
    transfer_rate_per_s: float = 0.18
    """Rate at which the upper body arrives over the feet that marks Phase I."""
    transfer_delta: float = 0.16
    """Total travel of the lean measure across the window, not just its slope."""
    lean_over_feet: float = -0.32
    """How close the shoulders must already be to over the feet, as a ratio."""
    rise_hip_speed_mps: float = 0.25
    """Upward hip speed that means the person is already on the way up."""
    stand_hip_height_m: float = 0.78
    """Hip height above the floor at which the person is standing, not seated."""
    prepare_hold_s: float = 0.40
    """How long the preparing-to-rise evidence must hold before a call is raised."""
    prepare_grace_s: float = 0.35
    """How long the evidence may lapse without restarting the clock."""

    # --- gait ---------------------------------------------------------------
    gait_window_s: float = 4.0
    """Observation window for every gait statistic."""
    gait_min_steps: int = 4
    """Below this many footfalls the engine reports 'not enough steps', not a score."""
    sway_rms_warn_cm: float = 4.5
    """Lateral sway RMS about the walking line that starts to count against."""
    sway_rms_high_cm: float = 9.0
    """Sway RMS that on its own is enough to call the gait unsteady."""
    step_time_cv_warn: float = 0.12
    """Reported only. At 15 Hz the measurement noise floor is about 0.3, which is
    the size of the effect, so step-time variability is shown and not scored."""
    step_time_cv_high: float = 0.28
    """Reported only; see above."""
    halting_step_ratio: float = 1.8
    """Reported only; a step longer than this multiple of the median is a halt."""
    wall_reach_m: float = 0.45
    """A wrist this close to a wall polygon, held, counts as reaching for support."""
    wall_reach_hold_s: float = 0.6
    gait_unsteady_score: float = 0.55
    """Fused gait score at or above which the state is 'unsteady on feet'."""

    # --- on the floor -------------------------------------------------------
    floor_head_height_m: float = 0.60
    """Reported only. A height measured from a base the head is not above is
    meaningless, and a person on the floor is exactly that case."""
    floor_body_min_m: float = 0.55
    """The head must land at least this far from the feet: closer is a crouch."""
    floor_body_max_m: float = 1.75
    """And no further than this. Measured on UR Fall: a person really on the floor
    puts their head 0.99 to 1.69 m from their feet (10th to 90th percentile),
    while someone bending down to pick something up puts it at 1.05 to 2.07 m.
    The bound sits where those two distributions separate."""
    floor_spread_min_m: float = 0.80
    """Reported only; it saturates and does not separate lying from standing."""
    floor_hold_s: float = 0.8
    """The floor state must hold this long before the urgent call goes out."""
    floor_grace_s: float = 0.4
    """How long the floor evidence may lapse without restarting that clock.
    Pose on a person lying on the floor drops out often: on UR Fall only 68 per
    cent of ground-truth lying frames yielded six usable joints. Without a grace
    the hold never completes and the urgent call never goes out."""
    floor_consistency_min: float = 0.85
    """Fraction of keypoints that must back-project onto the floor plane."""

    # --- hazards ------------------------------------------------------------
    corridor_width_m: float = 0.90
    """Width of the walking route the engine keeps clear."""
    clutter_area_cm2: float = 400.0
    """Floor footprint below which an object is not worth calling clutter."""
    aid_reach_m: float = 0.75
    """A walking frame further than this from the seated person is out of reach."""

    # --- view quality -------------------------------------------------------
    dark_mean_v: float = 28.0
    """Mean V channel below which the view is too dark to trust."""
    dark_low_bin_fraction: float = 0.92
    """Fraction of pixels in the bottom 16 V bins that also means too dark."""
    blocked_laplacian_var: float = 12.0
    """Whole-frame Laplacian variance below which the lens is blocked or fogged."""
    camera_moved_px: float = 18.0
    """Median reprojection shift against the reference frame that invalidates zones."""
    absent_s: float = 6.0
    """No usable pose for this long means the person is out of frame."""

    # --- escalation ---------------------------------------------------------
    nudge_to_station_s: float = 20.0
    """An unacknowledged nudge becomes a station call after this long."""
    acknowledge_cooldown_s: float = 90.0
    """After staff acknowledge, the same rung is not raised again for this long."""

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class RoomConfig:
    """Everything about one room. Serialised to JSON and shipped with the footage."""

    room: str = "room"
    bed_name: str = "bed"
    floor: FloorPlane | None = None
    zones: tuple[Zone, ...] = ()
    thresholds: Thresholds = field(default_factory=Thresholds)
    privacy_mode: str = "strict"
    """strict discards every frame after pose extraction; diagnostic keeps them
    for engineering only and is refused by the deployed service."""
    fps_hint: float = 0.0
    """Used when the container reports no frame rate. 0 means trust the file."""
    notes: str = ""

    def zones_of(self, kind: str) -> tuple[Zone, ...]:
        return tuple(z for z in self.zones if z.kind == kind)

    def zone(self, name: str) -> Zone | None:
        return next((z for z in self.zones if z.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "room": self.room,
            "bed_name": self.bed_name,
            "floor": self.floor.to_dict() if self.floor else None,
            "zones": [z.to_dict() for z in self.zones],
            "thresholds": self.thresholds.to_dict(),
            "privacy_mode": self.privacy_mode,
            "fps_hint": self.fps_hint,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoomConfig:
        thresholds = Thresholds(**data.get("thresholds", {})) if data.get("thresholds") else Thresholds()
        return cls(
            room=data.get("room", "room"),
            bed_name=data.get("bed_name", "bed"),
            floor=FloorPlane.from_dict(data["floor"]) if data.get("floor") else None,
            zones=tuple(Zone.from_dict(z) for z in data.get("zones", ())),
            thresholds=thresholds,
            privacy_mode=data.get("privacy_mode", "strict"),
            fps_hint=float(data.get("fps_hint", 0.0)),
            notes=data.get("notes", ""),
        )

    @classmethod
    def load(cls, path: str | Path) -> RoomConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
