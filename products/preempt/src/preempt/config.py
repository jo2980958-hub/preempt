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
            "reference_base_px": (list(self.reference_base_px) if self.reference_base_px else None),
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


CALIBRATION_SOURCES = ("measured", "assumed", "synthetic", "unstated")


@dataclass(frozen=True)
class Calibration:
    """Where this room's camera height and focal length came from.

    Every metre Preempt reports rests on these two numbers, so the room says how
    they were obtained and the result carries that through to the evidence card.
    `measured` means somebody measured them in the room, `assumed` means they
    were worked out from something else and could be wrong (the CDC rooms took
    the clinician to be 1.65 m tall), `synthetic` means the camera generated the
    scene so they are exact, and `unstated` is what a room that says nothing
    gets. The interface treats `unstated` as `assumed`, never as `measured`.
    """

    camera_height: str = "unstated"
    focal_length: str = "unstated"
    note: str = ""

    @property
    def calibrated(self) -> bool:
        """Only a measured or exactly known camera counts as calibrated."""
        good = ("measured", "synthetic")
        return self.camera_height in good and self.focal_length in good

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_height": self.camera_height,
            "focal_length": self.focal_length,
            "note": self.note,
            "calibrated": self.calibrated,
        }


class RoomError(ValueError):
    """A room setup that cannot be used, naming the field that is wrong."""

    def __init__(self, field_name: str, problem: str) -> None:
        super().__init__(f"{field_name}: {problem}")
        self.field = field_name
        self.problem = problem


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
    calibration: Calibration = field(default_factory=Calibration)

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
            "calibration": {
                k: v for k, v in self.calibration.to_dict().items() if k != "calibrated"
            },
        }

    @classmethod
    def from_dict(cls, data: Any) -> RoomConfig:
        """Build a room from its JSON form, refusing anything malformed by name.

        The command line, the pose-track loader and the upload API all come
        through here, so a room the CLI accepts is exactly a room the service
        accepts. Every refusal is a `RoomError` naming the field.
        """
        return parse_room(data)

    @classmethod
    def load(cls, path: str | Path) -> RoomConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_ROOM_KEYS = {
    "room",
    "bed_name",
    "floor",
    "zones",
    "thresholds",
    "privacy_mode",
    "fps_hint",
    "notes",
    "calibration",
}
_FLOOR_KEYS = {
    "image_points",
    "world_points",
    "vertical_vanishing_point",
    "reference_height_m",
    "reference_base_px",
    "reference_top_px",
}


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RoomError(where, f"must be a number, got {type(value).__name__}")
    number = float(value)
    if not np.isfinite(number):
        raise RoomError(where, "must be a finite number")
    return number


def _point(value: Any, where: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RoomError(where, "must be an [x, y] pair")
    return _number(value[0], f"{where}[0]"), _number(value[1], f"{where}[1]")


def _points(value: Any, where: str, *, exactly: int | None = None, at_least: int = 0) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise RoomError(where, "must be a list of [x, y] points")
    if exactly is not None and len(value) != exactly:
        raise RoomError(where, f"needs exactly {exactly} points, got {len(value)}")
    if len(value) < at_least:
        raise RoomError(where, f"needs at least {at_least} points, got {len(value)}")
    return tuple(_point(p, f"{where}[{i}]") for i, p in enumerate(value))


def _text_field(data: dict[str, Any], key: str, default: str, where: str = "") -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise RoomError(f"{where}{key}", f"must be text, got {type(value).__name__}")
    return value


def _unknown(data: dict[str, Any], known: set[str], where: str) -> None:
    extra = sorted(set(data) - known)
    if extra:
        raise RoomError(f"{where}{extra[0]}", f"is not a room setup field; known: {sorted(known)}")


def parse_room(data: Any) -> RoomConfig:
    """Validate a room setup's JSON form and build it. Raises `RoomError`."""
    if not isinstance(data, dict):
        raise RoomError("room", "a room setup must be a JSON object")
    _unknown(data, _ROOM_KEYS, "")

    floor_data = data.get("floor")
    if floor_data is None:
        raise RoomError(
            "floor",
            "is required: mark four floor points so anything can be measured in metres",
        )
    if not isinstance(floor_data, dict):
        raise RoomError("floor", "must be an object")
    _unknown(floor_data, _FLOOR_KEYS, "floor.")
    for key in ("image_points", "world_points"):
        if key not in floor_data:
            raise RoomError(f"floor.{key}", "is required")
    image_points = _points(floor_data["image_points"], "floor.image_points", exactly=4)
    world_points = _points(floor_data["world_points"], "floor.world_points", exactly=4)
    if polygon_area(world_points) < 1e-3:
        raise RoomError("floor.world_points", "the four points must enclose an area of floor")

    def optional_point(key: str) -> tuple[float, float] | None:
        value = floor_data.get(key)
        return None if value is None else _point(value, f"floor.{key}")

    reference = _number(floor_data.get("reference_height_m", 1.70), "floor.reference_height_m")
    if not 0.2 <= reference <= 5.0:
        raise RoomError(
            "floor.reference_height_m", f"{reference} m is not a plausible reference height"
        )
    floor = FloorPlane(
        image_points=image_points,
        world_points=world_points,
        vertical_vanishing_point=optional_point("vertical_vanishing_point"),
        reference_height_m=reference,
        reference_base_px=optional_point("reference_base_px"),
        reference_top_px=optional_point("reference_top_px"),
    )

    zones_data = data.get("zones", [])
    if not isinstance(zones_data, (list, tuple)):
        raise RoomError("zones", "must be a list")
    zones = []
    for i, zone in enumerate(zones_data):
        where = f"zones[{i}]"
        if not isinstance(zone, dict):
            raise RoomError(where, "must be an object with name, kind and points")
        _unknown(zone, {"name", "kind", "points"}, f"{where}.")
        name = _text_field(zone, "name", "", f"{where}.")
        if not name:
            raise RoomError(f"{where}.name", "is required")
        kind = zone.get("kind")
        if kind not in ZONE_KINDS:
            raise RoomError(
                f"{where}.kind", f"{kind!r} is not a zone kind; known: {list(ZONE_KINDS)}"
            )
        points = _points(zone.get("points"), f"{where}.points", at_least=3)
        zones.append(Zone(name=name, kind=kind, points=points))

    thresholds_data = data.get("thresholds") or {}
    if not isinstance(thresholds_data, dict):
        raise RoomError("thresholds", "must be an object")
    known = set(Thresholds.__dataclass_fields__)
    values: dict[str, Any] = {}
    for key, value in thresholds_data.items():
        if key not in known:
            raise RoomError(f"thresholds.{key}", "is not a threshold Preempt knows")
        number = _number(value, f"thresholds.{key}")
        default = getattr(Thresholds, key)
        values[key] = (
            int(number) if isinstance(default, int) and not isinstance(default, bool) else number
        )

    privacy_mode = _text_field(data, "privacy_mode", "strict")
    if privacy_mode not in ("strict", "diagnostic"):
        raise RoomError("privacy_mode", f"must be strict or diagnostic, not {privacy_mode!r}")

    calibration_data = data.get("calibration") or {}
    if not isinstance(calibration_data, dict):
        raise RoomError("calibration", "must be an object")
    _unknown(
        calibration_data, {"camera_height", "focal_length", "note", "calibrated"}, "calibration."
    )
    sources = {}
    for key in ("camera_height", "focal_length"):
        value = calibration_data.get(key, "unstated")
        if value not in CALIBRATION_SOURCES:
            raise RoomError(
                f"calibration.{key}", f"{value!r} is not one of {list(CALIBRATION_SOURCES)}"
            )
        sources[key] = value

    fps_hint = _number(data.get("fps_hint", 0.0), "fps_hint")
    if fps_hint < 0:
        raise RoomError("fps_hint", "must not be negative")

    return RoomConfig(
        room=_text_field(data, "room", "room") or "room",
        bed_name=_text_field(data, "bed_name", "bed"),
        floor=floor,
        zones=tuple(zones),
        thresholds=Thresholds(**values),
        privacy_mode=privacy_mode,
        fps_hint=fps_hint,
        notes=_text_field(data, "notes", ""),
        calibration=Calibration(
            camera_height=sources["camera_height"],
            focal_length=sources["focal_length"],
            note=_text_field(calibration_data, "note", "", "calibration."),
        ),
    )


def polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    """Unsigned polygon area, by the shoelace formula."""
    xy = np.asarray(points, dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)
