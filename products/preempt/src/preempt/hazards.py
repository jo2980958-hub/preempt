"""What is in the way, and what is not within reach.

Three hazards, chosen because each one is both common in the incident reports and
genuinely visible from a ceiling corner:

1. **The walking frame out of reach.** The commonest preventable mechanism on a
   ward: the aid is parked where the porter left it, the patient decides not to
   wait, and the first three steps are unsupported. Measured as the distance in
   metres from the person's seat to the aid's registered zone.

2. **Clutter on the route.** A visitor's bag, a commode, a cable. Found by
   differencing the current frame against the room as it was when the ward
   signed off the setup, then keeping only components whose floor footprint sits
   inside the walking corridor and is big enough to trip over.

3. **A wet-floor sign.** The sign means the floor is wet, which is the hazard;
   detecting the sign is easier and more reliable than detecting the water, and
   it is what the ward already puts out. Yellow in HSV, an A-frame's shape, and
   a position on the floor inside the corridor.

All three are classical OpenCV: `absdiff`, `threshold`, `morphologyEx`,
`connectedComponentsWithStats`, `inRange`, `findContours`, `approxPolyDP`,
`pointPolygonTest`. No model, nothing learned, nothing that needs a GPU, and
every number traceable to a pixel count and a homography.

Privacy note: the reference image of the empty room is held in memory for the
run. It contains no person by construction (it is captured at setup, before
anyone is in the room) and it is never written out. The hazard results are
positions and areas, not crops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .config import Thresholds, Zone
from .geometry import FloorFrame, corridor_polygon, polygon_centroid, zone_floor_polygon

CLUTTER = "clutter on the route"
AID_OUT_OF_REACH = "walking frame out of reach"
WET_FLOOR = "wet floor sign on the route"

# A wet-floor sign is a saturated yellow; the band is wide enough for tungsten
# ward lighting and narrow enough to exclude skin, wood and beige linen.
WET_FLOOR_HSV_LOW = np.array([20, 110, 110], dtype=np.uint8)
WET_FLOOR_HSV_HIGH = np.array([35, 255, 255], dtype=np.uint8)


@dataclass
class Hazard:
    """One thing in the way, with where it is and how sure the engine is."""

    kind: str
    description: str
    floor_xy: tuple[float, float] | None = None
    metres: float | None = None
    area_cm2: float | None = None
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "description": self.description,
            "floor_xy": (
                None if self.floor_xy is None else [round(self.floor_xy[0], 2), round(self.floor_xy[1], 2)]
            ),
            "metres": None if self.metres is None else round(self.metres, 2),
            "area_cm2": None if self.area_cm2 is None else round(self.area_cm2, 0),
            "confidence": round(self.confidence, 2),
        }


@dataclass
class HazardReport:
    hazards: list[Hazard] = field(default_factory=list)
    corridor: list[list[float]] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def any(self) -> bool:
        return bool(self.hazards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hazards": [h.to_dict() for h in self.hazards],
            "corridor": self.corridor,
            "checked": list(self.checked),
            "skipped": list(self.skipped),
        }


def floor_area_cm2(frame: FloorFrame, contour: np.ndarray) -> float | None:
    """Real floor area of an image contour, by mapping its convex hull to metres."""
    hull = cv2.convexHull(contour.astype(np.float32)).reshape(-1, 2)
    mapped = frame.to_floor(hull.astype(np.float64))
    good = mapped[~np.isnan(mapped).any(axis=1)]
    if good.shape[0] < 3:
        return None
    return float(abs(cv2.contourArea(good.astype(np.float32))) * 10_000.0)


def _in_corridor(corridor: np.ndarray | None, point: np.ndarray) -> bool:
    if corridor is None or np.isnan(point).any():
        return False
    return (
        cv2.pointPolygonTest(
            corridor.astype(np.float32), (float(point[0]), float(point[1])), False
        )
        >= 0
    )


def find_clutter(
    current: np.ndarray,
    reference: np.ndarray,
    frame: FloorFrame,
    corridor: np.ndarray | None,
    thresholds: Thresholds,
    *,
    person_box: tuple[float, float, float, float] | None = None,
) -> list[Hazard]:
    """Objects present now that were not there when the ward signed off the room."""
    if current.shape != reference.shape:
        reference = cv2.resize(reference, (current.shape[1], current.shape[0]))
    a = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY) if current.ndim == 3 else current
    b = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
    a = cv2.GaussianBlur(a, (5, 5), 0)
    b = cv2.GaussianBlur(b, (5, 5), 0)
    diff = cv2.absdiff(a, b)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if person_box is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in person_box)
        pad = 24
        mask[max(0, y1 - pad) : y2 + pad, max(0, x1 - pad) : x2 + pad] = 0

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    hazards: list[Hazard] = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < 150:
            continue
        component = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        # Something on the floor touches the floor at its lowest point.
        lowest = contour.reshape(-1, 2)[np.argmax(contour.reshape(-1, 2)[:, 1])]
        foot = frame.to_floor(np.asarray(lowest, dtype=np.float64).reshape(1, 2))[0]
        if not _in_corridor(corridor, foot):
            continue
        area = floor_area_cm2(frame, contour.reshape(-1, 2))
        if area is None or area < thresholds.clutter_area_cm2:
            continue
        hazards.append(
            Hazard(
                kind=CLUTTER,
                description=(
                    f"something about {area / 10_000:.2f} square metres across is on the "
                    "route between the bed and the door, and was not there at setup"
                ),
                floor_xy=(float(foot[0]), float(foot[1])),
                area_cm2=area,
                confidence=0.6,
            )
        )
    return hazards


def find_wet_floor_sign(
    image: np.ndarray, frame: FloorFrame, corridor: np.ndarray | None
) -> list[Hazard]:
    """A yellow A-frame standing on the route."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WET_FLOOR_HSV_LOW, WET_FLOOR_HSV_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hazards: list[Hazard] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 400:
            continue
        hull = cv2.convexHull(contour)
        solidity = area / max(1.0, cv2.contourArea(hull))
        approx = cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True)
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(1.0, h)
        # An A-frame sign: taller than wide, convex, three to six hull corners.
        if not (0.35 <= aspect <= 1.25 and solidity > 0.7 and 3 <= len(approx) <= 6):
            continue
        base = frame.to_floor(np.array([[x + w / 2.0, y + h]], dtype=np.float64))[0]
        if not _in_corridor(corridor, base):
            continue
        hazards.append(
            Hazard(
                kind=WET_FLOOR,
                description="a wet floor sign is standing on the route between the bed and the door",
                floor_xy=(float(base[0]), float(base[1])),
                confidence=float(min(0.9, 0.5 + solidity / 3.0)),
            )
        )
    return hazards


def check_aid_reach(
    frame: FloorFrame,
    aid_zones: tuple[Zone, ...],
    seat_floor_xy: np.ndarray | None,
    thresholds: Thresholds,
) -> list[Hazard]:
    """Is the walking frame close enough to be picked up from where the person is sitting?"""
    if not aid_zones or seat_floor_xy is None or np.isnan(seat_floor_xy).any():
        return []
    hazards: list[Hazard] = []
    for zone in aid_zones:
        poly = zone_floor_polygon(frame, zone)
        if poly.shape[0] < 3:
            continue
        distance = float(
            -cv2.pointPolygonTest(
                poly.astype(np.float32),
                (float(seat_floor_xy[0]), float(seat_floor_xy[1])),
                True,
            )
        )
        if distance > thresholds.aid_reach_m:
            centre = polygon_centroid(poly)
            hazards.append(
                Hazard(
                    kind=AID_OUT_OF_REACH,
                    description=(
                        f"the {zone.name} is {distance:.1f} m away, further than the "
                        f"{thresholds.aid_reach_m:.2f} m a seated person can reach"
                    ),
                    floor_xy=(float(centre[0]), float(centre[1])),
                    metres=distance,
                    confidence=0.85,
                )
            )
    return hazards


class HazardScanner:
    """Runs the three checks, on a schedule rather than every frame.

    Clutter and signs do not move between one frame and the next, and differencing
    a 1080p frame costs more than the pose does, so the scan runs every
    `interval_s` and the last report stands in between.
    """

    def __init__(
        self,
        frame: FloorFrame,
        thresholds: Thresholds,
        *,
        bed_zones: tuple[Zone, ...] = (),
        door_zones: tuple[Zone, ...] = (),
        aid_zones: tuple[Zone, ...] = (),
        interval_s: float = 2.0,
    ) -> None:
        self.frame = frame
        self.t = thresholds
        self.bed_zones = bed_zones
        self.door_zones = door_zones
        self.aid_zones = aid_zones
        self.interval_s = interval_s
        self.reference: np.ndarray | None = None
        self.last: HazardReport = HazardReport(skipped=["no scan has run yet"])
        self._last_scan_s: float | None = None

    def corridor(self) -> np.ndarray | None:
        """The route the person will walk: bed to door, in floor metres."""
        if not self.bed_zones or not self.door_zones:
            return None
        bed = zone_floor_polygon(self.frame, self.bed_zones[0])
        door = zone_floor_polygon(self.frame, self.door_zones[0])
        if bed.shape[0] < 3 or door.shape[0] < 3:
            return None
        return corridor_polygon(
            polygon_centroid(bed), polygon_centroid(door), self.t.corridor_width_m
        )

    def set_reference(self, image: np.ndarray) -> None:
        """The room as the ward signed it off. Held in memory, never written out."""
        self.reference = image.copy()

    def scan(
        self,
        image: np.ndarray,
        time_s: float,
        *,
        seat_floor_xy: np.ndarray | None = None,
        person_box: tuple[float, float, float, float] | None = None,
        force: bool = False,
    ) -> HazardReport:
        if not force and self._last_scan_s is not None:
            if time_s - self._last_scan_s < self.interval_s:
                return self.last
        self._last_scan_s = time_s

        corridor = self.corridor()
        report = HazardReport()
        report.corridor = [] if corridor is None else [[round(float(x), 2), round(float(y), 2)] for x, y in corridor]

        if corridor is None:
            report.skipped.append(
                "no walking route could be drawn: the room needs a bed zone and a door zone"
            )
        else:
            report.checked.append("clutter on the route")
            if self.reference is None:
                report.skipped.append(
                    "no reference photograph of the empty room, so clutter cannot be told "
                    "from furniture"
                )
            else:
                report.hazards.extend(
                    find_clutter(
                        image, self.reference, self.frame, corridor, self.t,
                        person_box=person_box,
                    )
                )
            report.checked.append("wet floor sign")
            report.hazards.extend(find_wet_floor_sign(image, self.frame, corridor))

        if self.aid_zones:
            report.checked.append("walking frame within reach")
            report.hazards.extend(
                check_aid_reach(self.frame, self.aid_zones, seat_floor_xy, self.t)
            )
        else:
            report.skipped.append("no walking frame was registered for this room")

        self.last = report
        return report
