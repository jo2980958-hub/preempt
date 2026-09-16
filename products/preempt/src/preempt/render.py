"""Evidence a ward can look at, drawn from numbers and never from a photograph.

Read the signatures. Not one function in this module accepts an image. They take
keypoints, zone polygons, a floor frame and some text, and they draw on a blank
canvas. That is the mechanical reason the privacy claim holds: there is no code
path from a camera frame to a saved file, because the only thing that saves
files will not accept one.

What gets drawn is a room outline with the bed, chair and door where the ward
put them, the person as an open line figure, and the state with its reasons.
Text uses OpenCV 5's `cv2.FontFace`, which renders through a real TrueType engine
instead of the old Hershey strokes, so a ward screenshot does not look like a
1990s machine-vision demo.

Colours are Vellum's, from `docs/design/atlas.md`: a lilac-white ground, deep
indigo for structure, and a single crimson-rose that appears only when a call has
been raised.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Zone
from .geometry import FloorFrame
from .pose import KEYPOINT_NAMES, SKELETON, PoseFrame

GROUND = (251, 246, 247)  # #F7F6FB in BGR
SURFACE = (255, 255, 255)
RAISED = (247, 237, 239)  # #EFEDF7
INK = (46, 27, 28)  # #1C1B2E
INK_DIM = (128, 87, 90)  # #5A5780
INDIGO = (144, 51, 59)  # #3B3390
ROSE = (74, 23, 176)  # #B0174A
OK_GREEN = (110, 140, 60)

ZONE_COLOUR = {
    "bed": INDIGO,
    "chair": INDIGO,
    "door": INK_DIM,
    "wall": INK_DIM,
    "aid": OK_GREEN,
    "exclude": INK_DIM,
}

_SANS = cv2.FontFace("sans")
_ITALIC = cv2.FontFace("italic")


def _text(
    canvas: np.ndarray,
    origin: tuple[int, int],
    message: str,
    colour: tuple[int, int, int],
    size: int = 16,
    italic: bool = False,
) -> int:
    """Draw one line and return the x advance, so callers can lay out a column."""
    face = _ITALIC if italic else _SANS
    advance, _ = cv2.putText(canvas, message, origin, colour, face, size, cv2.FILLED)
    return int(advance[0])


def room_outline(
    width: int,
    height: int,
    frame: FloorFrame,
    zones: tuple[Zone, ...],
    *,
    margin: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """A plan view of the room in floor metres, plus the metres-to-canvas transform.

    A plan rather than the camera's view, deliberately: a plan cannot be mistaken
    for a photograph, and it is the view a ward already has on the wall.
    """
    canvas = np.full((height, width, 3), GROUND, dtype=np.uint8)
    world = np.asarray(frame.plane.world_points, dtype=np.float64)
    lo, hi = world.min(axis=0), world.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    scale = float(min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1]))
    offset = np.array([margin, margin], dtype=np.float64) - lo * scale
    transform = np.array([[scale, 0.0, offset[0]], [0.0, scale, offset[1]]])

    def to_canvas(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return (pts * scale + offset).astype(np.int32)

    cv2.fillPoly(canvas, [to_canvas(world)], SURFACE)
    cv2.polylines(canvas, [to_canvas(world)], True, RAISED, 2, cv2.LINE_AA)

    for zone in zones:
        mapped = frame.to_floor(zone.polygon.astype(np.float64))
        good = mapped[~np.isnan(mapped).any(axis=1)]
        if good.shape[0] < 3:
            continue
        colour = ZONE_COLOUR.get(zone.kind, INK_DIM)
        poly = to_canvas(good)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [poly], colour)
        cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)
        cv2.polylines(canvas, [poly], True, colour, 1, cv2.LINE_AA)
        centre = poly.mean(axis=0).astype(int)
        _text(canvas, (int(centre[0]) - 18, int(centre[1])), zone.name, colour, 13)
    return canvas, transform


def draw_figure(
    canvas: np.ndarray,
    pose: PoseFrame,
    min_score: float,
    *,
    colour: tuple[int, int, int] = INK,
    scale: float = 1.0,
    origin: tuple[int, int] = (0, 0),
) -> None:
    """The person, as an open line figure. Keypoints in, lines out, nothing else."""
    xy = pose.xy * scale + np.asarray(origin, dtype=np.float64)
    seen = pose.scores >= min_score
    for a, b in SKELETON:
        if seen[a] and seen[b]:
            cv2.line(
                canvas,
                tuple(np.round(xy[a]).astype(int)),
                tuple(np.round(xy[b]).astype(int)),
                colour,
                2,
                cv2.LINE_AA,
            )
    for i in range(len(KEYPOINT_NAMES)):
        if seen[i]:
            cv2.circle(canvas, tuple(np.round(xy[i]).astype(int)), 3, colour, -1, cv2.LINE_AA)


def pose_card(
    pose: PoseFrame | None,
    state: str,
    reasons: list[str],
    *,
    width: int = 720,
    height: int = 520,
    alert: bool = False,
    footer: str = "",
    subfooter: str = "",
    source_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """The evidence image: the figure, the state, and the reasons. No photograph.

    `source_size` is only used to scale the stick figure to fit the card; the
    frame it came from is long gone by the time this runs.
    """
    canvas = np.full((height, width, 3), GROUND, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, 64), SURFACE, -1)
    cv2.line(canvas, (0, 64), (width, 64), RAISED, 1, cv2.LINE_AA)
    accent = ROSE if alert else INDIGO
    cv2.rectangle(canvas, (0, 0), (6, 64), accent, -1)
    _text(canvas, (22, 42), state, accent, 26)

    field_top, field_height = 80, height - 200
    cv2.rectangle(canvas, (16, field_top), (width - 16, field_top + field_height), SURFACE, -1)

    if pose is None:
        _text(
            canvas, (32, field_top + field_height // 2), "no pose in this frame", INK_DIM, 16, True
        )
    else:
        seen = pose.scores > 0
        pts = pose.xy[seen]
        if pts.size:
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            span = np.maximum(hi - lo, 1.0)
            scale = float(min((width - 160) / span[0], (field_height - 60) / span[1]))
            origin = (
                int(16 + (width - 32 - span[0] * scale) / 2 - lo[0] * scale),
                int(field_top + (field_height - span[1] * scale) / 2 - lo[1] * scale),
            )
            draw_figure(canvas, pose, 0.30, colour=INK, scale=scale, origin=origin)

    y = field_top + field_height + 26
    for reason in reasons[: 3 if subfooter else 4]:
        cv2.circle(canvas, (26, y - 5), 3, accent, -1, cv2.LINE_AA)
        _text(canvas, (40, y), reason[:96], INK_DIM, 15)
        y += 24
    if footer:
        _text(canvas, (22, height - (34 if subfooter else 14)), footer, INK_DIM, 13, True)
    if subfooter:
        _text(canvas, (22, height - 14), subfooter[:100], INK_DIM, 13, True)
    return canvas


def plan_card(
    frame: FloorFrame,
    zones: tuple[Zone, ...],
    *,
    person_xy: np.ndarray | None = None,
    corridor: list[list[float]] | None = None,
    hazards: list[tuple[float, float]] = (),
    title: str = "Room",
    subtitle: str = "",
    width: int = 720,
    height: int = 520,
) -> np.ndarray:
    """The plan view with the person's floor position and anything in the way."""
    canvas, transform = room_outline(width, height - 48, frame, zones)
    canvas = np.vstack([np.full((48, width, 3), SURFACE, dtype=np.uint8), canvas])
    _text(canvas, (22, 32), title[:60], INK, 20)
    if subtitle:
        band = np.full((28, width, 3), SURFACE, dtype=np.uint8)
        _text(band, (22, 19), subtitle[:100], INK_DIM, 13, True)
        canvas = np.vstack([canvas, band])

    def to_canvas(point) -> tuple[int, int]:
        p = np.asarray(point, dtype=np.float64).reshape(2)
        x = p[0] * transform[0, 0] + transform[0, 2]
        y = p[1] * transform[1, 1] + transform[1, 2]
        return round(x), round(y) + 48

    if corridor:
        pts = np.array([to_canvas(p) for p in corridor], dtype=np.int32)
        cv2.polylines(canvas, [pts], True, INDIGO, 1, cv2.LINE_AA)
    for hazard in hazards:
        cv2.drawMarker(canvas, to_canvas(hazard), ROSE, cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
    if person_xy is not None and not np.isnan(person_xy).any():
        centre = to_canvas(person_xy)
        cv2.circle(canvas, centre, 9, INDIGO, 2, cv2.LINE_AA)
        cv2.circle(canvas, centre, 3, INDIGO, -1, cv2.LINE_AA)
    return canvas


def encode_png(canvas: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("failed to encode the evidence card")
    return buffer.tobytes()
