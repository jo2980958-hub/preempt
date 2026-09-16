"""Floor-plane geometry, so "on the floor" is measured rather than guessed.

The whole product turns on one idea: a single homography between the image and
the floor gives you a metric frame for free, and everything interesting about a
person about to fall is a distance or a height in that frame.

Three things this module computes, and what each is worth:

**Back-projection.** `H` maps floor metres to pixels; `H^-1` maps a pixel back to
floor metres. That is exact for any point that really is on the floor and
meaningless for any point that is not, which is the property we exploit: a
standing person's head does not back-project anywhere sensible, and a person
lying on the floor's head does. The measure is `floor_consistency`.

**The horizon.** The vanishing line of the floor plane is `l = H^-T · (0,0,1)^T`,
exactly, with no extra calibration. A point imaged above that line cannot be on
the floor at all, so back-projection there is not just wrong but impossible, and
the engine says so instead of returning a number.

**Height, and the floor position of a point that is not on the floor.** With a
vertical vanishing point and one object of known height, the vanishing point can
be rescaled so that one unit of it is one metre. After that, any world point
satisfies `p ~ H·b + h·vz_h`, which gives a height when the floor position is
known and a floor position when the height is known. Neither alone is solvable
from one view. That is a real limitation and it is not worked around here: a
point of unknown height is placed by assuming it is above the person's foot
contact, which is true to within a few centimetres for a hip and false for an
outstretched hand, and the code says which of the two it is doing. Everything
that needs to be exact -- the feet, the zones, "on the floor" -- uses only the
homography, which needs no vertical at all.

OpenCV 5 does the work: `findHomography`, `perspectiveTransform`,
`pointPolygonTest`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import FloorPlane, Zone


class GeometryError(ValueError):
    """The room setup is not usable: degenerate quad, singular homography."""


@dataclass(frozen=True)
class HeightEstimate:
    """A height above the floor, or an explicit refusal to state one."""

    metres: float | None
    method: str
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.metres is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metres": None if self.metres is None else round(self.metres, 3),
            "method": self.method,
            "reason": self.reason,
        }


class FloorFrame:
    """The metric frame for one camera in one room.

    Construct once per run. It is immutable in use, so it is safe to share
    between the feature extractor, the hazard pass and the renderer.
    """

    def __init__(self, plane: FloorPlane) -> None:
        image = np.asarray(plane.image_points, dtype=np.float64)
        world = np.asarray(plane.world_points, dtype=np.float64)
        h, _ = cv2.findHomography(world, image, method=0)
        if h is None or not np.all(np.isfinite(h)) or abs(float(np.linalg.det(h))) < 1e-12:
            raise GeometryError(
                "the four floor points do not define a usable plane; "
                "they may be collinear or clicked out of order"
            )
        self.plane = plane
        self.world_to_image = h
        self.image_to_world = np.linalg.inv(h)
        # Vanishing line of the floor: the image of the plane's line at infinity.
        horizon = np.linalg.inv(h).T @ np.array([0.0, 0.0, 1.0])
        norm = math.hypot(float(horizon[0]), float(horizon[1]))
        self.horizon = horizon / norm if norm > 1e-12 else horizon
        self.residual_px = self._residual(world, image)
        self._vz = (
            np.array([*plane.vertical_vanishing_point, 1.0], dtype=np.float64)
            if plane.vertical_vanishing_point
            else None
        )
        self._vz_h = self._calibrate_vertical()

    # -- setup quality ------------------------------------------------------
    def _residual(self, world: np.ndarray, image: np.ndarray) -> float:
        projected = self.to_image(world)
        return float(np.sqrt(np.mean(np.sum((projected - image) ** 2, axis=1))))

    def _calibrate_vertical(self) -> np.ndarray | None:
        """The vertical vanishing point, scaled so that one unit of it is one metre.

        A projective camera satisfies, for a world point one metre above the floor
        point `b`,

            p ~ H·b~ + h·vz_h

        where `vz_h` is the *unnormalised* homogeneous vanishing point of the
        vertical direction. Its scale is what carries the metre; the Euclidean
        vanishing point a ward can click has lost it.

        One reference object of known height puts it back. With base `b_ref` on
        the floor and top `t_ref` at `H_ref` metres, and with vz_h = s·(vx,vy,1)
        for the unknown s,

            mu·t_ref = H·b_ref~ + H_ref·s·(vx,vy,1)

        which is three equations in the two unknowns (mu, s), solved in least
        squares. Recovering the scale this way, rather than through the usual
        cross-ratio, means the same quantity then gives *both* a height and a
        floor position, which is what `resolve` needs.
        """
        plane = self.plane
        if self._vz is None or not plane.reference_base_px or not plane.reference_top_px:
            return None
        base = np.array([*plane.reference_base_px, 1.0], dtype=np.float64)
        top = np.array([*plane.reference_top_px, 1.0], dtype=np.float64)
        floor_point = self.image_to_world @ base
        if abs(floor_point[2]) < 1e-12:
            return None
        rhs = self.world_to_image @ (floor_point / floor_point[2])
        a = np.column_stack([top, -plane.reference_height_m * self._vz])
        solution, *_ = np.linalg.lstsq(a, rhs, rcond=None)
        scale = float(solution[1])
        if not np.isfinite(scale) or abs(scale) < 1e-12:
            return None
        return scale * self._vz

    # -- the projective vertical -------------------------------------------
    @property
    def metric_height_available(self) -> bool:
        return self._vz_h is not None

    def point_at(self, floor_xy: np.ndarray | tuple[float, float], height_m: float) -> np.ndarray:
        """Where a point `height_m` above floor position `floor_xy` lands in the image."""
        if self._vz_h is None:
            raise GeometryError("this camera has no metric vertical; heights are unavailable")
        b = np.array([float(floor_xy[0]), float(floor_xy[1]), 1.0])
        p = self.world_to_image @ b + height_m * self._vz_h
        if abs(p[2]) < 1e-12:
            raise GeometryError("the requested point is on the horizon")
        return p[:2] / p[2]

    def floor_shadow(self, image_point, height_m: float) -> np.ndarray | None:
        """The floor position directly below an image point of known height."""
        if self._vz_h is None:
            return None
        p = np.array([float(image_point[0]), float(image_point[1]), 1.0])
        q = self.image_to_world @ p
        r = self.image_to_world @ self._vz_h
        if abs(q[2]) < 1e-12:
            return None
        mu = (1.0 + height_m * r[2]) / q[2]
        b = mu * q - height_m * r
        return b[:2]

    def _height_from_base(self, base_px, top_px) -> float | None:
        """Metres between a floor point and a point vertically above it."""
        if self._vz_h is None:
            return None
        b = np.array([float(base_px[0]), float(base_px[1]), 1.0])
        t = np.array([float(top_px[0]), float(top_px[1]), 1.0])
        floor_point = self.image_to_world @ b
        if abs(floor_point[2]) < 1e-12:
            return None
        rhs = self.world_to_image @ (floor_point / floor_point[2])
        a = np.column_stack([t, -self._vz_h])
        solution, *_ = np.linalg.lstsq(a, rhs, rcond=None)
        height = float(solution[1])
        return height if np.isfinite(height) else None

    # -- image <-> floor ----------------------------------------------------
    def to_image(self, world_points: np.ndarray) -> np.ndarray:
        pts = np.asarray(world_points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.world_to_image).reshape(-1, 2)

    def to_floor(self, image_points: np.ndarray) -> np.ndarray:
        """Back-project pixels to floor metres. NaN where the point is above the horizon."""
        pts = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        out = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
        above = self.above_horizon(pts)
        usable = ~above
        if np.any(usable):
            mapped = cv2.perspectiveTransform(
                pts[usable].reshape(-1, 1, 2), self.image_to_world
            ).reshape(-1, 2)
            out[usable] = mapped
        return out

    def above_horizon(self, image_points: np.ndarray, margin_px: float = 2.0) -> np.ndarray:
        """True where a pixel lies on the far side of the floor's vanishing line.

        The sign convention is fixed by the setup points, which are on the floor
        by construction, so 'the side the floor is on' is whatever sign they give.
        """
        pts = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        homog = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        signed = homog @ self.horizon
        setup = np.concatenate(
            [np.asarray(self.plane.image_points, dtype=np.float64), np.ones((4, 1))], axis=1
        )
        floor_side = float(np.sign(np.mean(setup @ self.horizon))) or 1.0
        return (signed * floor_side) < margin_px

    # -- heights ------------------------------------------------------------
    def height_above_floor(self, base_px, top_px) -> HeightEstimate:
        """Metric height of a vertical segment whose foot is on the floor."""
        if self._vz_h is None:
            return HeightEstimate(
                None,
                "vertical-vanishing-point",
                "this camera has no vertical reference, so no height in metres can be "
                "stated; mark a vertical edge of known height during setup",
            )
        height = self._height_from_base(base_px, top_px)
        if height is None:
            return HeightEstimate(
                None, "vertical-vanishing-point", "the segment is degenerate against the horizon"
            )
        return HeightEstimate(abs(height), "vertical-vanishing-point")

    def up_direction(self, image_point, floor_xy=None) -> tuple[np.ndarray, str]:
        """The image direction of 'one metre further up' at a point, and its source.

        Computed rather than assumed: a camera in the corner of a room images
        vertical lines as converging, so 'up' is different in each part of the
        frame and taking the image y axis for it puts several degrees of error
        into every trunk angle. Falls back to the image vertical, and says so,
        when the camera was never given a vertical reference.
        """
        if self._vz_h is None:
            return np.array([0.0, -1.0]), "image"
        seed = floor_xy
        if seed is None:
            resolved = self.resolve(image_point)
            if resolved is None:
                return np.array([0.0, -1.0]), "image"
            seed, height = resolved
        else:
            height = self._height_from_base(self.point_at(seed, 0.0), image_point) or 0.0
        try:
            here = self.point_at(seed, height)
            above = self.point_at(seed, height + 0.10)
        except GeometryError:
            return np.array([0.0, -1.0]), "image"
        direction = above - here
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return np.array([0.0, -1.0]), "image"
        return direction / norm, "vertical vanishing point"

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "residual_px": round(self.residual_px, 3),
            "horizon": [round(float(v), 6) for v in self.horizon],
            "metric_height_available": self._vz_h is not None,
            "world_extent_m": [
                round(float(np.ptp(np.asarray(self.plane.world_points)[:, 0])), 3),
                round(float(np.ptp(np.asarray(self.plane.world_points)[:, 1])), 3),
            ],
        }


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------


def inside(zone: Zone, point: tuple[float, float]) -> bool:
    """Is an image-space point inside the polygon a nurse drew?"""
    return cv2.pointPolygonTest(zone.polygon, (float(point[0]), float(point[1])), False) >= 0


def signed_distance_px(zone: Zone, point: tuple[float, float]) -> float:
    """Pixels from the polygon edge. Positive inside, negative outside."""
    return float(cv2.pointPolygonTest(zone.polygon, (float(point[0]), float(point[1])), True))


def zone_floor_polygon(frame: FloorFrame, zone: Zone) -> np.ndarray:
    """A zone's polygon in floor metres, dropping vertices above the horizon."""
    mapped = frame.to_floor(zone.polygon.astype(np.float64))
    return mapped[~np.isnan(mapped).any(axis=1)]


def distance_to_zone_m(frame: FloorFrame, zone: Zone, floor_point: np.ndarray) -> float | None:
    """Metres from a floor position to a zone's floor polygon. None if unmappable."""
    poly = zone_floor_polygon(frame, zone)
    if poly.shape[0] < 3 or np.isnan(floor_point).any():
        return None
    d = cv2.pointPolygonTest(
        poly.astype(np.float32), (float(floor_point[0]), float(floor_point[1])), True
    )
    return float(-d)  # positive when outside, which is the way a nurse would say it


def corridor_polygon(start: np.ndarray, end: np.ndarray, width_m: float) -> np.ndarray | None:
    """The walking route between two floor points, as a rectangle in floor metres."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    if np.isnan(start).any() or np.isnan(end).any():
        return None
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return None
    unit = direction / length
    normal = np.array([-unit[1], unit[0]]) * (width_m / 2.0)
    return np.array([start + normal, end + normal, end - normal, start - normal])


def polygon_centroid(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return pts.mean(axis=0)
