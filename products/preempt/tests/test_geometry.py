"""Floor-plane geometry, checked against a camera whose answers are known exactly.

The synthetic camera in `preempt.synth` projects with a real pinhole model, so
every height and every floor position in these tests has a closed-form truth
value. If the geometry is wrong, these fail by metres rather than by rounding.
"""

from __future__ import annotations

import numpy as np
import pytest
from preempt.config import FloorPlane, Zone
from preempt.geometry import (
    FloorFrame,
    GeometryError,
    corridor_polygon,
    distance_to_zone_m,
    inside,
    signed_distance_px,
)
from preempt.synth import SynthCamera, default_room

GRID = [(1.0, 1.0), (2.0, 2.0), (3.0, 1.0), (1.5, 2.8), (3.5, 3.0)]
HEIGHTS = [0.15, 0.5, 0.9, 1.4, 1.7]


@pytest.fixture(scope="module")
def frame_and_camera():
    room, camera = default_room()
    return FloorFrame(room.floor), camera


def test_the_setup_points_round_trip_to_the_pixel(frame_and_camera):
    frame, _ = frame_and_camera
    assert frame.residual_px < 0.01


def test_a_floor_point_back_projects_to_where_it_came_from(frame_and_camera):
    frame, camera = frame_and_camera
    for x, y in GRID:
        pixel = camera.project(np.array([[x, y, 0.0]]))[0]
        recovered = frame.to_floor(pixel.reshape(1, 2))[0]
        assert np.allclose(recovered, (x, y), atol=1e-4)


def test_heights_above_the_floor_are_recovered_in_metres(frame_and_camera):
    frame, camera = frame_and_camera
    assert frame.metric_height_available
    for x, y in GRID:
        base = camera.project(np.array([[x, y, 0.0]]))[0]
        for height in HEIGHTS:
            top = camera.project(np.array([[x, y, height]]))[0]
            estimate = frame.height_above_floor(tuple(base), tuple(top))
            assert estimate.ok
            assert abs(estimate.metres - height) < 0.01, (
                f"{estimate.metres:.3f} m recovered for a true {height} m"
            )


def test_the_floor_shadow_of_an_elevated_point_is_exact(frame_and_camera):
    frame, camera = frame_and_camera
    for x, y in GRID:
        for height in HEIGHTS:
            pixel = camera.project(np.array([[x, y, height]]))[0]
            shadow = frame.floor_shadow(pixel, height)
            assert shadow is not None
            assert np.allclose(shadow, (x, y), atol=1e-3)


def test_up_is_the_projective_vertical_not_the_image_vertical(frame_and_camera):
    frame, camera = frame_and_camera
    for x, y in GRID:
        base = camera.project(np.array([[x, y, 0.0]]))[0]
        top = camera.project(np.array([[x, y, 1.0]]))[0]
        truth = (top - base) / np.linalg.norm(top - base)
        direction, source = frame.up_direction(top, floor_xy=np.array([x, y]))
        assert source == "vertical vanishing point"
        assert np.allclose(direction, truth, atol=1e-3)
    # And the difference from the naive answer is large enough to matter. Near
    # the middle of the frame it is under two degrees, which is why taking the
    # image y axis for 'up' looks fine until someone walks into a corner: across
    # this room it reaches about 30 degrees, and a trunk angle wrong by 30 degrees
    # is a posture read wrong.
    worst = 0.0
    naive = np.array([0.0, -1.0])
    for x in np.linspace(0.4, 3.6, 9):
        for y in np.linspace(0.4, 3.2, 9):
            base = camera.project(np.array([[x, y, 0.0]]))[0]
            top = camera.project(np.array([[x, y, 1.0]]))[0]
            truth = (top - base) / np.linalg.norm(top - base)
            worst = max(worst, np.degrees(np.arccos(abs(float(np.dot(truth, naive))))))
    assert worst > 20.0, f"only {worst:.1f} deg apart; the correction would be moot"


def test_a_point_above_the_horizon_is_refused_not_guessed(frame_and_camera):
    frame, _ = frame_and_camera
    far_above = np.array([[640.0, -400.0]])
    assert np.isnan(frame.to_floor(far_above)).all()


def test_without_a_vertical_reference_the_height_is_refused_with_a_reason():
    room, _camera = default_room()
    plane = FloorPlane(
        image_points=room.floor.image_points,
        world_points=room.floor.world_points,
    )
    frame = FloorFrame(plane)
    assert not frame.metric_height_available
    estimate = frame.height_above_floor((100.0, 500.0), (100.0, 300.0))
    assert not estimate.ok
    assert "vertical reference" in estimate.reason
    assert frame.up_direction((100.0, 300.0))[1] == "image"


def test_a_degenerate_room_setup_raises_rather_than_producing_nonsense():
    plane = FloorPlane(
        image_points=((0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)),
        world_points=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)),
    )
    with pytest.raises(GeometryError):
        FloorFrame(plane)


def test_zone_membership_and_distance(frame_and_camera):
    frame, camera = frame_and_camera
    corners = np.array([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0], [1.0, 2.0, 0.0]])
    pixels = camera.project(corners)
    zone = Zone("test bed", "bed", tuple((float(x), float(y)) for x, y in pixels))
    centre = camera.project(np.array([[1.5, 1.5, 0.0]]))[0]
    outside = camera.project(np.array([[3.5, 3.0, 0.0]]))[0]
    assert inside(zone, tuple(centre))
    assert not inside(zone, tuple(outside))
    assert signed_distance_px(zone, tuple(centre)) > 0
    assert signed_distance_px(zone, tuple(outside)) < 0

    distance = distance_to_zone_m(frame, zone, np.array([3.0, 1.5]))
    assert distance is not None
    assert 0.9 < distance < 1.1, f"a point 1 m outside measured {distance:.2f} m"


def test_the_walking_corridor_is_the_right_width_and_length():
    corridor = corridor_polygon(np.array([0.0, 0.0]), np.array([3.0, 0.0]), 0.9)
    assert corridor is not None
    ys = corridor[:, 1]
    assert abs(float(ys.max() - ys.min()) - 0.9) < 1e-6
    xs = corridor[:, 0]
    assert abs(float(xs.max() - xs.min()) - 3.0) < 1e-6
    assert corridor_polygon(np.array([1.0, 1.0]), np.array([1.0, 1.0]), 0.9) is None


def test_a_camera_that_looks_straight_down_still_works():
    """A ceiling camera is a real deployment, and it is the degenerate case."""
    camera = SynthCamera(position=(2.0, 1.8, 2.8), target=(2.0, 1.81, 0.0))
    corners = np.array([[0.4, 0.4], [3.6, 0.4], [3.6, 3.2], [0.4, 3.2]])
    frame = FloorFrame(camera.floor_plane(corners))
    base = camera.project(np.array([[2.0, 2.0, 0.0]]))[0]
    top = camera.project(np.array([[2.0, 2.0, 1.2]]))[0]
    estimate = frame.height_above_floor(tuple(base), tuple(top))
    assert estimate.ok
    assert abs(estimate.metres - 1.2) < 0.05
