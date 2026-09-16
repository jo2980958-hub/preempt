"""Hazards on the route: a bag, a wet floor sign, a walking frame out of reach."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from preempt.config import Thresholds, Zone
from preempt.geometry import FloorFrame, corridor_polygon, polygon_centroid, zone_floor_polygon
from preempt.hazards import (
    AID_OUT_OF_REACH,
    CLUTTER,
    WET_FLOOR,
    HazardScanner,
    check_aid_reach,
    find_clutter,
    find_wet_floor_sign,
    floor_area_cm2,
)
from preempt.synth import default_room


@pytest.fixture(scope="module")
def room_and_camera():
    return default_room()


@pytest.fixture
def scanner(room_and_camera):
    room, _ = room_and_camera
    return HazardScanner(
        FloorFrame(room.floor),
        room.thresholds,
        bed_zones=room.zones_of("bed"),
        door_zones=room.zones_of("door"),
        aid_zones=room.zones_of("aid"),
    )


def empty_room(camera) -> np.ndarray:
    rng = np.random.default_rng(11)
    frame = rng.integers(100, 140, (720, 1280, 3), dtype=np.uint8)
    return frame


def place_box(camera, frame, x, y, size=0.45, height=0.35, colour=(40, 40, 40)):
    """Draw a box sitting on the floor at a real position, via the real projection."""
    corners = np.array(
        [
            [x - size / 2, y - size / 2, 0.0],
            [x + size / 2, y - size / 2, 0.0],
            [x + size / 2, y + size / 2, 0.0],
            [x - size / 2, y + size / 2, 0.0],
            [x - size / 2, y - size / 2, height],
            [x + size / 2, y - size / 2, height],
            [x + size / 2, y + size / 2, height],
            [x - size / 2, y + size / 2, height],
        ]
    )
    pixels = camera.project(corners).astype(np.int32)
    hull = cv2.convexHull(pixels)
    cv2.fillConvexPoly(frame, hull, colour)
    x, y, w, h = cv2.boundingRect(hull)
    return frame, (float(x), float(y), float(x + w), float(y + h))


def test_the_walking_route_runs_from_the_bed_to_the_door(scanner):
    corridor = scanner.corridor()
    assert corridor is not None
    assert corridor.shape == (4, 2)
    width = float(np.linalg.norm(corridor[0] - corridor[3]))
    assert abs(width - scanner.t.corridor_width_m) < 1e-6


def test_a_bag_left_on_the_route_is_found_and_measured(room_and_camera, scanner):
    room, camera = room_and_camera
    reference = empty_room(camera)
    scanner.set_reference(reference)
    corridor = scanner.corridor()
    centre = polygon_centroid(corridor)
    current, _ = place_box(camera, reference.copy(), float(centre[0]), float(centre[1]))
    hazards = find_clutter(current, reference, scanner.frame, corridor, scanner.t)
    assert hazards, "a bag in the middle of the route was not found"
    assert hazards[0].kind == CLUTTER
    assert hazards[0].area_cm2 > scanner.t.clutter_area_cm2
    assert "route between the bed and the door" in hazards[0].description


def test_a_bag_well_off_the_route_is_not_called_a_hazard(room_and_camera, scanner):
    room, camera = room_and_camera
    reference = empty_room(camera)
    scanner.set_reference(reference)
    corridor = scanner.corridor()
    current, _ = place_box(camera, reference.copy(), 3.6, 0.6)
    hazards = find_clutter(current, reference, scanner.frame, corridor, scanner.t)
    assert not hazards


def test_the_person_is_not_mistaken_for_clutter(room_and_camera, scanner):
    room, camera = room_and_camera
    reference = empty_room(camera)
    corridor = scanner.corridor()
    centre = polygon_centroid(corridor)
    current, person = place_box(
        camera, reference.copy(), float(centre[0]), float(centre[1]), size=0.6, height=1.7
    )
    hazards = find_clutter(
        current, reference, scanner.frame, corridor, scanner.t, person_box=person
    )
    assert not hazards, "the person standing in the corridor was reported as clutter"


def test_a_wet_floor_sign_on_the_route_is_recognised(room_and_camera, scanner):
    room, camera = room_and_camera
    frame = empty_room(camera)
    corridor = scanner.corridor()
    centre = polygon_centroid(corridor)
    apex = camera.project(np.array([[centre[0], centre[1], 0.62]]))[0]
    left = camera.project(np.array([[centre[0] - 0.22, centre[1], 0.0]]))[0]
    right = camera.project(np.array([[centre[0] + 0.22, centre[1], 0.0]]))[0]
    triangle = np.array([apex, right, left], dtype=np.int32)
    cv2.fillConvexPoly(frame, triangle, (40, 220, 245))
    hazards = find_wet_floor_sign(frame, scanner.frame, corridor)
    assert hazards, "a yellow A-frame on the route was not recognised"
    assert hazards[0].kind == WET_FLOOR
    assert 0.0 < hazards[0].confidence <= 1.0


def test_beige_furniture_is_not_a_wet_floor_sign(room_and_camera, scanner):
    room, camera = room_and_camera
    frame = empty_room(camera)
    corridor = scanner.corridor()
    centre = polygon_centroid(corridor)
    pixels = camera.project(
        np.array(
            [
                [centre[0] - 0.3, centre[1] - 0.3, 0.0],
                [centre[0] + 0.3, centre[1] - 0.3, 0.0],
                [centre[0] + 0.3, centre[1] + 0.3, 0.5],
                [centre[0] - 0.3, centre[1] + 0.3, 0.5],
            ]
        )
    ).astype(np.int32)
    cv2.fillConvexPoly(frame, cv2.convexHull(pixels), (170, 200, 210))
    assert not find_wet_floor_sign(frame, scanner.frame, corridor)


def test_a_walking_frame_across_the_room_is_out_of_reach(room_and_camera, scanner):
    room, _ = room_and_camera
    seat = np.array([1.9, 1.55])
    hazards = check_aid_reach(scanner.frame, room.zones_of("aid"), seat, scanner.t)
    assert hazards and hazards[0].kind == AID_OUT_OF_REACH
    assert hazards[0].metres > scanner.t.aid_reach_m
    assert "reach" in hazards[0].description


def test_a_walking_frame_beside_the_bed_is_not_a_hazard(room_and_camera, scanner):
    room, _ = room_and_camera
    aid = room.zones_of("aid")[0]
    beside = polygon_centroid(zone_floor_polygon(scanner.frame, aid))
    assert not check_aid_reach(scanner.frame, (aid,), beside, scanner.t)


def test_the_scanner_says_what_it_checked_and_what_it_skipped(room_and_camera, scanner):
    room, camera = room_and_camera
    report = scanner.scan(empty_room(camera), 0.0, seat_floor_xy=np.array([1.9, 1.55]), force=True)
    assert "clutter on the route" in report.checked
    assert "walking frame within reach" in report.checked
    assert any("reference photograph" in s for s in report.skipped)


def test_a_room_without_a_door_zone_refuses_to_invent_a_route(room_and_camera):
    room, camera = room_and_camera
    scanner = HazardScanner(
        FloorFrame(room.floor), room.thresholds, bed_zones=room.zones_of("bed")
    )
    report = scanner.scan(empty_room(camera), 0.0, force=True)
    assert scanner.corridor() is None
    assert any("walking route" in s for s in report.skipped)
    assert not report.hazards
