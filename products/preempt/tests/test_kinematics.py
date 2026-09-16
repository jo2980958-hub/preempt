"""Kinematics, gait statistics, configuration round-trips and the evaluation harness."""

from __future__ import annotations

import numpy as np
import pytest
from preempt.config import FloorPlane, RoomConfig, Thresholds, Zone
from preempt.evaluate import Outcome, Report, run_synthetic
from preempt.gait import GaitWindow, perpendicular_deviation, smooth, stance_onsets
from preempt.geometry import FloorFrame
from preempt.kinematics import ComTracker, Kinematics, centre_of_mass, knee_angle_deg
from preempt.pose import KEYPOINT_NAMES, PoseFrame
from preempt.synth import default_room, place, posture


@pytest.fixture(scope="module")
def room_and_camera():
    return default_room()


def pose_from(camera, joints_world, index=0, time_s=0.0, score=0.9) -> PoseFrame:
    xy = camera.project(joints_world)
    return PoseFrame(index, time_s, xy, np.full(17, score))


def test_the_centre_of_mass_sits_between_the_hips_and_the_shoulders(room_and_camera):
    _room, camera = room_and_camera
    pose = pose_from(camera, place(posture(), (2.0, 2.0), 0.0))
    com, seen = centre_of_mass(pose, 0.3)
    assert seen == 13
    hip = pose.midpoint("left hip", "right hip", 0.3)
    shoulder = pose.midpoint("left shoulder", "right shoulder", 0.3)
    assert min(shoulder[1], hip[1]) - 40 < com[1] < max(shoulder[1], hip[1]) + 60


def test_an_occluded_arm_shifts_the_centre_of_mass_rather_than_biasing_it(room_and_camera):
    _room, camera = room_and_camera
    joints = place(posture(), (2.0, 2.0), 0.0)
    full = pose_from(camera, joints)
    scores = np.full(17, 0.9)
    for name in ("left wrist", "left elbow"):
        scores[KEYPOINT_NAMES.index(name)] = 0.05
    partial = PoseFrame(0, 0.0, camera.project(joints), scores)
    a, _ = centre_of_mass(full, 0.3)
    b, seen = centre_of_mass(partial, 0.3)
    assert seen == 11
    assert float(np.linalg.norm(a - b)) < 8.0


def test_the_knee_angle_is_straight_standing_and_bent_sitting(room_and_camera):
    _room, camera = room_and_camera
    standing = pose_from(camera, place(posture(), (2.0, 2.2), 90.0))
    seated = pose_from(camera, place(posture(knee_bend=0.95, seat_height=0.5), (2.0, 2.2), 90.0))
    assert knee_angle_deg(standing, 0.3) > 150.0
    assert knee_angle_deg(seated, 0.3) < 140.0


def test_hip_height_separates_standing_from_sitting_in_metres(room_and_camera):
    room, camera = room_and_camera
    kinematics = Kinematics(FloorFrame(room.floor), Thresholds())
    standing = kinematics.state(pose_from(camera, place(posture(), (2.0, 2.2), 90.0)))
    seated = kinematics.state(
        pose_from(camera, place(posture(knee_bend=0.95, seat_height=0.46), (2.0, 2.2), 90.0))
    )
    assert standing.hip_height > 0.80
    assert seated.hip_height < 0.55
    assert standing.head_height > 1.5
    assert abs(standing.trunk_deg) < 12.0


def test_leaning_forward_over_the_feet_is_a_positive_lean_and_lying_back_is_negative(
    room_and_camera,
):
    room, camera = room_and_camera
    kinematics = Kinematics(FloorFrame(room.floor), Thresholds())
    upright = kinematics.state(
        pose_from(camera, place(posture(knee_bend=0.95, seat_height=0.6), (1.9, 1.6), 90.0))
    )
    forward = kinematics.state(
        pose_from(
            camera,
            place(posture(knee_bend=0.95, seat_height=0.6, trunk_lean_deg=40.0), (1.9, 1.6), 90.0),
        )
    )
    backward = kinematics.state(
        pose_from(
            camera,
            place(posture(knee_bend=0.95, seat_height=0.6, trunk_lean_deg=-40.0), (1.9, 1.6), 90.0),
        )
    )
    assert forward.lean_offset > upright.lean_offset
    assert backward.lean_offset < upright.lean_offset


def test_the_kalman_tracker_smooths_a_noisy_track_without_lagging_it():
    tracker = ComTracker()
    truth, estimates = [], []
    rng = np.random.default_rng(2)
    for i in range(60):
        t = i / 30.0
        position = np.array([0.5 * t, 0.0])
        truth.append(position)
        noisy = position + rng.normal(0.0, 0.02, 2)
        result = tracker.update(noisy, t)
        if result is not None:
            estimates.append(result)
    final_position, final_velocity = estimates[-1]
    assert abs(float(final_velocity[0]) - 0.5) < 0.12, final_velocity
    assert float(np.linalg.norm(final_position - truth[-1])) < 0.05


def test_the_tracker_predicts_through_a_dropout_rather_than_spiking():
    tracker = ComTracker()
    for i in range(90):
        tracker.update(np.array([0.5 * i / 30.0, 0.0]), i / 30.0)
    predicted = tracker.update(None, 91 / 30.0)
    assert predicted is not None
    position, velocity = predicted
    assert 0.42 < float(velocity[0]) < 0.58, velocity
    assert 1.45 < float(position[0]) < 1.58, position


def test_sway_is_measured_about_the_fitted_walking_line_not_a_fixed_axis():
    """Walking diagonally is not swaying, and the line fit is what knows that."""
    t = np.linspace(0, 3, 60)
    straight = np.stack([t, 0.6 * t], axis=1)
    deviation, _ = perpendicular_deviation(straight)
    assert float(np.sqrt(np.mean(deviation**2))) < 0.01

    wobbly = straight + np.stack([-0.6 * np.sin(6 * t), np.sin(6 * t)], axis=1) * 0.08 / np.sqrt(
        1.36
    )
    deviation, _ = perpendicular_deviation(wobbly)
    assert float(np.sqrt(np.mean(deviation**2))) > 0.04


def test_footfalls_are_the_peaks_of_the_gap_between_the_feet():
    times = np.linspace(0, 4, 60)
    gaps = 0.3 + 0.25 * np.abs(np.sin(2 * np.pi * 1.0 * times))
    onsets = stance_onsets(gaps, times)
    assert 6 <= len(onsets) <= 10, onsets


def test_smoothing_uses_an_opencv_kernel_and_preserves_length():
    series = np.random.default_rng(1).normal(size=40)
    out = smooth(series, 2.0)
    assert out.shape == series.shape
    assert float(out.std()) < float(series.std())


def test_gait_refuses_to_score_someone_standing_still(room_and_camera):
    room, camera = room_and_camera
    frame = FloorFrame(room.floor)
    kinematics = Kinematics(frame, Thresholds())
    window = GaitWindow(Thresholds(), frame)
    for i in range(80):
        pose = pose_from(camera, place(posture(), (2.0, 2.0), 90.0), i, i / 15.0)
        window.push(kinematics.state(pose))
    report = window.score(15.0)
    assert not report.scored
    assert "standing still" in report.insufficient or "footfalls" in report.insufficient


def test_the_room_configuration_round_trips_through_json(tmp_path):
    room, _ = default_room()
    path = tmp_path / "room.json"
    room.save(path)
    back = RoomConfig.load(path)
    assert back.room == room.room
    assert len(back.zones) == len(room.zones)
    assert back.zones_of("bed")[0].name == room.zones_of("bed")[0].name
    assert back.thresholds.stand_hip_height_m == room.thresholds.stand_hip_height_m
    assert back.floor.vertical_vanishing_point == room.floor.vertical_vanishing_point


def test_a_zone_of_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="unknown zone kind"):
        Zone("bad", "sofa", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)))
    with pytest.raises(ValueError, match="at least 3 points"):
        Zone("bad", "bed", ((0.0, 0.0), (1.0, 0.0)))


def test_a_floor_plane_needs_exactly_four_points():
    with pytest.raises(ValueError, match="exactly 4"):
        FloorPlane(image_points=((0.0, 0.0),), world_points=((0.0, 0.0),))


def test_the_evaluation_summary_reports_what_it_claims_to():
    report = Report(source="test")
    report.outcomes = [
        Outcome("a", True, True, "rising soon", "rising soon", 3.0, 1, 20.0, True),
        Outcome("b", True, False, "watch", "rising soon", None, 0, 20.0, False, "missed"),
        Outcome("c", False, False, "settled", "", None, 0, 600.0, True),
        Outcome("d", False, True, "unsteady", "", None, 1, 600.0, False, "false alarm"),
    ]
    summary = report.summary()
    assert summary["detection_rate"] == 0.5
    assert summary["lead_time_s_bed_and_chair_exits"]["median"] == 3.0
    assert summary["false_alarms"] == 1
    assert abs(summary["quiet_hours_observed"] - 1200 / 3600) < 1e-4
    # One stray call in twenty minutes of quiet is 36 a night, which is the point
    # of scaling it: the small number sounds fine and the scaled one does not.
    assert summary["false_alarms_per_bed_night"] == pytest.approx(36.0, abs=0.1)
    assert len(summary["failures"]) == 2


@pytest.mark.slow
def test_the_whole_synthetic_evaluation_finds_every_event_and_calls_on_nothing_else():
    summary = run_synthetic(seeds=2, quiet_loops=4).summary()
    assert summary["detection_rate"] == 1.0
    assert summary["false_alarms"] == 0
    assert summary["lead_time_s_bed_and_chair_exits"]["median"] > 2.0
    assert summary["failures"] == []
