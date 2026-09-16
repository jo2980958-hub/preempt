"""Regression tests from real footage: the CDC chair-stand clip, as keypoints.

The fixture is the RTMPose output for the oblique shot of the CDC STEADI
"30-Second Chair Stand Test" (public domain), with the hand-made room it was run
in. It holds coordinates and scores only. No pixel of the video is in this
repository, which is the same rule the product lives by.

Two flaws were found on it, and neither showed on the synthetic track:

* the rise rule fired as she sat back down, three times in 12.8 seconds;
* standing still between repetitions was reported as walking, because the hips'
  floor position is singular when the camera is at about hip height.

The event times in the fixture were read from the keypoints themselves (knee
angle crossing 150 and 130 degrees), not from Preempt's own states, so these
tests cannot pass by agreeing with themselves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from preempt.config import RoomConfig
from preempt.exits import STANDING, WALKING
from preempt.geometry import FloorFrame
from preempt.kinematics import SHADOW_M_PER_PX_MAX, Kinematics, shadow_if_well_conditioned
from preempt.pipeline import Pipeline
from preempt.privacy import PrivacyGuard
from preempt.synth import default_room, load_track

FIXTURE = Path(__file__).parent / "data" / "cdc_chair_stand_oblique.track.json"
CALL_RUNGS = ("nudge", "station", "urgent")


@pytest.fixture(scope="module")
def replay():
    frames, room, meta = load_track(FIXTURE)
    pipeline = Pipeline(room, guard=PrivacyGuard("strict"))
    analysis = pipeline.run_track(frames, meta["fps"])
    return pipeline, analysis, meta["truth"]


def _calls(analysis) -> list[float]:
    return [c["time_s"] for c in analysis.calls if c["rung"] in CALL_RUNGS]


def test_the_fixture_is_keypoints_and_nothing_else():
    text = FIXTURE.read_text(encoding="utf-8")
    assert len(text) < 200_000
    for marker in ("base64", "data:image", "jpeg", "png"):
        assert marker not in text.lower()
    frames, room, _meta = load_track(FIXTURE)
    assert len(frames) > 100 and frames[0].xy.shape == (17, 2)
    assert room.calibration.camera_height == "assumed", (
        "the CDC room is a hand setup and must say so"
    )


def test_every_stand_is_called_before_the_knees_straighten(replay):
    _, analysis, truth = replay
    calls = _calls(analysis)
    for straight in truth["knees_straight_s"]:
        before = [t for t in calls if straight - 2.0 <= t <= straight]
        assert before, f"the stand that straightened at {straight} s was not called; calls {calls}"


def test_sitting_back_down_raises_no_call(replay):
    """The finding: three nudges as she sat down, at 2.83, 7.00 and 11.00 s."""
    _, analysis, truth = replay
    calls = _calls(analysis)
    rises = truth["knees_straight_s"]
    stray = [t for t in calls if not any(r - 2.0 <= t <= r for r in rises)]
    assert not stray, f"calls raised outside any stand, as she sat down: {stray}"
    assert len(calls) == len(rises), f"{len(calls)} calls for {len(rises)} stands: {calls}"


def test_standing_still_between_repetitions_is_not_walking(replay):
    pipeline, _, truth = replay
    for up, down in zip(truth["knees_straight_s"], truth["knees_bent_again_s"], strict=True):
        held = [r.state for r in pipeline.exits.timeline if up + 0.3 <= r.time_s <= down - 0.5]
        assert held, f"no readings between {up} and {down} s"
        walking = held.count(WALKING) / len(held)
        assert walking <= 0.1, f"{walking:.0%} of a still stand at {up} s read as walking"
        assert held.count(STANDING) / len(held) >= 0.8


def test_the_sit_down_says_so_in_its_reasons(replay):
    pipeline, _, truth = replay
    after = truth["knees_bent_again_s"][0]
    reasons = [
        reason
        for r in pipeline.exits.timeline
        if after - 0.3 <= r.time_s <= after + 0.8
        for reason in r.reasons
    ]
    assert any(reason.startswith("sitting down") for reason in reasons), reasons


# --- the geometry behind the walking finding -------------------------------


def test_a_hip_at_camera_height_has_no_usable_floor_shadow(replay):
    """Standing in the oblique shot: level camera at 0.81 m, hips at about 0.86 m."""
    _, _, truth = replay
    frames, room, _ = load_track(FIXTURE)
    frame = FloorFrame(room.floor)
    kinematics = Kinematics(frame, room.thresholds)
    up, down = truth["knees_straight_s"][0], truth["knees_bent_again_s"][0]
    standing = [f for f in frames if up + 0.3 <= f.time_s <= down - 0.5]
    assert standing
    for pose in standing:
        body = kinematics.state(pose)
        hip = pose.midpoint("left hip", "right hip", room.thresholds.keypoint_score_min)
        assert body.hip_height is not None and hip is not None
        assert shadow_if_well_conditioned(frame, hip, body.hip_height) is None
        np.testing.assert_allclose(body.torso_floor, body.floor_xy)


def test_the_synthetic_ward_camera_keeps_its_hip_shadow():
    room, camera = default_room()
    frame = FloorFrame(room.floor)
    hip = camera.project(np.array([[2.0, 1.8, 0.93]]))[0]
    shadow = shadow_if_well_conditioned(frame, hip, 0.93)
    assert shadow is not None
    assert np.linalg.norm(shadow - np.array([2.0, 1.8])) < 0.05
    nudged = frame.floor_shadow(hip + np.array([0.0, 1.0]), 0.93)
    assert np.linalg.norm(nudged - shadow) < SHADOW_M_PER_PX_MAX


def test_the_hand_room_loads_the_same_way_the_cli_loads_it(tmp_path):
    _frames, room, _ = load_track(FIXTURE)
    path = tmp_path / "room.json"
    room.save(path)
    again = RoomConfig.load(path)
    assert again.to_dict() == room.to_dict()
