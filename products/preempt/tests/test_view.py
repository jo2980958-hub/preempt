"""The honesty rails. Each failure is manufactured and each must be named."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from preempt.config import Thresholds
from preempt.view import (
    BLOCKED,
    CAMERA_MOVED,
    NO_PERSON,
    TOO_DARK,
    USABLE,
    ViewMonitor,
    exposure,
    focus,
    summarise,
)


def room_frame(seed: int = 0, shift: int = 0, brightness: int = 1) -> np.ndarray:
    """A plausible room: texture, edges, furniture. Shifted, it is a knocked camera."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(60, 200, (240, 320, 3), dtype=np.uint8)
    cv2.rectangle(frame, (40 + shift, 60), (140 + shift, 180), (30, 30, 30), -1)
    cv2.rectangle(frame, (200 + shift, 40), (300 + shift, 220), (220, 210, 190), 3)
    for x in range(0, 320, 16):
        cv2.line(frame, (x + shift, 0), (x + shift, 240), (90, 90, 90), 1)
    return np.clip(frame.astype(np.int32) * brightness, 0, 255).astype(np.uint8)


def test_a_dark_room_is_named_as_too_dark_with_the_brightness_it_measured():
    monitor = ViewMonitor(Thresholds())
    dark = (room_frame() * 0.06).astype(np.uint8)
    report = monitor.grade(dark, 1.0, person_seen=True)
    assert report.state == TOO_DARK
    assert not report.usable
    assert "brightness" in report.detail
    assert report.remedy
    assert report.mean_v < Thresholds().dark_mean_v


def test_a_blocked_lens_is_named_and_not_confused_with_darkness():
    monitor = ViewMonitor(Thresholds())
    blocked = np.full((240, 320, 3), 118, dtype=np.uint8)
    report = monitor.grade(blocked, 1.0, person_seen=True)
    assert report.state == BLOCKED
    assert report.laplacian_var < Thresholds().blocked_laplacian_var
    assert "lens" in report.remedy


def test_a_knocked_camera_invalidates_the_zones_and_says_so():
    monitor = ViewMonitor(Thresholds())
    assert monitor.grade(room_frame(1), 0.0, person_seen=True).state == USABLE
    report = monitor.grade(room_frame(1, shift=60), 1.0, person_seen=True)
    assert report.state == CAMERA_MOVED
    assert report.shift_px is not None and report.shift_px > Thresholds().camera_moved_px
    assert "re-draw" in report.remedy
    assert monitor.moved_since_s == 1.0


def test_a_camera_that_has_not_moved_is_not_reported_as_moved():
    monitor = ViewMonitor(Thresholds())
    monitor.grade(room_frame(2), 0.0, person_seen=True)
    for t in (1.0, 2.0, 3.0):
        assert monitor.grade(room_frame(2), t, person_seen=True).state == USABLE


def test_an_empty_room_is_reported_only_after_the_configured_patience():
    monitor = ViewMonitor(Thresholds(absent_s=5.0))
    monitor.grade(room_frame(3), 0.0, person_seen=True)
    assert monitor.grade(room_frame(3), 3.0, person_seen=False).state == USABLE
    assert monitor.grade(room_frame(3), 6.0, person_seen=False).state == NO_PERSON


def test_a_recorded_track_still_notices_that_nobody_is_in_view():
    monitor = ViewMonitor(Thresholds(absent_s=4.0))
    monitor.grade_recorded(0.0, person_seen=True)
    assert monitor.grade_recorded(2.0, person_seen=False).state == USABLE
    assert monitor.grade_recorded(5.0, person_seen=False).state == NO_PERSON


def test_a_recorded_view_verdict_is_carried_through():
    monitor = ViewMonitor(Thresholds())
    report = monitor.grade_recorded(1.0, person_seen=True, hint=BLOCKED)
    assert report.state == BLOCKED
    assert report.remedy


def test_exposure_and_focus_are_real_measurements():
    bright = np.full((64, 64, 3), 240, dtype=np.uint8)
    mean_v, dark_fraction = exposure(bright)
    assert mean_v > 200 and dark_fraction == 0.0
    assert focus(np.full((64, 64), 128, dtype=np.uint8)) < 1.0
    assert focus(room_frame(4)[:, :, 0]) > 50.0


def test_the_summary_counts_seconds_in_each_state():
    monitor = ViewMonitor(Thresholds())
    reports = [monitor.grade(room_frame(5), float(i), person_seen=True) for i in range(4)]
    reports.append(monitor.grade(np.full((240, 320, 3), 5, dtype=np.uint8), 5.0, person_seen=True))
    summary = summarise(reports)
    assert summary["frames"] == 5
    assert 0.0 < summary["usable_fraction"] < 1.0
    assert summary["by_state"][TOO_DARK] == 1
