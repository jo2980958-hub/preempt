"""Pose extraction: the crop maths, the SimCC decode, and the real models."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from conftest import needs_models

from preempt.pose import (
    KEYPOINT_NAMES,
    RTMPOSE_T,
    PoseFrame,
    bbox_to_affine,
    decode_simcc,
)


def test_the_crop_fixes_the_aspect_ratio_before_warping():
    """Skipping the aspect fix squashes the person and moves every wrist."""
    forward, inverse = bbox_to_affine((100.0, 100.0, 200.0, 500.0), (192, 256))
    scale_x = float(np.hypot(forward[0, 0], forward[1, 0]))
    scale_y = float(np.hypot(forward[0, 1], forward[1, 1]))
    assert abs(scale_x - scale_y) < 1e-6, "the warp is anisotropic; the person is squashed"


def test_the_crop_inverse_puts_a_point_back_where_it_started():
    box = (120.0, 80.0, 360.0, 620.0)
    forward, inverse = bbox_to_affine(box, (192, 256))
    point = np.array([240.0, 350.0, 1.0])
    in_crop = forward @ point
    back = inverse @ np.array([in_crop[0], in_crop[1], 1.0])
    assert np.allclose(back, point[:2], atol=1e-6)


def test_the_box_centre_lands_at_the_crop_centre():
    forward, _ = bbox_to_affine((0.0, 0.0, 100.0, 400.0), (192, 256))
    centre = forward @ np.array([50.0, 200.0, 1.0])
    assert np.allclose(centre, (96.0, 128.0), atol=1e-6)


def test_simcc_decoding_recovers_the_peak_and_the_weaker_axis_score():
    simcc_x = np.zeros((1, 17, 384), dtype=np.float32)
    simcc_y = np.zeros((1, 17, 512), dtype=np.float32)
    simcc_x[0, 0, 100] = 0.9
    simcc_y[0, 0, 300] = 0.4
    coords, scores = decode_simcc(simcc_x, simcc_y)
    assert coords.shape == (17, 2)
    assert np.allclose(coords[0], (50.0, 150.0))
    assert abs(float(scores[0]) - 0.4) < 1e-6, "a joint is only as sure as its weaker axis"


def test_a_pose_frame_hides_joints_below_the_score_floor():
    pose = PoseFrame(
        index=0,
        time_s=0.0,
        xy=np.tile(np.array([10.0, 20.0]), (17, 1)),
        scores=np.linspace(0.0, 0.8, 17),
    )
    assert pose.point("nose", 0.3) is None
    assert pose.point("right ankle", 0.3) is not None
    seen = pose.visible(0.3)
    assert seen.sum() == int((np.linspace(0.0, 0.8, 17) >= 0.3).sum())


def test_the_keypoint_names_are_the_coco_seventeen_in_order():
    assert len(KEYPOINT_NAMES) == 17
    assert KEYPOINT_NAMES[0] == "nose"
    assert KEYPOINT_NAMES[-1] == "right ankle"


@needs_models
def test_the_pose_model_loads_in_opencv_5_and_returns_seventeen_joints(estimator):
    assert estimator.pose.spec is RTMPOSE_T
    assert "Apache-2.0" in RTMPOSE_T.licence
    image = np.random.default_rng(1).integers(0, 255, (480, 320, 3), dtype=np.uint8)
    xy, scores = estimator.keypoints(image, (40.0, 40.0, 280.0, 440.0))
    assert xy.shape == (17, 2)
    assert scores.shape == (17,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


@needs_models
def test_no_person_in_the_frame_means_no_pose_rather_than_a_guess(estimator):
    flat = np.full((240, 320, 3), 128, dtype=np.uint8)
    assert estimator.estimate(flat, 0, 0.0) is None


@needs_models
def test_the_new_dnn_engine_agrees_with_the_classic_one():
    """Same weights, two graph engines. If they disagree the speed claim is void."""
    from preempt.pose import PoseEstimator

    image = np.random.default_rng(9).integers(0, 255, (480, 320, 3), dtype=np.uint8)
    box = (40.0, 40.0, 280.0, 440.0)
    new_xy, _ = PoseEstimator(engine="new").keypoints(image, box)
    classic_xy, _ = PoseEstimator(engine="classic").keypoints(image, box)
    assert np.allclose(new_xy, classic_xy, atol=1.0)


@needs_models
def test_the_estimator_keeps_no_image_after_it_returns(estimator):
    image = np.random.default_rng(3).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    before = estimator.stats.image_bytes_read
    estimator.estimate(image, 0, 0.0)
    assert estimator.stats.image_bytes_read > before
    assert not any(
        isinstance(v, np.ndarray) and v.shape == image.shape
        for v in vars(estimator).values()
    )
