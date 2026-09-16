"""Pose extraction: the only stage that ever sees a picture of a person.

Everything downstream of `PoseEstimator.estimate` works on seventeen (x, y,
score) triples. That is not a stylistic choice, it is the product's whole
privacy argument, so this module is the boundary and it is written to be read by
someone checking that claim.

Models, and why these two
------------------------
* **YOLOX-tiny** (Apache-2.0, Megvii) finds the person box. Already packaged in
  `visioncore`; it loads in `cv2.dnn` with no second runtime.
* **RTMPose-t** (Apache-2.0, OpenMMLab) turns that box into 17 COCO keypoints.
  256x192 input, SimCC output: two 1-D probability fields per joint, `simcc_x`
  at 384 bins and `simcc_y` at 512 bins, both at `simcc_split_ratio = 2`.

Ultralytics' pose models would have been easier and are AGPL-3.0, whose section
13 makes a hosted demo a source-disclosure event. They are not used here and
must not be added. See `docs/report.md`.

Measured on this machine (x86, 22 threads, `opencv-python==5.0.0.93`):
RTMPose-t forward is 3.1 ms with `ENGINE_NEW` against 5.3 ms with
`ENGINE_CLASSIC`, and YOLOX-tiny is 13.6 ms. Numbers reproduce with
`python -m preempt.cli bench`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from visioncore import Detection, DnnRunner, ModelSpec, YoloxDetector
from visioncore.timing import stage

KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left eye",
    "right eye",
    "left ear",
    "right ear",
    "left shoulder",
    "right shoulder",
    "left elbow",
    "right elbow",
    "left wrist",
    "right wrist",
    "left hip",
    "right hip",
    "left knee",
    "right knee",
    "left ankle",
    "right ankle",
)
K = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

SKELETON: tuple[tuple[int, int], ...] = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
)

RTMPOSE_T = ModelSpec(
    name="rtmpose-t-body7",
    uri=os.environ.get(
        "PREEMPT_RTMPOSE_URI",
        str(Path(__file__).resolve().parents[2] / "models" / "rtmpose-t-body7.onnx"),
    ),
    input_size=(192, 256),
    licence="Apache-2.0 (OpenMMLab MMPose / RTMPose)",
    version="body7-420e-256x192",
    notes=(
        "SimCC head: simcc_x (1,17,384) and simcc_y (1,17,512), split ratio 2.0. "
        "ImageNet mean/std, RGB. Source: download.openmmlab.com rtmposev1 onnx_sdk."
    ),
)

RTMPOSE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
RTMPOSE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
SIMCC_SPLIT_RATIO = 2.0


@dataclass(frozen=True)
class PoseFrame:
    """One person's pose at one instant, and nothing else from that frame.

    `xy` is (17, 2) in original image pixels; `scores` is (17,). A score below
    the configured floor means the joint was not seen, and every consumer checks
    `visible()` rather than trusting a coordinate.
    """

    index: int
    time_s: float
    xy: np.ndarray
    scores: np.ndarray
    box: tuple[float, float, float, float] | None = None
    box_score: float = 0.0

    def visible(self, min_score: float) -> np.ndarray:
        return self.scores >= min_score

    def point(self, name: str, min_score: float) -> np.ndarray | None:
        i = K[name]
        return self.xy[i] if self.scores[i] >= min_score else None

    def midpoint(self, a: str, b: str, min_score: float) -> np.ndarray | None:
        pa, pb = self.point(a, min_score), self.point(b, min_score)
        if pa is not None and pb is not None:
            return (pa + pb) / 2.0
        return pa if pa is not None else pb

    def to_dict(self, min_score: float = 0.0) -> dict[str, Any]:
        return {
            "index": self.index,
            "time_s": round(self.time_s, 3),
            "keypoints": [
                {
                    "name": name,
                    "x": round(float(self.xy[i][0]), 1),
                    "y": round(float(self.xy[i][1]), 1),
                    "score": round(float(self.scores[i]), 3),
                    "seen": bool(self.scores[i] >= min_score),
                }
                for i, name in enumerate(KEYPOINT_NAMES)
            ],
        }


@dataclass
class PoseStats:
    """What the estimator did, for the privacy ledger and the report."""

    frames_seen: int = 0
    frames_with_person: int = 0
    detector_ms: float = 0.0
    pose_ms: float = 0.0
    image_bytes_read: int = 0
    keypoint_bytes_kept: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_seen": self.frames_seen,
            "frames_with_person": self.frames_with_person,
            "detector_ms_per_frame": round(
                self.detector_ms / max(1, self.frames_seen), 2
            ),
            "pose_ms_per_frame": round(
                self.pose_ms / max(1, self.frames_with_person), 2
            ),
            "image_bytes_read": self.image_bytes_read,
            "keypoint_bytes_kept": self.keypoint_bytes_kept,
            "compression_ratio": round(
                self.image_bytes_read / max(1, self.keypoint_bytes_kept), 1
            ),
        }


# ---------------------------------------------------------------------------
# Top-down crop, the part everyone gets wrong
# ---------------------------------------------------------------------------


def bbox_to_affine(
    box: tuple[float, float, float, float],
    input_size: tuple[int, int],
    padding: float = 1.25,
) -> tuple[np.ndarray, np.ndarray]:
    """The 2x3 warp RTMPose expects, plus its inverse.

    MMPose's `TopDownGetBboxCenterScale` pads the box by 1.25 and then *fixes the
    aspect ratio to the model input* before the affine warp. Skipping the aspect
    fix squashes the person and the wrists land several centimetres out, which is
    exactly the size of error this product is trying to measure. So it is done
    here, once, and tested.
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = max(1.0, x2 - x1) * padding, max(1.0, y2 - y1) * padding
    in_w, in_h = input_size
    aspect = in_w / in_h
    if w > h * aspect:
        h = w / aspect
    else:
        w = h * aspect
    src = np.array([[cx, cy], [cx, cy - h / 2.0], [cx - w / 2.0, cy]], dtype=np.float32)
    dst = np.array(
        [[in_w / 2.0, in_h / 2.0], [in_w / 2.0, 0.0], [0.0, in_h / 2.0]], dtype=np.float32
    )
    forward = cv2.getAffineTransform(src, dst)
    inverse = cv2.invertAffineTransform(forward)
    return forward, inverse


def decode_simcc(
    simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float = SIMCC_SPLIT_RATIO
) -> tuple[np.ndarray, np.ndarray]:
    """SimCC argmax to (17, 2) model-space pixels and (17,) scores.

    SimCC represents each coordinate as a 1-D classification over sub-pixel bins,
    so the coordinate is `argmax / split_ratio` and the score is the joint
    minimum of the two peak responses -- a joint is only as certain as its less
    certain axis.
    """
    x = np.asarray(simcc_x, dtype=np.float32).reshape(-1, simcc_x.shape[-1])
    y = np.asarray(simcc_y, dtype=np.float32).reshape(-1, simcc_y.shape[-1])
    ix = x.argmax(axis=1)
    iy = y.argmax(axis=1)
    sx = x[np.arange(x.shape[0]), ix]
    sy = y[np.arange(y.shape[0]), iy]
    coords = np.stack([ix / split_ratio, iy / split_ratio], axis=1).astype(np.float64)
    scores = np.minimum(sx, sy).astype(np.float64)
    coords[scores <= 0.0] = 0.0
    return coords, np.clip(scores, 0.0, 1.0)


class PoseEstimator:
    """YOLOX-tiny finds the person, RTMPose-t reads the joints, the frame is dropped.

    `estimate` takes an image and returns a `PoseFrame` or None. It keeps no
    reference to the image after it returns, and it never writes one anywhere.
    """

    def __init__(
        self,
        *,
        detector: YoloxDetector | None = None,
        pose_spec: ModelSpec = RTMPOSE_T,
        engine: str = "new",
        person_score: float = 0.25,
        cache_dir: Path | None = None,
        coast_frames: int = 6,
    ) -> None:
        self.detector = detector or YoloxDetector(engine=engine, score_threshold=person_score)
        self.pose = DnnRunner(pose_spec, engine=engine, cache_dir=cache_dir)
        self.person_score = person_score
        self.coast_frames = coast_frames
        self.stats = PoseStats()
        self._out_names = self.pose.net.getUnconnectedOutLayersNames()
        self._last_box: tuple[float, float, float, float] | None = None
        self._coasting = 0

    # -- pieces, each testable on its own ----------------------------------
    def largest_person(self, image: np.ndarray) -> Detection | None:
        with stage("detect:yolox") as timer:
            detections = self.detector.detect(image)
        self.stats.detector_ms += timer.ms
        people = [d for d in detections if d.class_name == "person"]
        if not people:
            return None
        return max(people, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))

    def keypoints(
        self, image: np.ndarray, box: tuple[float, float, float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        in_w, in_h = self.pose.spec.input_size
        forward, inverse = bbox_to_affine(box, (in_w, in_h))
        crop = cv2.warpAffine(image, forward, (in_w, in_h), flags=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        normalised = (rgb - RTMPOSE_MEAN) / RTMPOSE_STD
        blob = np.ascontiguousarray(normalised.transpose(2, 0, 1)[None, ...])
        with stage("pose:rtmpose") as timer:
            with self.pose._lock:  # noqa: SLF001 - DnnRunner's documented serialisation
                self.pose.net.setInput(blob)
                outputs = self.pose.net.forward(self._out_names)
        self.stats.pose_ms += timer.ms
        by_name = dict(zip(self._out_names, outputs, strict=True))
        coords, scores = decode_simcc(by_name["simcc_x"], by_name["simcc_y"])
        homogeneous = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
        return homogeneous @ inverse.T, scores

    # -- the boundary -------------------------------------------------------
    def _coast_box(self, image: np.ndarray) -> tuple[float, float, float, float] | None:
        """The last known box, widened, for a few frames after the detector loses them.

        This is not a nicety. Measured on the UR Fall Detection Dataset, the
        detector loses a person the moment they are on the floor -- a person lying
        down is a wide short blob that looks nothing like the upright people a
        detector sees most of -- and the pose estimator never gets a chance. The
        frames it loses are exactly the frames that matter.

        A person who has just gone down is still within a body length of where
        they were, so re-running the pose estimator on the last box widened by a
        fifth recovers most of them. It coasts for a few frames only: coasting
        forever would invent a person who has walked out of the room.
        """
        if self._last_box is None or self._coasting >= self.coast_frames:
            return None
        self._coasting += 1
        x1, y1, x2, y2 = self._last_box
        width, height = x2 - x1, y2 - y1
        pad_x, pad_y = width * 0.20, height * 0.20
        return (
            max(0.0, x1 - pad_x),
            max(0.0, y1 - pad_y),
            min(float(image.shape[1]), x2 + pad_x),
            min(float(image.shape[0]), y2 + pad_y),
        )

    def estimate(self, image: np.ndarray, index: int, time_s: float) -> PoseFrame | None:
        """Image in, keypoints out. Nothing about the image survives this call."""
        self.stats.frames_seen += 1
        self.stats.image_bytes_read += int(image.nbytes)
        detection = self.largest_person(image)
        if detection is not None:
            box, score = detection.bbox, detection.score
            self._coasting = 0
        else:
            box = self._coast_box(image)
            if box is None:
                self._last_box = None
                return None
            score = 0.0
            self.stats.notes.append(f"coasted on the last box at {time_s:.2f} s")
        xy, scores = self.keypoints(image, box)
        if float(np.median(scores)) < 0.15 and detection is None:
            return None  # the coast landed on nothing; do not invent a person
        self._last_box = _box_around(xy, scores, image.shape) or box
        self.stats.frames_with_person += 1
        self.stats.keypoint_bytes_kept += int(xy.nbytes + scores.nbytes)
        return PoseFrame(
            index=index, time_s=time_s, xy=xy, scores=scores, box=box, box_score=score,
        )

    def info(self) -> dict[str, Any]:
        return {"detector": self.detector.info(), "pose": self.pose.info()}


def _box_around(
    xy: np.ndarray, scores: np.ndarray, shape: tuple[int, ...], min_score: float = 0.3
) -> tuple[float, float, float, float] | None:
    """A box around the joints we are sure of, to seed the next frame."""
    seen = scores >= min_score
    if seen.sum() < 4:
        return None
    points = xy[seen]
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    pad_x = max(12.0, (x2 - x1) * 0.12)
    pad_y = max(12.0, (y2 - y1) * 0.12)
    return (
        float(max(0.0, x1 - pad_x)),
        float(max(0.0, y1 - pad_y)),
        float(min(shape[1], x2 + pad_x)),
        float(min(shape[0], y2 + pad_y)),
    )
