"""Test fixtures. The models live under `models/`; `models/fetch.sh` puts them there."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

os.environ.setdefault("OPENCV26_YOLOX_URI", str(MODELS / "yolox_tiny.onnx"))
os.environ.setdefault("PREEMPT_RTMPOSE_URI", str(MODELS / "rtmpose-t-body7.onnx"))
os.environ.setdefault("PREEMPT_SAMPLES_DIR", str(ROOT / "samples"))


def models_present() -> bool:
    return (MODELS / "yolox_tiny.onnx").is_file() and (MODELS / "rtmpose-t-body7.onnx").is_file()


needs_models = pytest.mark.skipif(
    not models_present(),
    reason="ONNX models are absent; run products/preempt/models/fetch.sh",
)


@pytest.fixture(scope="session")
def estimator():
    from preempt.pose import PoseEstimator

    return PoseEstimator(engine="new")
