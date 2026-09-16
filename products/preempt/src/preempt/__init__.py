"""Preempt - a call before the fall, from an abstracted pose and nothing else.

A camera in a hospital or care-home room watches for the movement that comes
before standing, scores how steady someone is once they are up, checks what is in
the way, and escalates. The frame is destroyed as soon as the pose has been read,
so the evidence a nurse sees is a stick figure on a room plan and never a picture
of a patient.

Start here:
    from preempt import Pipeline, RoomConfig, analyse_track
    record, analysis = analyse_track("samples/bed-exit-steady.json")
    print(record.metrics["lead_time_s"])
"""

from __future__ import annotations

from .config import FloorPlane, RoomConfig, Thresholds, Zone
from .escalation import Call, EscalationLadder
from .exits import ExitDetector, ExitReading
from .gait import GaitReport, GaitWindow
from .geometry import FloorFrame, GeometryError, HeightEstimate
from .hazards import Hazard, HazardReport, HazardScanner
from .kinematics import BodyState, Kinematics
from .pipeline import (
    Analysis,
    Moment,
    Pipeline,
    analyse_track,
    analyse_video,
    attach_evidence,
    build_record,
)
from .pose import KEYPOINT_NAMES, PoseEstimator, PoseFrame
from .privacy import PrivacyGuard, PrivacyViolation, Provenance
from .risk import RiskState, assess
from .view import ViewMonitor, ViewReport

__version__ = "1.0.0"

__all__ = [
    "KEYPOINT_NAMES",
    "Analysis",
    "BodyState",
    "Call",
    "EscalationLadder",
    "ExitDetector",
    "ExitReading",
    "FloorFrame",
    "FloorPlane",
    "GaitReport",
    "GaitWindow",
    "GeometryError",
    "Hazard",
    "HazardReport",
    "HazardScanner",
    "HeightEstimate",
    "Kinematics",
    "Moment",
    "Pipeline",
    "PoseEstimator",
    "PoseFrame",
    "PrivacyGuard",
    "PrivacyViolation",
    "Provenance",
    "RiskState",
    "RoomConfig",
    "Thresholds",
    "ViewMonitor",
    "ViewReport",
    "Zone",
    "__version__",
    "analyse_track",
    "analyse_video",
    "assess",
    "attach_evidence",
    "build_record",
]
