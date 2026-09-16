"""The whole loop, in one place, in the order it runs.

    frame -> view grade -> pose -> **frame destroyed** -> kinematics -> exit state
          -> gait window -> hazard scan -> risk state -> escalation -> timeline

Two entry points, because a ward deployment and a demonstration are genuinely
different things:

`analyse_video` runs the full pipeline including YOLOX and RTMPose. That is the
path a camera takes.

`analyse_track` starts from a recorded pose track. That is not a shortcut: a pose
track is exactly what a Preempt device holds after the camera stage, so replaying
one is replaying what the device really keeps. It is also the only honest way to
ship a demonstration, because shipping a video of a patient would contradict the
product. The bundled samples are tracks.

Both produce the same `RunRecord`, so the UI, the tests and the evaluation
harness never have to know which one ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from visioncore import Evidence, RunRecord, iter_video, recording, video_info
from visioncore.timing import stage

from .config import RoomConfig
from .escalation import EscalationLadder
from .exits import STANDING, WALKING, ExitDetector, ExitReading
from .gait import GaitReport, GaitWindow
from .geometry import FloorFrame
from .hazards import HazardReport, HazardScanner
from .kinematics import Kinematics
from .pose import PoseEstimator, PoseFrame
from .privacy import PrivacyGuard, Provenance
from .render import encode_png, plan_card, pose_card
from .risk import FLOOR, RISING_SOON, UNSTEADY, RiskState, assess
from .synth import load_track
from .view import USABLE, ViewMonitor, ViewReport, summarise

PRODUCT = "preempt"


@dataclass
class Moment:
    """One analysed instant. The timeline the UI scrubs is a list of these."""

    time_s: float
    risk: RiskState
    view: ViewReport
    exit_reading: ExitReading | None
    gait: GaitReport | None
    pose: PoseFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": round(self.time_s, 3),
            "risk": self.risk.to_dict(),
            "view": self.view.to_dict(),
            "exit": self.exit_reading.to_dict() if self.exit_reading else None,
            "gait": self.gait.to_dict() if self.gait else None,
        }


@dataclass
class Analysis:
    """Everything one run produced, before it is flattened into a RunRecord."""

    room: RoomConfig
    fps: float
    moments: list[Moment] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    hazards: HazardReport | None = None
    privacy: dict[str, Any] = field(default_factory=dict)
    lead_time_s: float | None = None
    source: str = ""

    @property
    def peak(self) -> Moment | None:
        if not self.moments:
            return None
        return max(self.moments, key=lambda m: m.risk.rank)


class Pipeline:
    """One room, one run. Construct, call `run_*`, read the record."""

    def __init__(
        self,
        room: RoomConfig,
        *,
        guard: PrivacyGuard | None = None,
        estimator: PoseEstimator | None = None,
        sample_hz: float = 15.0,
    ) -> None:
        if room.floor is None:
            raise ValueError(
                "this room has no floor plane; a member of staff must mark four floor "
                "points before anything can be measured in metres"
            )
        self.room = room
        self.frame = FloorFrame(room.floor)
        self.guard = guard or PrivacyGuard(room.privacy_mode)
        self.estimator = estimator
        self.sample_hz = sample_hz
        t = room.thresholds
        self.kinematics = Kinematics(self.frame, t)
        self.exits = ExitDetector(
            t, self.frame,
            bed_zones=room.zones_of("bed"),
            chair_zones=room.zones_of("chair"),
        )
        self.gait = GaitWindow(t, self.frame, walls=room.zones_of("wall"))
        self.hazards = HazardScanner(
            self.frame, t,
            bed_zones=room.zones_of("bed"),
            door_zones=room.zones_of("door"),
            aid_zones=room.zones_of("aid"),
        )
        self.ladder = EscalationLadder(t)
        self.view = ViewMonitor(t)
        self.moments: list[Moment] = []
        self.view_reports: list[ViewReport] = []

    # -- one instant --------------------------------------------------------
    def step(
        self,
        pose: PoseFrame | None,
        time_s: float,
        view: ViewReport,
        *,
        hazards: HazardReport | None = None,
    ) -> Moment:
        self.view_reports.append(view)
        exit_reading: ExitReading | None = None
        gait_report: GaitReport | None = None

        if pose is not None and view.usable:
            body = self.kinematics.state(pose)
            exit_reading = self.exits.update(body)
            # Gait is a property of walking, so the window only ever holds frames
            # where the person was on their feet. Letting a sit-to-stand into it
            # put a metre of apparent sway into the next four seconds of score.
            if exit_reading.state in (STANDING, WALKING):
                self.gait.push(body)
                gait_report = self.gait.score(self.sample_hz)
            else:
                self.gait.clear()

        risk = assess(
            time_s, view, exit_reading, gait_report, hazards, self.room.thresholds
        )
        self.ladder.update(risk, has_hazard=bool(hazards and hazards.any))
        moment = Moment(time_s, risk, view, exit_reading, gait_report, pose)
        self.moments.append(moment)
        return moment

    # -- whole runs ---------------------------------------------------------
    def run_track(
        self,
        frames: list[PoseFrame],
        fps: float,
        progress=None,
        view_hints: dict[int, str] | None = None,
    ) -> Analysis:
        """Replay a recorded pose track. No images exist at any point in this call."""
        view_hints = view_hints or {}
        stride = max(1, int(round(fps / self.sample_hz)))
        kept = [f for i, f in enumerate(frames) if i % stride == 0]
        total = max(1, len(kept))
        for n, pose in enumerate(kept):
            usable_pose = (
                pose
                if int((pose.scores >= self.room.thresholds.keypoint_score_min).sum())
                >= self.room.thresholds.pose_keypoints_min
                else None
            )
            view = self.view.grade_recorded(
                pose.time_s,
                person_seen=usable_pose is not None,
                hint=view_hints.get(pose.index, USABLE),
            )
            hazards = self.hazards.scan(
                np.zeros((2, 2, 3), dtype=np.uint8), pose.time_s,
                seat_floor_xy=self._seat_position(),
                force=False,
            ) if self.room.zones_of("aid") else None
            self.step(usable_pose, pose.time_s, view, hazards=hazards)
            self.guard.ledger.frames_examined += 1
            self.guard.keypoints_kept(int(pose.xy.nbytes + pose.scores.nbytes))
            if progress and n % 20 == 0:
                progress(100.0 * n / total, f"{pose.time_s:.1f} s of pose replayed")
        return self._finish(fps, source="pose track")

    def run_images(self, paths: list[Path], *, fps: float = 30.0, progress=None) -> Analysis:
        """The camera path over a directory of numbered stills.

        Public fall datasets ship extracted frames rather than a container, so
        this exists for the evaluation harness. It is the same loop as
        `run_video` and it destroys each frame in the same place.
        """
        if self.estimator is None:
            self.estimator = PoseEstimator()
        total = max(1, len(paths))
        for index, path in enumerate(paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            time_s = index / fps
            self.guard.examined(image)
            if self.hazards.reference is None:
                self.hazards.set_reference(image)
            with stage("pose:total"):
                pose = self.estimator.estimate(image, index, time_s)
            view = self.view.grade(image, time_s, person_seen=pose is not None)
            self.guard.release(image)
            if pose is not None:
                self.guard.keypoints_kept(int(pose.xy.nbytes + pose.scores.nbytes))
            self.step(pose, time_s, view, hazards=None)
            if progress and index % 20 == 0:
                progress(min(99.0, 100.0 * index / total), f"{time_s:.1f} s analysed")
        return self._finish(fps, source=f"{len(paths)} frames")

    def run_video(self, path: str | Path, progress=None) -> Analysis:
        """The camera path: decode, grade, pose, destroy the frame, reason."""
        if self.estimator is None:
            self.estimator = PoseEstimator()
        info = video_info(path)
        fps = info.fps if info.fps > 0 else (self.room.fps_hint or 30.0)
        stride = max(1, int(round(fps / self.sample_hz)))
        total_frames = max(1, info.frame_count // stride)
        seen = 0
        for frame in iter_video(path, stride=stride, max_side=1280):
            time_s = frame.timestamp_ms / 1000.0
            image = frame.image
            self.guard.examined(image)
            if self.hazards.reference is None:
                self.hazards.set_reference(image)
            with stage("pose:total"):
                pose = self.estimator.estimate(image, frame.index, time_s)
            view = self.view.grade(image, time_s, person_seen=pose is not None)
            hazards = self.hazards.scan(
                image, time_s,
                seat_floor_xy=self._seat_position(),
                person_box=pose.box if pose else None,
            )
            # Everything the camera gave us has now been read. Destroy it.
            self.guard.release(image)
            if pose is not None:
                self.guard.keypoints_kept(int(pose.xy.nbytes + pose.scores.nbytes))
            self.step(pose, time_s, view, hazards=hazards)
            seen += 1
            if progress and seen % 10 == 0:
                progress(
                    min(99.0, 100.0 * seen / total_frames),
                    f"{time_s:.1f} s analysed, {self.guard.ledger.frames_retained} frames kept",
                )
        return self._finish(fps, source=str(Path(path).name))

    def lead_time_s(self) -> float | None:
        """Seconds between the first call going out and the person being upright.

        Measured from the *call*, not from the internal state that triggered it,
        because what matters to a ward is how long they had after the alarm
        sounded. A call that fires during the swing of the legs out of bed earns
        its lead time just as much as one that fires during the lean.
        """
        call = self.ladder.first_call()
        if call is None:
            return None
        upright = self.first_sustained_upright(after_s=call.time_s)
        return None if upright is None else float(upright - call.time_s)

    def first_sustained_upright(self, *, after_s: float = 0.0, hold_s: float = 0.6) -> float | None:
        """When the person was really on their feet, not when a noisy frame said so.

        A single frame of standing during the lean is keypoint noise pushing the
        hip height over a threshold. Requiring the state to hold for half a second
        before counting it stops that noise from eating the lead time the product
        is measured on -- which it did, by about a second, in the first version.
        """
        started: float | None = None
        for reading in self.exits.timeline:
            if reading.time_s <= after_s:
                continue
            if reading.state in (STANDING, WALKING):
                if started is None:
                    started = reading.time_s
                elif reading.time_s - started >= hold_s:
                    return started
            else:
                started = None
        return None

    # -- internals ----------------------------------------------------------
    def _seat_position(self) -> np.ndarray | None:
        for moment in reversed(self.moments):
            if moment.exit_reading and moment.exit_reading.support:
                for state in reversed(self.exits.history):
                    if state.floor_xy is not None:
                        return state.floor_xy
        for state in reversed(self.exits.history):
            if state.floor_xy is not None:
                return state.floor_xy
        return None

    def _finish(self, fps: float, source: str) -> Analysis:
        return Analysis(
            room=self.room,
            fps=fps,
            moments=self.moments,
            calls=[c.to_dict() for c in self.ladder.calls],
            hazards=self.hazards.last,
            privacy=self.guard.to_dict(),
            lead_time_s=self.lead_time_s(),
            source=source,
        )


# ---------------------------------------------------------------------------
# RunRecord assembly
# ---------------------------------------------------------------------------


def _headline_metrics(analysis: Analysis) -> dict[str, Any]:
    states = [m.risk.state for m in analysis.moments]
    peak = analysis.peak
    gait_scores = [m.gait.score for m in analysis.moments if m.gait and m.gait.scored]
    return {
        "duration_s": round(analysis.moments[-1].time_s, 2) if analysis.moments else 0.0,
        "samples": len(analysis.moments),
        "peak_state": peak.risk.state if peak else "settled",
        "peak_headline": peak.risk.headline if peak else "Settled",
        "lead_time_s": None if analysis.lead_time_s is None else round(analysis.lead_time_s, 2),
        "calls": len(analysis.calls),
        "highest_rung": (
            max((c["rung"] for c in analysis.calls), key=lambda r: ["none", "nudge", "station", "urgent", "maintenance"].index(r))
            if analysis.calls else "none"
        ),
        "seconds_rising_soon": round(states.count(RISING_SOON) / max(1e-9, len(states)) * (analysis.moments[-1].time_s if analysis.moments else 0), 2),
        "peak_gait_score": round(max(gait_scores), 3) if gait_scores else None,
        "view": summarise(analysis.view_reports if hasattr(analysis, "view_reports") else []),
    }


def build_record(analysis: Analysis, pipeline: Pipeline, *, params: dict[str, Any] | None = None) -> RunRecord:
    """Flatten an Analysis into the shared RunRecord shape every product returns."""
    record = RunRecord(product=PRODUCT, params=dict(params or {}))
    record.input = {"source": analysis.source, "room": analysis.room.room, "fps": round(analysis.fps, 2)}
    peak = analysis.peak

    metrics = _headline_metrics(analysis)
    metrics["view"] = summarise(pipeline.view_reports)
    metrics["privacy"] = analysis.privacy
    metrics["floor_frame"] = pipeline.frame.to_dict()
    record.metrics = metrics

    record.results = [
        {
            "kind": "risk",
            "state": peak.risk.state if peak else "settled",
            "headline": peak.risk.headline if peak else "Settled",
            "at_s": round(peak.time_s, 2) if peak else 0.0,
            "reasons": peak.risk.reasons if peak else [],
            "hazards": peak.risk.hazard_reasons if peak else [],
            "certainty": peak.risk.certainty if peak else "observed",
        },
        {"kind": "calls", "calls": analysis.calls},
        {"kind": "hazards", **(analysis.hazards.to_dict() if analysis.hazards else {"hazards": []})},
        {
            "kind": "timeline",
            "samples": [m.to_dict() for m in analysis.moments],
        },
        {
            "kind": "keypoint-ledger",
            "note": "the camera's entire output for this instant, as the device keeps it",
            "frame": (peak.pose.to_dict(analysis.room.thresholds.keypoint_score_min) if peak and peak.pose else None),
        },
    ]

    for note in pipeline.guard.ledger.refusals:
        record.refuse("PRIVACY_REFUSED", note)
    unusable = [r for r in pipeline.view_reports if not r.usable]
    if unusable:
        worst = unusable[0]
        record.refuse(
            "VIEW_UNUSABLE",
            f"{worst.state}: {worst.detail}",
            remedy=worst.remedy,
            frames=len(unusable),
        )
    return record


def attach_evidence(record: RunRecord, analysis: Analysis, pipeline: Pipeline) -> None:
    """Draw the two evidence cards. Both are synthetic; see `privacy.py`."""
    peak = analysis.peak
    if peak is not None:
        alert = peak.risk.state in (RISING_SOON, UNSTEADY, FLOOR)
        card = pose_card(
            peak.pose,
            peak.risk.headline,
            peak.risk.reasons,
            alert=alert,
            footer=(
                f"drawn from {17} keypoints; "
                f"{pipeline.guard.ledger.frames_retained} camera frames retained"
            ),
        )
        uri = pipeline.guard.emit("risk-state.png", encode_png(card), Provenance.SYNTHETIC)
        record.add_evidence(
            Evidence(
                label="risk state",
                kind="overlay",
                uri=uri,
                timestamp_ms=peak.time_s * 1000.0,
                caption=f"{peak.risk.headline} at {peak.time_s:.1f} s, drawn from keypoints only",
                metrics={"state": peak.risk.state},
            )
        )

    last_floor = next(
        (m for m in reversed(analysis.moments) if m.exit_reading and m.exit_reading.lean_offset is not None),
        None,
    )
    position = None
    if last_floor is not None:
        for state in reversed(pipeline.exits.history):
            if state.floor_xy is not None:
                position = state.floor_xy
                break
    plan = plan_card(
        pipeline.frame,
        analysis.room.zones,
        person_xy=position,
        corridor=(analysis.hazards.corridor if analysis.hazards else None),
        hazards=[h.floor_xy for h in (analysis.hazards.hazards if analysis.hazards else []) if h.floor_xy],
        title=f"{analysis.room.room}: plan view",
    )
    uri = pipeline.guard.emit("room-plan.png", encode_png(plan), Provenance.SYNTHETIC)
    record.add_evidence(
        Evidence(
            label="room plan",
            kind="chart",
            uri=uri,
            caption="the room as the ward drew it, with the walking route and anything on it",
        )
    )
    # Re-read the ledger now that the evidence has been written, so the privacy
    # panel in the UI counts the bytes this run actually produced rather than the
    # bytes it had produced halfway through.
    record.metrics["privacy"] = pipeline.guard.to_dict()


def analyse_track(path: str | Path, *, sink: Any = None, progress=None) -> tuple[RunRecord, Analysis]:
    frames, room, meta = load_track(path)
    guard = PrivacyGuard(room.privacy_mode, sink=sink)
    pipeline = Pipeline(room, guard=guard)
    with recording(RunRecord(product=PRODUCT)):
        analysis = pipeline.run_track(
            frames, meta["fps"], progress=progress, view_hints=meta.get("view_hints", {})
        )
    analysis.source = f"{meta['name']} (pose track)"
    record = build_record(analysis, pipeline, params={"input_kind": "pose track", **meta.get("truth", {})})
    attach_evidence(record, analysis, pipeline)
    return record, analysis


def analyse_video(
    path: str | Path,
    room: RoomConfig,
    *,
    sink: Any = None,
    progress=None,
    estimator: PoseEstimator | None = None,
) -> tuple[RunRecord, Analysis]:
    guard = PrivacyGuard(room.privacy_mode, sink=sink)
    pipeline = Pipeline(room, guard=guard, estimator=estimator)
    record = RunRecord(product=PRODUCT)
    with recording(record):
        analysis = pipeline.run_video(path, progress=progress)
    built = build_record(analysis, pipeline, params={"input_kind": "video"})
    built.stages = record.stages
    if pipeline.estimator is not None:
        built.metrics["models"] = pipeline.estimator.info()
        built.metrics["pose_stats"] = pipeline.estimator.stats.to_dict()
    attach_evidence(built, analysis, pipeline)
    return built, analysis
