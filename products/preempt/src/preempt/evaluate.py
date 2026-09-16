"""The evaluation harness: lead time, detection rate, false alarms, failures.

Four numbers are reported, and each one is chosen because it is the number a
ward would ask for rather than the number that flatters the system.

**Lead time.** Seconds between the first call going out and the person being
upright on their feet, over the bed and chair exits only, because those are the
sequences where being upright is a thing that happens. This is the whole product.
A fall-detection system's lead time is negative by definition.

**Detection rate.** The fraction of sequences that should have raised a call and
did. Counted per sequence, not per frame: a ward gets one call, not nine hundred.

**False alarms per bed-night.** Calls raised on sequences where nothing was
happening, scaled to twelve hours. Scaled, and said to be scaled -- the synthetic
negatives are a few minutes long and extrapolating them to a night is an
assumption, not a measurement, and `docs/evaluation.md` says so in those words.

**Failure cases.** Every sequence that got the wrong answer is listed by name
with what it did instead. A table of successes is marketing.

Two sources
-----------
`run_synthetic` evaluates the decision layer on sequences whose answers were
written into them. It holds perception constant and measures the reasoning.

`run_urfall` evaluates the whole pipeline, pose estimation included, on the UR
Fall Detection Dataset: real people, real rooms, real falls, with per-frame
ground truth of whether the person is on the floor. It is **CC BY-NC-SA 4.0,
non-commercial academic use**, so it is downloaded at evaluation time and no
frame or derived image from it is ever committed to this repository.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .pipeline import Pipeline
from .privacy import PrivacyGuard
from .risk import FLOOR, RISING_SOON, UNSTEADY
from .synth import SCENARIOS, make

CALL_RUNGS = ("nudge", "station", "urgent")


@dataclass
class Outcome:
    """What one sequence did, against what it should have done."""

    name: str
    should_call: bool
    called: bool
    peak_state: str
    expected_state: str
    lead_time_s: float | None
    calls: int
    duration_s: float
    correct: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "should_call": self.should_call,
            "called": self.called,
            "peak_state": self.peak_state,
            "expected_state": self.expected_state,
            "lead_time_s": None if self.lead_time_s is None else round(self.lead_time_s, 2),
            "calls": self.calls,
            "duration_s": round(self.duration_s, 1),
            "correct": self.correct,
            "note": self.note,
        }


@dataclass
class Report:
    """The whole evaluation, in the shape `docs/evaluation.md` quotes."""

    source: str
    outcomes: list[Outcome] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def positives(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.should_call]

    @property
    def negatives(self) -> list[Outcome]:
        return [o for o in self.outcomes if not o.should_call]

    def summary(self) -> dict[str, Any]:
        positives, negatives = self.positives, self.negatives
        detected = [o for o in positives if o.called]
        exits = [o for o in detected if o.expected_state == RISING_SOON]
        leads = [o.lead_time_s for o in exits if o.lead_time_s is not None]
        false_alarms = [o for o in negatives if o.called]
        quiet_hours = sum(o.duration_s for o in negatives) / 3600.0
        return {
            "source": self.source,
            "sequences": len(self.outcomes),
            "positives": len(positives),
            "negatives": len(negatives),
            "detection_rate": round(len(detected) / len(positives), 3) if positives else None,
            "lead_time_s_bed_and_chair_exits": {
                "n": len(leads),
                "median": round(statistics.median(leads), 2) if leads else None,
                "mean": round(statistics.fmean(leads), 2) if leads else None,
                "min": round(min(leads), 2) if leads else None,
                "max": round(max(leads), 2) if leads else None,
            },
            "false_alarms": len(false_alarms),
            "quiet_hours_observed": round(quiet_hours, 4),
            "false_alarms_per_bed_night": (
                round(len(false_alarms) / quiet_hours * 12.0, 2) if quiet_hours > 0 else None
            ),
            "failures": [o.to_dict() for o in self.outcomes if not o.correct],
            **self.extra,
        }

    def to_json(self) -> str:
        return json.dumps(
            {"summary": self.summary(), "outcomes": [o.to_dict() for o in self.outcomes]},
            indent=2,
        )


def _grade(name: str, analysis, truth: dict[str, Any]) -> Outcome:
    should_call = bool(truth.get("should_call"))
    expected = truth.get("expect_state", "")
    called = any(c["rung"] in CALL_RUNGS for c in analysis.calls)
    peak = analysis.peak.risk.state if analysis.peak else "settled"
    duration = analysis.moments[-1].time_s if analysis.moments else 0.0

    note = ""
    if should_call and not called:
        correct, note = False, "no call was raised on a sequence that needed one"
    elif not should_call and called:
        correct, note = False, f"called at {analysis.calls[0]['time_s']:.1f} s with nothing happening"
    elif should_call and expected and peak != expected:
        correct = peak in (RISING_SOON, UNSTEADY, FLOOR) and expected in (RISING_SOON, UNSTEADY, FLOOR)
        if not correct:
            note = f"called, but reached {peak!r} where {expected!r} was expected"
        else:
            note = f"called; reached {peak!r} rather than {expected!r}, both actionable"
    else:
        correct = True
    return Outcome(
        name=name,
        should_call=should_call,
        called=called,
        peak_state=peak,
        expected_state=expected,
        lead_time_s=analysis.lead_time_s,
        calls=len(analysis.calls),
        duration_s=duration,
        correct=correct,
        note=note,
    )


def run_synthetic(*, seeds: int = 5, fps: float = 30.0, quiet_loops: int = 12) -> Report:
    """Every scenario at several noise seeds, so one lucky draw cannot carry it."""
    report = Report(
        source=(
            f"synthetic, {len(SCENARIOS)} scenarios x {seeds} noise seeds, "
            f"quiet scenarios looped {quiet_loops}x"
        )
    )
    for name in sorted(SCENARIOS):
        for seed in range(seeds):
            sequence = make(name, fps=fps, seed=20261026 + seed * 977, loops=quiet_loops)
            pipeline = Pipeline(sequence.room, guard=PrivacyGuard("strict"))
            analysis = pipeline.run_track(sequence.frames, sequence.fps)
            label = name if seeds == 1 else f"{name}#{seed}"
            report.outcomes.append(_grade(label, analysis, sequence.truth))
    report.extra["thresholds"] = make(next(iter(sorted(SCENARIOS)))).room.thresholds.to_dict()
    return report


# ---------------------------------------------------------------------------
# UR Fall Detection Dataset
# ---------------------------------------------------------------------------

URFALL_BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data"
URFALL_LICENCE = (
    "UR Fall Detection Dataset, Kwolek and Kepski, University of Rzeszow. "
    "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International, "
    "intended for non-commercial academic use. Downloaded at evaluation time; "
    "no frame or derived image from it is committed to this repository."
)


def urfall_labels(csv_path: Path) -> dict[str, dict[int, int]]:
    """Per-frame ground truth: 1 lying on the ground, -1 not, 0 mid-fall."""
    labels: dict[str, dict[int, int]] = {}
    for line in csv_path.read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        labels.setdefault(parts[0], {})[int(parts[1])] = int(parts[2])
    return labels


def _first_lying_second(frames: dict[int, int], fps: float) -> float | None:
    lying = sorted(i for i, label in frames.items() if label == 1)
    return (lying[0] - 1) / fps if lying else None


def run_urfall(
    root: Path,
    *,
    room_json: Path,
    fps: float = 30.0,
    limit: int | None = None,
) -> Report:
    """Whole-pipeline evaluation on real footage, pose estimation included.

    `root` holds one directory per sequence of extracted RGB frames, plus
    `urfall-cam0-falls.csv` and `urfall-cam0-adls.csv`. `eval/fetch_urfall.sh`
    puts them there. The room must be set up for this dataset's camera; see
    `eval/urfall-room.json` and the caveats in `docs/evaluation.md`.
    """
    from .config import RoomConfig
    from .pose import PoseEstimator

    room = RoomConfig.load(room_json)
    labels: dict[str, dict[int, int]] = {}
    for csv_name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        path = root / csv_name
        if path.is_file():
            labels.update(urfall_labels(path))

    report = Report(source="UR Fall Detection Dataset (cam0 RGB)")
    report.extra["licence"] = URFALL_LICENCE
    estimator = PoseEstimator()
    sequences = sorted(d for d in root.iterdir() if d.is_dir())
    if limit:
        sequences = sequences[:limit]

    detections: list[dict[str, Any]] = []
    for directory in sequences:
        name = directory.name
        images = sorted(directory.glob("*.png")) + sorted(directory.glob("*.jpg"))
        if not images:
            continue
        pipeline = Pipeline(room, guard=PrivacyGuard("strict"), estimator=estimator)
        analysis = pipeline.run_images(images, fps=fps)
        truth_frames = labels.get(name, {})
        truth_s = _first_lying_second(truth_frames, fps)
        floor_moment = next(
            (m.time_s for m in analysis.moments if m.risk.state == FLOOR), None
        )
        is_fall = name.startswith("fall")
        detections.append(
            {
                "sequence": name,
                "frames": len(images),
                "is_fall": is_fall,
                "ground_truth_on_floor_s": None if truth_s is None else round(truth_s, 2),
                "detected_on_floor_s": None if floor_moment is None else round(floor_moment, 2),
                "delay_s": (
                    round(floor_moment - truth_s, 2)
                    if (floor_moment is not None and truth_s is not None)
                    else None
                ),
                "peak_state": analysis.peak.risk.state if analysis.peak else "settled",
                "calls": [c["rung"] for c in analysis.calls],
            }
        )
        report.outcomes.append(
            Outcome(
                name=name,
                should_call=is_fall,
                called=any(c["rung"] in CALL_RUNGS for c in analysis.calls),
                peak_state=analysis.peak.risk.state if analysis.peak else "settled",
                expected_state=FLOOR if is_fall else "",
                lead_time_s=analysis.lead_time_s,
                calls=len(analysis.calls),
                duration_s=len(images) / fps,
                correct=(floor_moment is not None) == is_fall,
                note="" if (floor_moment is not None) == is_fall else (
                    "no on-the-floor state on a fall sequence"
                    if is_fall
                    else "on-the-floor state on an activity-of-daily-living sequence"
                ),
            )
        )
    report.extra["per_sequence"] = detections
    delays = [d["delay_s"] for d in detections if d["delay_s"] is not None]
    if delays:
        report.extra["detection_delay_s"] = {
            "n": len(delays),
            "median": round(statistics.median(delays), 2),
            "mean": round(statistics.fmean(delays), 2),
            "min": round(min(delays), 2),
            "max": round(max(delays), 2),
        }
    return report
