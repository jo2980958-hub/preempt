"""The synthetic sequences, where the answer was written into the sequence.

These are the tests that would fail if the product stopped working. Each one
names a situation a ward would recognise and asserts what should happen, in
seconds.
"""

from __future__ import annotations

import pytest
from preempt.pipeline import Pipeline, analyse_track
from preempt.privacy import PrivacyGuard
from preempt.risk import FLOOR, RISING_SOON, UNSTEADY
from preempt.synth import QUIET, SCENARIOS, make

CALL_RUNGS = ("nudge", "station", "urgent")


def run(name: str, *, seed: int = 20261026, loops: int = 1):
    sequence = make(name, seed=seed, loops=loops)
    pipeline = Pipeline(sequence.room, guard=PrivacyGuard("strict"))
    analysis = pipeline.run_track(sequence.frames, sequence.fps, view_hints=sequence.view_hints)
    return sequence, pipeline, analysis


def called(analysis) -> bool:
    return any(c["rung"] in CALL_RUNGS for c in analysis.calls)


# --- the positives ---------------------------------------------------------


@pytest.mark.parametrize("name", ["bed-exit-steady", "bed-exit-unsteady", "bed-exit-to-floor"])
def test_a_bed_exit_is_called_before_the_person_is_on_their_feet(name):
    sequence, _pipeline, analysis = run(name)
    assert called(analysis), "no call was raised on a bed exit"
    assert analysis.lead_time_s is not None
    assert analysis.lead_time_s > 1.5, (
        f"only {analysis.lead_time_s:.1f} s of warning; the product's claim is seconds"
    )
    truth_standing = sequence.truth["events"]["standing"]
    first_call = analysis.calls[0]["time_s"]
    assert first_call < truth_standing, (
        f"called at {first_call:.1f} s, after the person stood at {truth_standing:.1f} s"
    )


def test_the_call_carries_reasons_a_nurse_could_act_on():
    _, _, analysis = run("bed-exit-steady")
    call = analysis.calls[0]
    assert call["reasons"], "a call with no reasons is an alarm, not information"
    assert all(isinstance(r, str) and len(r) > 8 for r in call["reasons"])


def test_someone_on_the_floor_reaches_the_urgent_rung():
    sequence, _, analysis = run("bed-exit-to-floor")
    assert analysis.peak.risk.state == FLOOR
    assert any(c["rung"] == "urgent" for c in analysis.calls)
    truth = sequence.truth["events"]["on_floor"]
    urgent = next(c["time_s"] for c in analysis.calls if c["rung"] == "urgent")
    assert urgent >= truth, "detected before it happened, which means the test is wrong"
    assert urgent - truth < 4.0, f"took {urgent - truth:.1f} s to notice someone on the floor"


def test_an_unsteady_walk_is_called_and_a_steady_one_is_not():
    _, _, unsteady = run("unsteady-walk")
    _, _, steady = run("steady-walk")
    assert unsteady.peak.risk.state == UNSTEADY
    assert called(unsteady)
    assert steady.peak.risk.state not in (UNSTEADY, RISING_SOON, FLOOR)
    assert not called(steady)


def test_the_unsteady_call_says_what_was_unsteady():
    _, _, analysis = run("unsteady-walk")
    peak = analysis.peak
    assert peak.risk.gait_score is not None and peak.risk.gait_score >= 0.55
    assert any("sway" in r for r in peak.risk.reasons), peak.risk.reasons


# --- the negatives ---------------------------------------------------------


@pytest.mark.parametrize("name", list(QUIET))
def test_nothing_is_called_when_nothing_is_happening(name):
    _, _, analysis = run(name, loops=4)
    assert not called(analysis), (
        f"{name} raised {[c['rung'] for c in analysis.calls]} with nothing happening"
    )


def test_turning_over_in_bed_is_not_getting_out_of_bed():
    _, pipeline, analysis = run("settled-turning-over", loops=4)
    assert pipeline.exits.first_time_in("preparing to rise") is None
    assert analysis.peak.risk.state in ("settled", "watch")


def test_sitting_up_and_lying_back_down_is_not_a_bed_exit():
    """The sign test: lying back drives the same lean measure the other way."""
    _, pipeline, analysis = run("sat-up-and-lay-back", loops=4)
    assert pipeline.exits.first_time_in("preparing to rise") is None
    assert not called(analysis)


def test_sitting_down_in_a_chair_is_not_getting_up():
    """Lowering onto a seat leans over the feet with bent knees, like a rise does."""
    for seed in range(5):
        _, _pipeline, analysis = run("sit-down-in-chair", seed=20261026 + seed * 977)
        assert not called(analysis), f"seed {seed}: called at {analysis.calls[0]['time_s']:.1f} s"


def test_standing_up_and_sitting_straight_back_down_is_one_call_not_two():
    for seed in range(5):
        sequence, _, analysis = run("chair-stand-and-sit-back", seed=20261026 + seed * 977)
        events = sequence.truth["events"]
        times = [c["time_s"] for c in analysis.calls if c["rung"] in CALL_RUNGS]
        assert times and times[0] < events["standing"], f"seed {seed}: rise not called in time"
        late = [t for t in times if t >= events["sit_down_starts"]]
        assert not late, f"seed {seed}: called as the person sat back down, at {late}"


# --- the honesty rail ------------------------------------------------------


def test_a_blocked_lens_produces_a_maintenance_notice_and_no_risk_claim():
    _, _, analysis = run("curtain-drawn")
    rungs = [c["rung"] for c in analysis.calls]
    assert "maintenance" in rungs
    assert not called(analysis), "a clinical call was raised on a view we cannot use"
    blocked = [m for m in analysis.moments if m.risk.state == "view unusable"]
    assert blocked, "the blocked frames were not marked unusable"
    assert blocked[0].risk.certainty == "cannot assess"
    assert blocked[0].risk.reasons[-1], "an unusable view must name its remedy"


def test_the_run_record_records_the_unusable_view_as_a_refusal(tmp_path):
    sequence = make("curtain-drawn")
    path = sequence.save(tmp_path / "curtain.json")
    record, _ = analyse_track(path)
    codes = [r.code for r in record.refusals]
    assert "VIEW_UNUSABLE" in codes
    assert record.metrics["view"]["usable_fraction"] < 1.0


# --- the whole set ---------------------------------------------------------


def test_every_scenario_produces_a_serialisable_record(tmp_path):
    for name in sorted(SCENARIOS):
        sequence = make(name)
        path = sequence.save(tmp_path / f"{name}.json")
        record, _ = analyse_track(path)
        text = record.to_json()
        assert len(text) > 500
        assert record.metrics["privacy"]["clean"] is True
        assert record.metrics["peak_state"] in (
            "settled",
            "watch",
            "rising soon",
            "unsteady",
            "on the floor",
            "view unusable",
        )


# --- the cost of privacy ---------------------------------------------------


def test_a_paused_camera_is_reported_as_blind_by_choice_not_as_a_fault():
    """Falls cluster around exactly the activities a ward pauses the camera for."""
    _, _, analysis = run("personal-care-pause")
    assert not called(analysis), "a call was raised while the camera was paused"
    assert not analysis.calls, "even a maintenance notice is wrong here: staff paused it"
    paused = [m for m in analysis.moments if m.risk.state == "paused"]
    assert paused, "the paused window was not marked"
    assert paused[0].risk.certainty == "deliberately not watching"
    assert any("unknowable" in r for r in paused[0].risk.reasons)


def test_the_record_says_how_long_it_was_blind_and_that_the_window_is_unknowable(tmp_path):
    sequence = make("personal-care-pause")
    path = sequence.save(tmp_path / "pause.json")
    record, _ = analyse_track(path)
    view = record.metrics["view"]
    assert view["blind_by_choice_s"] > 5.0
    assert view["faulty_s"] == 0.0, "a deliberate pause is not a fault"
    refusal = next(r for r in record.refusals if r.code == "BLIND_BY_CHOICE")
    assert "unknowable" in refusal.message
    assert "VIEW_UNUSABLE" not in {r.code for r in record.refusals}
