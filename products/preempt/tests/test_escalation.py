"""The escalation ladder. The rules a ward would ask about, one test each."""

from __future__ import annotations

from preempt.config import Thresholds
from preempt.escalation import MAINTENANCE, NUDGE, STATION, URGENT, EscalationLadder
from preempt.risk import FLOOR, RISING_SOON, SETTLED, UNSTEADY, UNUSABLE, RiskState


def state(kind: str, time_s: float) -> RiskState:
    return RiskState(time_s=time_s, state=kind, headline=kind, reasons=["a reason"])


def test_rising_soon_raises_the_quiet_rung_first():
    ladder = EscalationLadder(Thresholds())
    call = ladder.update(state(RISING_SOON, 1.0), has_hazard=False)
    assert call is not None
    assert call.rung == NUDGE
    assert call.reasons == ["a reason"]


def test_a_hazard_on_the_route_raises_the_rung_by_one():
    ladder = EscalationLadder(Thresholds())
    call = ladder.update(state(RISING_SOON, 1.0), has_hazard=True)
    assert call.rung == STATION
    assert call.raised_by_hazard is True


def test_a_hazard_never_raises_anything_into_urgent():
    """Urgent means somebody is on the floor. A bag in the way is not that."""
    ladder = EscalationLadder(Thresholds())
    call = ladder.update(state(UNSTEADY, 1.0), has_hazard=True)
    assert call.rung == STATION


def test_on_the_floor_goes_straight_to_urgent():
    ladder = EscalationLadder(Thresholds())
    call = ladder.update(state(FLOOR, 1.0), has_hazard=False)
    assert call.rung == URGENT


def test_an_unanswered_nudge_becomes_a_station_call():
    thresholds = Thresholds(nudge_to_station_s=10.0)
    ladder = EscalationLadder(thresholds)
    assert ladder.update(state(RISING_SOON, 0.0), has_hazard=False).rung == NUDGE
    assert ladder.update(state(RISING_SOON, 5.0), has_hazard=False) is None
    late = ladder.update(state(RISING_SOON, 11.0), has_hazard=False)
    assert late is not None and late.rung == STATION


def test_an_acknowledged_nudge_does_not_escalate_by_itself():
    thresholds = Thresholds(nudge_to_station_s=10.0, acknowledge_cooldown_s=90.0)
    ladder = EscalationLadder(thresholds)
    ladder.update(state(RISING_SOON, 0.0), has_hazard=False)
    ladder.acknowledge(2.0)
    assert ladder.update(state(RISING_SOON, 11.0), has_hazard=False) is None


def test_acknowledgement_cannot_silence_someone_on_the_floor():
    ladder = EscalationLadder(Thresholds())
    ladder.update(state(RISING_SOON, 0.0), has_hazard=False)
    ladder.acknowledge(1.0)
    urgent = ladder.update(state(FLOOR, 2.0), has_hazard=False)
    assert urgent is not None and urgent.rung == URGENT


def test_the_ladder_does_not_repeat_a_rung_it_is_already_on():
    ladder = EscalationLadder(Thresholds())
    assert ladder.update(state(UNSTEADY, 1.0), has_hazard=False).rung == STATION
    assert ladder.update(state(UNSTEADY, 2.0), has_hazard=False) is None
    assert ladder.update(state(UNSTEADY, 3.0), has_hazard=False) is None


def test_settling_back_down_clears_the_ladder_so_a_later_event_calls_again():
    ladder = EscalationLadder(Thresholds())
    ladder.update(state(RISING_SOON, 1.0), has_hazard=False)
    assert ladder.update(state(SETTLED, 5.0), has_hazard=False) is None
    again = ladder.update(state(RISING_SOON, 30.0), has_hazard=False)
    assert again is not None and again.rung == NUDGE


def test_an_unusable_view_raises_a_maintenance_notice_and_no_clinical_call():
    ladder = EscalationLadder(Thresholds())
    call = ladder.update(state(UNUSABLE, 1.0), has_hazard=False)
    assert call is not None
    assert call.rung == MAINTENANCE
    assert ladder.update(state(UNUSABLE, 2.0), has_hazard=False) is None
    assert all(c.rung == MAINTENANCE for c in ladder.calls)


def test_the_summary_names_the_highest_rung_and_the_first_call():
    ladder = EscalationLadder(Thresholds())
    ladder.update(state(RISING_SOON, 1.0), has_hazard=False)
    ladder.update(state(FLOOR, 9.0), has_hazard=False)
    summary = ladder.summary()
    assert summary["highest_rung"] == URGENT
    assert summary["first_call_s"] == 1.0
    assert len(summary["calls"]) == 2
