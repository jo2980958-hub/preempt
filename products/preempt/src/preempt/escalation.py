"""The escalation ladder: a nudge, a call, an emergency.

Three rungs, because a ward already works in three and a system that invents a
fourth gets ignored:

* **nudge** - a spoken prompt in the room. "Please wait, someone is coming." No
  radio traffic, no interruption. Most of the time this is the whole
  intervention: a person who is told someone is coming usually waits.
* **station** - the call arrives at the nurses' station with the reason and the
  room. Someone walks down.
* **urgent** - someone is on the floor. This one goes out immediately and is not
  subject to any hold.

Rules that keep it from becoming noise
--------------------------------------
A nudge that nobody acknowledges becomes a station call after
`nudge_to_station_s`. That is what makes the quiet rung safe: nothing is lost if
the patient ignores it.

A hazard on the route raises the rung by one. A person getting out of bed with
their walking frame across the room is not the same situation as one with it
beside them, and the ladder should say so.

Acknowledgement puts the same rung on a cooldown but never blocks a higher one.
Staff silencing a nudge cannot silence a person on the floor.

A view the engine cannot use raises no call at all, except a maintenance notice
so someone fixes the camera. Silently continuing would be the worst failure this
system could have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Thresholds
from .risk import FLOOR, PAUSED, RISING_SOON, UNSTEADY, UNUSABLE, RiskState

NONE = "none"
NUDGE = "nudge"
STATION = "station"
URGENT = "urgent"
MAINTENANCE = "maintenance"

RUNGS = (NONE, NUDGE, STATION, URGENT)
RUNG_RANK = {name: i for i, name in enumerate(RUNGS)}

BASE_RUNG = {
    RISING_SOON: NUDGE,
    UNSTEADY: STATION,
    FLOOR: URGENT,
}

WORDING = {
    NUDGE: "In-room prompt: please wait, a member of staff is on the way",
    STATION: "Call the station",
    URGENT: "Urgent: someone is on the floor",
    MAINTENANCE: "Camera needs attention",
}


@dataclass
class Call:
    """One thing that was sent to somebody, and why."""

    time_s: float
    rung: str
    wording: str
    risk_state: str
    reasons: list[str] = field(default_factory=list)
    raised_by_hazard: bool = False
    acknowledged_at: float | None = None

    @property
    def rank(self) -> int:
        return RUNG_RANK.get(self.rung, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": round(self.time_s, 3),
            "rung": self.rung,
            "wording": self.wording,
            "risk_state": self.risk_state,
            "reasons": list(self.reasons),
            "raised_by_hazard": self.raised_by_hazard,
            "acknowledged_at": (
                None if self.acknowledged_at is None else round(self.acknowledged_at, 3)
            ),
        }


def _raise_one(rung: str) -> str:
    """One rung up, but never into urgent.

    Urgent means a person is on the floor and somebody should run. A walking
    frame parked out of reach is a good reason to send someone sooner; it is not
    a reason to tell the ward that a patient has fallen when they have not.
    """
    if RUNG_RANK[rung] >= RUNG_RANK[STATION]:
        return rung
    return RUNGS[RUNG_RANK[rung] + 1]


class EscalationLadder:
    """Turns a stream of risk states into the smallest set of calls that covers them."""

    def __init__(self, thresholds: Thresholds) -> None:
        self.t = thresholds
        self.calls: list[Call] = []
        self.current: str = NONE
        self._since: float | None = None
        self._acknowledged: dict[str, float] = {}
        self._maintenance_sent = False

    # -- staff actions ------------------------------------------------------
    def acknowledge(self, time_s: float) -> Call | None:
        """A member of staff has seen the current call."""
        if not self.calls:
            return None
        call = self.calls[-1]
        call.acknowledged_at = time_s
        self._acknowledged[call.rung] = time_s
        return call

    # -- the step -----------------------------------------------------------
    def update(self, risk: RiskState, *, has_hazard: bool) -> Call | None:
        """Returns the call raised at this instant, or None."""
        if risk.state == PAUSED:
            # Not even a maintenance notice. Staff know: they pressed the button.
            self._reset()
            return None
        if risk.state == UNUSABLE:
            self._reset()
            if self._maintenance_sent:
                return None
            self._maintenance_sent = True
            return self._record(
                Call(
                    time_s=risk.time_s,
                    rung=MAINTENANCE,
                    wording=WORDING[MAINTENANCE],
                    risk_state=risk.state,
                    reasons=[r for r in risk.reasons if r],
                )
            )
        self._maintenance_sent = False

        wanted = BASE_RUNG.get(risk.state, NONE)
        by_hazard = False
        if wanted != NONE and has_hazard:
            raised = _raise_one(wanted)
            by_hazard = raised != wanted
            wanted = raised

        if wanted == NONE:
            self._reset()
            return None

        # An unanswered nudge becomes a station call.
        if (
            self.current == NUDGE
            and self._since is not None
            and risk.time_s - self._since >= self.t.nudge_to_station_s
            and not self._is_acknowledged(NUDGE, risk.time_s)
        ):
            wanted = max(wanted, STATION, key=lambda r: RUNG_RANK[r])

        if RUNG_RANK[wanted] <= RUNG_RANK[self.current]:
            return None
        if wanted != URGENT and self._is_acknowledged(wanted, risk.time_s):
            return None

        self.current = wanted
        self._since = risk.time_s
        return self._record(
            Call(
                time_s=risk.time_s,
                rung=wanted,
                wording=WORDING[wanted],
                risk_state=risk.state,
                reasons=list(risk.reasons) + (list(risk.hazard_reasons) if has_hazard else []),
                raised_by_hazard=by_hazard,
            )
        )

    # -- internals ----------------------------------------------------------
    def _is_acknowledged(self, rung: str, time_s: float) -> bool:
        seen = self._acknowledged.get(rung)
        return seen is not None and (time_s - seen) < self.t.acknowledge_cooldown_s

    def _reset(self) -> None:
        self.current = NONE
        self._since = None

    def _record(self, call: Call) -> Call:
        self.calls.append(call)
        return call

    # -- results ------------------------------------------------------------
    def first_call(self) -> Call | None:
        return next((c for c in self.calls if c.rung in (NUDGE, STATION, URGENT)), None)

    def summary(self) -> dict[str, Any]:
        return {
            "calls": [c.to_dict() for c in self.calls],
            "highest_rung": max(
                (c.rung for c in self.calls if c.rung in RUNGS),
                key=lambda r: RUNG_RANK[r],
                default=NONE,
            ),
            "first_call_s": (round(self.first_call().time_s, 3) if self.first_call() else None),
        }
