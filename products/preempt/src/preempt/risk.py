"""The risk state, and the reasons that raised it.

Preempt does not predict falls. It cannot, and a product that claims to would be
lying to a ward that has no way to check. What it does is name the situation it
can see and list the measurements behind that name, so a nurse can decide in two
seconds whether it is worth walking down the corridor.

    settled          nothing to do
    watch            something changed, no call
    rising soon      the movement that precedes standing, while still supported
    unsteady         upright, and the gait says they are not safe on their feet
    on the floor     geometrically on the floor, and it has held

The words on the left are what appears on the screen. The list of reasons under
each is generated from the measurements, never from a template, so if the engine
cannot say why, the state does not go up.

One deliberate asymmetry: a hazard on the route never raises a state on its own,
because a bag on the floor is not an emergency while the person is asleep. It
raises the *escalation* one rung when a state is already in play, which is how a
ward would treat it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Thresholds
from .exits import ON_FLOOR, PREPARING, RISING, STANDING, WALKING, ExitReading
from .gait import GaitReport
from .hazards import HazardReport
from .view import ViewReport

SETTLED = "settled"
WATCH = "watch"
RISING_SOON = "rising soon"
UNSTEADY = "unsteady"
FLOOR = "on the floor"
UNUSABLE = "view unusable"

RISK_ORDER = (UNUSABLE, SETTLED, WATCH, RISING_SOON, UNSTEADY, FLOOR)
RISK_RANK = {name: i for i, name in enumerate((SETTLED, WATCH, RISING_SOON, UNSTEADY, FLOOR))}

HEADLINE = {
    SETTLED: "Settled",
    WATCH: "Moving",
    RISING_SOON: "About to get up",
    UNSTEADY: "Unsteady on their feet",
    FLOOR: "On the floor",
    UNUSABLE: "Cannot see the room",
}


@dataclass
class RiskState:
    """What the engine believes right now, and everything behind it."""

    time_s: float
    state: str
    headline: str
    reasons: list[str] = field(default_factory=list)
    hazard_reasons: list[str] = field(default_factory=list)
    posture: str = ""
    gait_score: float | None = None
    lead_context: str = ""
    certainty: str = "observed"

    @property
    def rank(self) -> int:
        return RISK_RANK.get(self.state, -1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": round(self.time_s, 3),
            "state": self.state,
            "headline": self.headline,
            "reasons": list(self.reasons),
            "hazard_reasons": list(self.hazard_reasons),
            "posture": self.posture,
            "gait_score": None if self.gait_score is None else round(self.gait_score, 3),
            "lead_context": self.lead_context,
            "certainty": self.certainty,
        }


def assess(
    time_s: float,
    view: ViewReport,
    exit_reading: ExitReading | None,
    gait: GaitReport | None,
    hazards: HazardReport | None,
    thresholds: Thresholds,
) -> RiskState:
    """Fuse the four channels into one state. Order of precedence is deliberate."""
    hazard_reasons = [h.description for h in (hazards.hazards if hazards else [])]

    if not view.usable:
        return RiskState(
            time_s=time_s,
            state=UNUSABLE,
            headline=HEADLINE[UNUSABLE],
            reasons=[view.detail or view.state, view.remedy],
            hazard_reasons=hazard_reasons,
            certainty="cannot assess",
        )

    if exit_reading is None:
        return RiskState(
            time_s=time_s,
            state=SETTLED,
            headline=HEADLINE[SETTLED],
            reasons=["nobody in view"],
            hazard_reasons=hazard_reasons,
            certainty="no person",
        )

    posture = exit_reading.state

    if posture == ON_FLOOR:
        return RiskState(
            time_s=time_s,
            state=FLOOR,
            headline=HEADLINE[FLOOR],
            reasons=exit_reading.reasons or ["the body maps onto the floor plane"],
            hazard_reasons=hazard_reasons,
            posture=posture,
            certainty="measured on the floor plane",
        )

    if posture in (STANDING, WALKING) and gait is not None and gait.scored:
        if gait.score >= thresholds.gait_unsteady_score:
            return RiskState(
                time_s=time_s,
                state=UNSTEADY,
                headline=HEADLINE[UNSTEADY],
                reasons=gait.reasons,
                hazard_reasons=hazard_reasons,
                posture=posture,
                gait_score=gait.score,
                certainty=f"scored over {thresholds.gait_window_s:.0f} s of walking",
            )

    if posture in (PREPARING, RISING):
        context = (
            "the weight is still on the bed"
            if posture == PREPARING
            else "the weight has already left the bed"
        )
        return RiskState(
            time_s=time_s,
            state=RISING_SOON,
            headline=HEADLINE[RISING_SOON],
            reasons=exit_reading.reasons,
            hazard_reasons=hazard_reasons,
            posture=posture,
            gait_score=gait.score if gait and gait.scored else None,
            lead_context=context,
            certainty="observed movement, not a prediction of a fall",
        )

    if posture in ("stirring", "sitting up", "on the edge", STANDING, WALKING):
        return RiskState(
            time_s=time_s,
            state=WATCH,
            headline=HEADLINE[WATCH],
            reasons=exit_reading.reasons or [f"{posture}"],
            hazard_reasons=hazard_reasons,
            posture=posture,
            gait_score=gait.score if gait and gait.scored else None,
            certainty="observed",
        )

    return RiskState(
        time_s=time_s,
        state=SETTLED,
        headline=HEADLINE[SETTLED],
        reasons=["no movement that precedes standing"],
        hazard_reasons=hazard_reasons,
        posture=posture,
        certainty="observed",
    )
