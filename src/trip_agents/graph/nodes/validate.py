"""Rule validation node + the conditional edge that decides whether to re-plan."""

from __future__ import annotations

from typing import Any

from trip_agents.graph.deps import GraphOptions
from trip_agents.graph.state import TripState
from trip_agents.validation.rules import has_errors, validate_itinerary


def validate(state: TripState) -> dict[str, Any]:
    issues = validate_itinerary(state["itinerary"], state["request"], state["research"], state.get("budget"))
    return {"issues": issues, "issue_history": [issues]}


def route_after_validate(options: GraphOptions):  # type: ignore[no-untyped-def]
    def route(state: TripState) -> str:
        issues = state.get("issues", [])
        if not has_errors(issues):
            return "finalize"
        budget = state.get("budget")
        only_budget = all(i.code == "over_budget" for i in issues if i.severity == "error")
        if budget is not None and not budget.feasible and only_budget:
            return "finalize"  # infeasible: re-planning cannot fix it, explain instead
        if options.enable_revision and state.get("revision", 0) < options.max_revisions:
            return "itinerary"
        return "finalize"

    return route
