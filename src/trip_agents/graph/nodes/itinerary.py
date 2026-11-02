"""Itinerary agent: plans day by day with structured output.

Why per day: (1) a day is the natural unit for "skip and mark" -- if one day
fails 3 times we still return the rest; (2) re-planning after validation can
regenerate only the days that have issues; (3) shorter outputs fail less.
Trade-off: N sequential calls instead of 1 (more latency); see docs/DESIGN.md.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from trip_agents.graph.deps import AgentDeps
from trip_agents.graph.prompts import ITINERARY_SYSTEM
from trip_agents.graph.state import TripState
from trip_agents.llm.context import render_context
from trip_agents.llm.structured import invoke_structured
from trip_agents.schemas import DayPlan, Itinerary, ResearchBundle, Selection, SkippedItem, TripRequest, ValidationIssue

# Issues without a day_index that can only be fixed by re-planning every day.
GLOBAL_CODES = {"over_budget", "must_visit", "missing_hotel", "unknown_hotel"}


def days_to_replan(req: TripRequest, prev: Itinerary | None, issues: list[ValidationIssue]) -> list[int]:
    all_days = list(range(req.hard.days))
    if prev is None:
        return all_days
    errors = [i for i in issues if i.severity == "error"]
    if any(i.code in GLOBAL_CODES or i.day_index is None for i in errors):
        return all_days
    have = {d.day_index for d in prev.days}
    bad = {i.day_index for i in errors if i.day_index is not None}
    return sorted(bad | (set(all_days) - have))


def make_day_checker(day_index: int, req: TripRequest, rb: ResearchBundle):  # type: ignore[no-untyped-def]
    """Cheap, local checks done inside the structured-output retry loop."""
    expected_date = req.hard.dates[day_index]

    def check(d: DayPlan) -> list[str]:
        problems = []
        if d.day_index != day_index:
            problems.append(f"day_index 应为 {day_index}，实际为 {d.day_index}")
        if d.date != expected_date:
            problems.append(f"date 应为 {expected_date}，实际为 {d.date}")
        if not rb.attractions and not rb.restaurants:
            return problems  # no-research mode: nothing to check references against
        for a in d.activities:
            if a.kind == "attraction" and rb.attraction(a.ref_id or "") is None:
                problems.append(f"景点 ref_id '{a.ref_id}' 不在候选列表中")
            if a.kind == "meal" and rb.restaurant(a.ref_id or "") is None:
                problems.append(f"餐厅 ref_id '{a.ref_id}' 不在候选列表中")
        return problems

    return check


def hotel_cost(req: TripRequest, rb: ResearchBundle, hotel_id: str | None) -> float:
    h = rb.hotel(hotel_id) if hotel_id else None
    if h is None or req.hard.days <= 1:
        return 0.0
    return h.price_per_night * (req.hard.days - 1) * ((req.hard.travelers + 1) // 2)


def cheaper_hotel(req: TripRequest, rb: ResearchBundle, current: str | None, other_costs: float) -> str | None:
    """Hard budget beats soft hotel preference.

    * some hotel alone closes the gap -> best-rated such hotel (preferred tier first)
    * none does, but lodging is the dominant cost -> cheapest hotel
    * otherwise keep the hotel and let the day plans get cheaper
    """
    if req.hard.days <= 1 or not rb.hotels:
        return current
    fits = [h for h in rb.hotels if hotel_cost(req, rb, h.id) + other_costs <= req.hard.budget_total]
    if fits:
        return min(fits, key=lambda h: (req.soft.hotel_tier is not None and h.tier != req.soft.hotel_tier, -h.rating)).id
    if hotel_cost(req, rb, current) > other_costs:
        return min(rb.hotels, key=lambda h: h.price_per_night).id
    return current


def make_itinerary_node(deps: AgentDeps):  # type: ignore[no-untyped-def]
    def itinerary(state: TripState, config: RunnableConfig) -> dict[str, Any]:
        req, rb = state["request"], state["research"]
        sel: Selection = state["selection"]
        prev = state.get("itinerary")
        revision = state.get("revision", -1) + 1
        issues = state.get("issues", []) if prev is not None else []
        targets = days_to_replan(req, prev, issues)

        kept: dict[int, DayPlan] = {d.day_index: d for d in (prev.days if prev else []) if d.day_index not in targets}
        attractions = [a for a in rb.attractions if a.id in sel.attraction_ids]
        restaurants = [r for r in rb.restaurants if r.id in sel.restaurant_ids]
        weather = {w.date: w for w in rb.weather}
        hotel_id = prev.hotel_id if prev else sel.hotel_id
        adjustments: list[str] = []
        budget = state.get("budget")
        if prev is not None and budget is not None and any(i.code == "over_budget" for i in issues):
            new_hotel = cheaper_hotel(req, rb, hotel_id, budget.total - hotel_cost(req, rb, hotel_id))
            if new_hotel and new_hotel != hotel_id:
                old, new = rb.hotel(hotel_id or ""), rb.hotel(new_hotel)
                adjustments.append(f"为满足预算，酒店由「{old.name if old else hotel_id}」调整为「{new.name if new else new_hotel}」")
                hotel_id = new_hotel
        per_day_budget = (req.hard.budget_total - hotel_cost(req, rb, hotel_id)) / req.hard.days
        skipped: list[SkippedItem] = []

        for i in targets:
            used_attr = {a.ref_id for d in kept.values() for a in d.activities if a.kind == "attraction"}
            used_rest = sorted({a.ref_id for d in kept.values() for a in d.activities if a.kind == "meal" and a.ref_id})
            visited_names = {x.name for x in attractions if x.id in used_attr}
            feedback = [iss.message for iss in issues if iss.severity == "error" and iss.day_index in (i, None)]
            ctx = {
                "task": "plan_day",
                "request": req.model_dump(mode="json"),
                "day_index": i,
                "date": req.hard.dates[i].isoformat(),
                "days_remaining": sum(1 for t in targets if t >= i),
                "weather": weather.get(req.hard.dates[i]),
                "hotel": rb.hotel(sel.hotel_id) if sel.hotel_id else None,
                "attractions": [a for a in attractions if a.id not in used_attr],
                "restaurants": restaurants,
                "used_restaurant_ids": used_rest,
                "must_visit_remaining": [m for m in req.hard.must_visit if m not in visited_names],
                "budget_per_day": round(per_day_budget, 1),
                "feedback": feedback,
                "no_candidates": not attractions,
                "used_names": sorted({a.title for d in kept.values() for a in d.activities if a.kind == "attraction"}),
            }
            res = invoke_structured(
                deps.model,
                DayPlan,
                [SystemMessage(content=ITINERARY_SYSTEM), HumanMessage(content=render_context(ctx))],
                max_retries=deps.structured_max_retries,
                extra_check=make_day_checker(i, req, rb),
                stage="itinerary",
                item=f"day{i}",
                config=config,
            )
            if res.value is not None:
                kept[i] = res.value
            else:
                skipped.append(SkippedItem(stage="itinerary", item=f"day{i}", errors=res.errors))

        plan = Itinerary(hotel_id=hotel_id, days=[kept[k] for k in sorted(kept)])
        plan.summary = " / ".join(f"D{d.day_index + 1} {d.theme}" for d in plan.days)
        return {"itinerary": plan, "revision": revision, "skipped_items": skipped, "adjustments": adjustments}

    return itinerary
