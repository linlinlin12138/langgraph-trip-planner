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
from trip_agents.schemas import DayPlan, Itinerary, ResearchBundle, Restaurant, Selection, SkippedItem, TripRequest, ValidationIssue

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


def plan_total(req: TripRequest, rb: ResearchBundle, it: Itinerary) -> float:
    """Same total as the Budget agent: lodging + every activity's est_cost."""
    return hotel_cost(req, rb, it.hotel_id) + sum(a.est_cost for d in it.days for a in d.activities)


def repair_budget(req: TripRequest, rb: ResearchBundle, candidates: list[Restaurant], it: Itinerary) -> tuple[Itinerary, list[str]] | None:
    """Fix an over-budget plan without the LLM: swap the priciest meals for cheaper compliant restaurants.

    Re-planning every day through the model costs N calls per revision (a 5-day trip with two
    revisions took ~20 LLM calls and $0.4 on Gemini in the dev trial), while "too expensive" is
    usually just a couple of meals or the hotel. Returns None when swaps can't reach the budget;
    the caller then falls back to re-planning.
    """
    budget, n = req.hard.budget_total, req.hard.travelers
    diet = set(req.hard.dietary_restrictions)
    pool = sorted((r for r in candidates if diet <= set(r.dietary_tags)), key=lambda r: r.avg_price_per_person)
    plan = it.model_copy(deep=True)
    notes: list[str] = []
    meals = sorted((a for d in plan.days for a in d.activities if a.kind == "meal"), key=lambda a: -a.est_cost)
    for a in meals:
        if plan_total(req, rb, plan) <= budget:
            break
        used_today = {x.ref_id for d in plan.days if a in d.activities for x in d.activities if x.kind == "meal"}
        options = [r for r in pool if r.avg_price_per_person * n < a.est_cost]
        if not options:
            continue
        r = next((r for r in options if r.id not in used_today), options[0])
        old = rb.restaurant(a.ref_id or "")
        old_name = old.name if old else a.title
        a.title = a.title.replace(old_name, r.name) if old_name in a.title else f"用餐：{r.name}"
        a.ref_id, a.est_cost = r.id, round(r.avg_price_per_person * n, 2)
        a.notes = "为控制预算更换餐厅"
        notes.append(f"为满足预算，{old_name} 换成了 {r.name}")
    return (plan, notes) if plan_total(req, rb, plan) <= budget else None


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
        errors = {i.code for i in issues if i.severity == "error"}
        if prev is not None and errors == {"over_budget"}:  # only the price is wrong: try a rule-based fix first
            fixed = repair_budget(req, rb, restaurants or rb.restaurants, prev.model_copy(update={"hotel_id": hotel_id}))
            if fixed is not None:
                plan, notes = fixed
                plan.summary = " / ".join(f"D{d.day_index + 1} {d.theme}" for d in plan.days)
                return {"itinerary": plan, "revision": revision, "skipped_items": [], "adjustments": adjustments + notes}
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
