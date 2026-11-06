"""Itinerary agent: plans day by day with structured output.

Why per day: (1) a day is the natural unit for "skip and mark" -- if one day
fails 3 times we still return the rest; (2) re-planning after validation can
regenerate only the days that have issues; (3) shorter outputs fail less.
Trade-off: N sequential calls instead of 1 (more latency); see docs/DESIGN.md.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from trip_agents.graph.costs import activity_cost, apply_catalog_costs, hotel_cost, plan_total
from trip_agents.graph.deps import AgentDeps
from trip_agents.graph.prompts import ITINERARY_SYSTEM
from trip_agents.graph.state import TripState
from trip_agents.llm.context import render_context
from trip_agents.llm.structured import invoke_structured
from trip_agents.schemas import (
    Attraction,
    DayPlan,
    Itinerary,
    ResearchBundle,
    Restaurant,
    Selection,
    SkippedItem,
    TripRequest,
    ValidationIssue,
)

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


def repair_budget(
    req: TripRequest,
    rb: ResearchBundle,
    candidates: list[Restaurant],
    it: Itinerary,
    attraction_candidates: list[Attraction] | None = None,
    preferred_attractions: set[str] | None = None,
    preferred_restaurants: set[str] | None = None,
) -> tuple[Itinerary, list[str], list[tuple[int, int]]] | None:
    """Fix an over-budget plan without the LLM.

    1. (caller) a cheaper hotel; 2. the priciest meals -> cheaper diet-compliant restaurants;
    3. the priciest paid attractions that are not must-visit -> free / cheaper ones open in the same
       time slot, not yet in the plan, preferring the same category and the user's selection.
    In the Gemini dev run the remaining over-budget failures were mostly tickets: hotel + tickets
    alone left too little for meals, so meal swaps could never reach the budget.

    Re-planning every day through the model costs N calls per revision (a 5-day trip with two
    revisions took ~20 LLM calls and $0.4 on Gemini in the dev trial), while "too expensive" is
    usually just a couple of meals or the hotel. Costs come from the catalog (graph/costs.py), the
    same total the Budget agent uses. Returns (plan, notes, changed (day_index, position) slots), or
    None when swaps can't reach the budget; the caller then falls back to re-planning.
    """
    budget, n = req.hard.budget_total, req.hard.travelers
    diet = set(req.hard.dietary_restrictions)
    pool = sorted((r for r in candidates if diet <= set(r.dietary_tags)), key=lambda r: r.avg_price_per_person)
    plan = apply_catalog_costs(it, rb, n)
    notes: list[str] = []
    changed: list[tuple[int, int]] = []
    slots = [(d, k, a) for d in plan.days for k, a in enumerate(d.activities) if a.kind == "meal"]
    for d, k, a in sorted(slots, key=lambda x: -activity_cost(x[2], rb, n)):
        if plan_total(req, rb, plan) <= budget:
            break
        used_today = {x.ref_id for x in d.activities if x.kind == "meal"}
        options = [r for r in pool if r.avg_price_per_person * n < activity_cost(a, rb, n)]
        if not options:
            continue
        uses = Counter(x.ref_id for d in plan.days for x in d.activities if x.kind == "meal")
        # spread the swaps: not twice the same day, then the least-used restaurant, the user's picks, the cheapest
        liked = preferred_restaurants or set()
        r = min(options, key=lambda r: (r.id in used_today, uses[r.id], r.id not in liked, r.avg_price_per_person))
        old = rb.restaurant(a.ref_id or "")
        old_name = old.name if old else a.title
        a.title = a.title.replace(old_name, r.name) if old_name in a.title else f"用餐：{r.name}"
        a.ref_id, a.est_cost = r.id, round(r.avg_price_per_person * n, 2)
        a.notes = "为控制预算更换餐厅"
        notes.append(f"为满足预算，{old_name} 换成了 {r.name}")
        changed.append((d.day_index, k))

    must = set(req.hard.must_visit)
    attr_pool = attraction_candidates if attraction_candidates is not None else rb.attractions
    preferred = preferred_attractions or set()
    paid = [(d, k, a) for d in plan.days for k, a in enumerate(d.activities) if a.kind == "attraction"]
    for d, k, a in sorted(paid, key=lambda x: -activity_cost(x[2], rb, n)):
        if plan_total(req, rb, plan) <= budget:
            break
        old_attr = rb.attraction(a.ref_id or "")
        if old_attr is None or old_attr.name in must:
            continue
        used = {y.ref_id for dd in plan.days for y in dd.activities if y.kind == "attraction"}
        cost_now = activity_cost(a, rb, n)
        attr_options = [
            alt for alt in attr_pool
            if alt.id not in used and alt.ticket_price * n < cost_now and alt.open_time <= a.start_time and alt.close_time >= a.end_time
        ]  # fmt: skip
        if not attr_options:
            continue
        new = min(attr_options, key=lambda o: (o.category != old_attr.category, o.id not in preferred, o.ticket_price, -o.rating))
        a.title = a.title.replace(old_attr.name, new.name) if old_attr.name in a.title else new.name
        a.ref_id, a.est_cost = new.id, round(new.ticket_price * n, 2)
        a.notes = "为控制预算更换景点"
        notes.append(f"为满足预算，{old_attr.name} 换成了 {new.name}")
        changed.append((d.day_index, k))
    return (plan, notes, changed) if plan_total(req, rb, plan) <= budget else None


def make_itinerary_node(deps: AgentDeps):  # type: ignore[no-untyped-def]
    def itinerary(state: TripState, config: RunnableConfig) -> dict[str, Any]:
        req, rb = state["request"], state["research"]
        sel: Selection = state["selection"]
        prev = state.get("itinerary")
        revision = state.get("revision", -1) + 1
        n = req.hard.travelers
        model_costs: dict[str, list[float]] = dict(state.get("model_costs") or {})
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
            fixed = repair_budget(
                # every researched candidate, not just the selected ones: the selection is the best-rated, i.e. pricey
                req, rb, rb.restaurants, prev.model_copy(update={"hotel_id": hotel_id}),
                attraction_candidates=rb.attractions, preferred_attractions=set(sel.attraction_ids),
                preferred_restaurants=set(sel.restaurant_ids),
            )  # fmt: skip
            if fixed is not None:
                plan, notes, changed = fixed
                plan.summary = " / ".join(f"D{d.day_index + 1} {d.theme}" for d in plan.days)
                costs = {k: list(v) for k, v in model_costs.items()}
                by_day = {d.day_index: d for d in plan.days}
                for day_i, pos in changed:  # swapped in by code: the "model estimate" there is the catalog price
                    if str(day_i) in costs and pos < len(costs[str(day_i)]):
                        costs[str(day_i)][pos] = by_day[day_i].activities[pos].est_cost
                return {"itinerary": plan, "revision": revision, "skipped_items": [], "adjustments": adjustments + notes,
                        "model_costs": costs}  # fmt: skip
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
                model_costs[str(i)] = [a.est_cost for a in res.value.activities]  # the model's numbers, for auditing only
                kept[i] = apply_catalog_costs(Itinerary(hotel_id=None, days=[res.value]), rb, n).days[0]
            else:
                skipped.append(SkippedItem(stage="itinerary", item=f"day{i}", errors=res.errors))

        plan = Itinerary(hotel_id=hotel_id, days=[kept[k] for k in sorted(kept)])
        plan.summary = " / ".join(f"D{d.day_index + 1} {d.theme}" for d in plan.days)
        model_costs = {k: v for k, v in model_costs.items() if int(k) in kept}
        return {"itinerary": plan, "revision": revision, "skipped_items": skipped, "adjustments": adjustments, "model_costs": model_costs}

    return itinerary
