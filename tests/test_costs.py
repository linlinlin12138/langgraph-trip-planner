"""Plan costs come from the catalog, not from the model's est_cost (graph/costs.py)."""

from __future__ import annotations

from datetime import date, time

from trip_agents.evals.world import evaluate_plan, world_bundle
from trip_agents.graph.costs import plan_total
from trip_agents.graph.nodes.budget import compute_budget
from trip_agents.graph.nodes.itinerary import repair_budget
from trip_agents.schemas import Activity, DayPlan, Itinerary, TripRequest
from trip_agents.validation.rules import validate_itinerary

RB = world_bundle("杭州")
PRICEY = max(RB.restaurants, key=lambda r: r.avg_price_per_person)
PAID = max(RB.attractions, key=lambda a: a.ticket_price)


def request(budget: float, travelers: int = 2) -> TripRequest:
    return TripRequest.model_validate(
        {"hard": {"destination": "杭州", "start_date": date(2027, 1, 14), "days": 1, "budget_total": budget, "travelers": travelers}}
    )


def plan(meal_ref: str, meal_cost: float, attr_ref: str | None = None, attr_cost: float = 0.0) -> Itinerary:
    acts = [Activity(start_time=time(12), end_time=time(13), kind="meal", title="午餐", ref_id=meal_ref, est_cost=meal_cost)]
    if attr_ref:
        acts.insert(
            0, Activity(start_time=time(9), end_time=time(11), kind="attraction", title="景点", ref_id=attr_ref, est_cost=attr_cost)
        )
    return Itinerary(hotel_id=None, days=[DayPlan(day_index=0, date=date(2027, 1, 14), activities=acts)])


def test_underquoted_meal_still_counts_at_catalog_price_and_fails_budget() -> None:
    real = PRICEY.avg_price_per_person * 2
    req = request(budget=real * 0.5)
    it = plan(PRICEY.id, meal_cost=10)  # the model claims 10 yuan for two people
    br = compute_budget(req, RB, it)
    assert br.total == real
    assert br.model_refs_total == 10 and br.catalog_refs_total == real and br.model_plan_total == 10
    assert "over_budget" in [i.code for i in validate_itinerary(it, req, RB, br)]


def test_name_references_are_priced_from_the_catalog_in_evaluation() -> None:
    gold = request(budget=100_000)
    it = plan(PRICEY.name, meal_cost=1, attr_ref=PAID.name, attr_cost=0)  # baseline style: names, made-up prices
    ev = evaluate_plan(it, gold)
    assert ev.budget is not None
    assert ev.budget.total == (PRICEY.avg_price_per_person + PAID.ticket_price) * 2
    assert ev.budget.model_refs_total == 1 and ev.budget.catalog_refs_total == ev.budget.total


def test_repair_uses_the_same_total_as_the_budget_agent() -> None:
    req = request(budget=1)
    it = plan(PRICEY.id, meal_cost=5, attr_ref=PAID.id, attr_cost=0)
    assert plan_total(req, RB, it) == compute_budget(req, RB, it).total
    req.hard.budget_total = compute_budget(req, RB, it).total - 1
    fixed = repair_budget(req, RB, RB.restaurants, it)
    assert fixed is not None
    assert plan_total(req, RB, fixed[0]) == compute_budget(req, RB, fixed[0]).total <= req.hard.budget_total
    # the under-quoted est_cost can't make an over-budget plan look fixed
    assert repair_budget(request(budget=1), RB, [PRICEY], plan(PRICEY.id, meal_cost=0)) is None


def test_graph_rewrites_costs_and_run_record_carries_the_audit() -> None:
    from trip_agents.evals.dataset import build_dataset
    from trip_agents.evals.systems import SYSTEMS, run_one
    from trip_agents.graph.costs import activity_cost
    from trip_agents.llm.mock import MockChatModel
    from trip_agents.schemas import Itinerary as It

    sample = next(s for s in build_dataset(10, seed=1) if s.gold.hard.days > 1)
    for name in ("full", "baseline"):
        r = run_one(sample, SYSTEMS[name], lambda: MockChatModel())
        assert r.cost_model_refs is not None and r.cost_catalog_refs is not None and r.plan_total_catalog is not None
        assert r.budget_flip is not None
    r = run_one(sample, SYSTEMS["full"], lambda: MockChatModel())
    it = It.model_validate(r.itinerary)
    world = world_bundle(sample.gold.hard.destination)
    n = sample.gold.hard.travelers
    assert all(a.est_cost == activity_cost(a, world, n) for d in it.days for a in d.activities if a.kind != "transit")
