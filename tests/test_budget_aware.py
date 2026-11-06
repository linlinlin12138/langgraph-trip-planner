"""Tight budgets: cheap candidates are fetched and ranked first; must-visit names match leniently."""

from __future__ import annotations

from datetime import date, time

from trip_agents.evals.world import world_bundle
from trip_agents.graph.deps import LocalToolExecutor
from trip_agents.graph.nodes.human_select import budget_caps, default_selection
from trip_agents.graph.nodes.research import assemble_bundle, top_up
from trip_agents.schemas import Activity, DayPlan, Itinerary, TripRequest
from trip_agents.validation.names import same_place
from trip_agents.validation.rules import check_must_visit


def req(budget: float, days: int = 4, travelers: int = 3, **hard: object) -> TripRequest:
    return TripRequest.model_validate(
        {"hard": {"destination": "成都", "start_date": date(2027, 2, 2), "days": days, "budget_total": budget,
                  "travelers": travelers, **hard}, "soft": {"pace": "moderate"}}
    )  # fmt: skip


def test_same_place_is_lenient_but_not_sloppy() -> None:
    assert same_place("成都大熊猫繁育研究基地", "大熊猫繁育研究基地")
    assert same_place("秦始皇帝陵博物院", "秦始皇帝陵博物院(兵马俑)")
    assert not same_place("武侯祠", "杜甫草堂")
    assert not same_place("园", "公园")  # one character is too short to match by containment


def test_must_visit_check_accepts_a_fuller_name() -> None:
    rb = world_bundle("成都")
    panda = next(a for a in rb.attractions if a.name == "大熊猫繁育研究基地")
    act = Activity(start_time=time(9), end_time=time(11), kind="attraction", title=panda.name, ref_id=panda.id, est_cost=0)
    it = Itinerary(hotel_id=None, days=[DayPlan(day_index=0, date=date(2027, 2, 2), activities=[act])])
    r = req(10_000, days=1, travelers=1, must_visit=["成都大熊猫繁育研究基地"])
    assert check_must_visit(it, r, rb, None) == []
    r.hard.must_visit = ["武侯祠"]
    assert [i.code for i in check_must_visit(it, r, rb, None)] == ["must_visit"]


def test_budget_caps_only_when_tight() -> None:
    assert budget_caps(req(20_000)) is None
    caps = budget_caps(req(2100))  # s166-like: 4 days, 3 people
    assert caps is not None and caps.per_person_day < 250 and 0 < caps.ticket < caps.meal


def test_tight_budget_fetches_and_ranks_affordable_options() -> None:
    tools = LocalToolExecutor()
    r = req(2100, dietary_restrictions=["no_spicy"])
    caps = budget_caps(r)
    assert caps is not None
    # the agent's own search: top-rated only (what Gemini did)
    first = tools.call("search_restaurants", {"city": "成都", "dietary": ["no_spicy"], "limit": 8})
    attrs = tools.call("search_attractions", {"city": "成都", "limit": 10})
    results = [{"name": "search_restaurants", "data": first.data}, {"name": "search_attractions", "data": attrs.data}]
    before = assemble_bundle(results)
    extra, notes = top_up(results, r, tools)
    bundle = assemble_bundle(results + extra)
    cheap = [x for x in bundle.restaurants if x.avg_price_per_person <= caps.meal and "no_spicy" in x.dietary_tags]
    assert len(cheap) > sum(x.avg_price_per_person <= caps.meal for x in before.restaurants)
    assert any("预算紧" in n for n in notes)
    sel = default_selection(r, bundle)
    picked = [bundle.restaurant(i) for i in sel.restaurant_ids]
    assert all(p is not None and p.avg_price_per_person <= caps.meal for p in picked[: len(cheap)])  # affordable first
    # loose budget: ranking by rating as before
    loose = req(50_000, dietary_restrictions=["no_spicy"])
    sel2 = default_selection(loose, bundle)
    ratings = [bundle.restaurant(i).rating for i in sel2.restaurant_ids]  # type: ignore[union-attr]
    assert ratings == sorted(ratings, reverse=True)
