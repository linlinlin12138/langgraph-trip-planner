from __future__ import annotations

from datetime import date, time

from trip_agents.graph.nodes.budget import compute_budget
from trip_agents.schemas import DayPlan, Itinerary
from trip_agents.validation.rules import has_errors, validate_itinerary

from .conftest import act


def codes(issues):  # type: ignore[no-untyped-def]
    return sorted(i.code for i in issues if i.severity == "error")


def test_good_itinerary_passes(world, good_itinerary) -> None:  # type: ignore[no-untyped-def]
    req, rb = world
    br = compute_budget(req, rb, good_itinerary)
    issues = validate_itinerary(good_itinerary, req, rb, br)
    assert not has_errors(issues), issues


def test_missing_and_wrong_date(world, good_itinerary) -> None:  # type: ignore[no-untyped-def]
    req, rb = world
    it = good_itinerary.model_copy(deep=True)
    it.days[0].date = date(2026, 11, 25)
    it.days.pop(1)
    issues = validate_itinerary(it, req, rb)
    assert {"wrong_date", "missing_day"} <= set(codes(issues))
    assert {i.day_index for i in issues if i.code == "missing_day"} == {1}


def test_time_overlap_and_closed(world, good_itinerary) -> None:  # type: ignore[no-untyped-def]
    req, rb = world
    it = good_itinerary.model_copy(deep=True)
    it.days[0].activities[1] = act("11:00", "12:30", "meal", "r1", 160)  # overlaps 09:00-11:30
    it.days[1].activities[0] = act("17:00", "18:30", "attraction", "a1", 150)  # 灵隐寺 closes 17:30
    c = codes(validate_itinerary(it, req, rb))
    assert "time_overlap" in c and "closed" in c


def test_dietary_unknown_ref_and_must_visit(world) -> None:  # type: ignore[no-untyped-def]
    req, rb = world
    it = Itinerary(
        hotel_id="h1",
        days=[
            DayPlan(day_index=0, date=date(2026, 11, 20), activities=[act("12:00", "13:00", "meal", "r2", 120)]),
            DayPlan(day_index=1, date=date(2026, 11, 21), activities=[act("09:00", "10:00", "attraction", "zzz")]),
        ],
    )
    c = codes(validate_itinerary(it, req, rb))
    assert c.count("dietary") == 1 and "unknown_ref" in c and "must_visit" in c


def test_budget_rule_and_computation(world, good_itinerary) -> None:  # type: ignore[no-untyped-def]
    req, rb = world
    br = compute_budget(req, rb, good_itinerary)
    # hotel 500 × 1 night × 1 room; meals 160×3; tickets 150
    assert br.total == 500 + 480 + 150
    assert br.min_feasible_cost == 500 + 80 * 2 * 2 * 2 + 75 * 2
    req.hard.budget_total = 1000
    br = compute_budget(req, rb, good_itinerary)
    assert "over_budget" in codes(validate_itinerary(good_itinerary, req, rb, br))
    assert not br.feasible


def test_relaxed_pace_is_only_a_warning(world, good_itinerary) -> None:  # type: ignore[no-untyped-def]
    req, rb = world
    it = good_itinerary.model_copy(deep=True)
    it.days[1].activities = [
        act("09:00", "10:00", "attraction", "a2"),
        act("10:30", "11:00", "attraction", "a2"),
        act("11:30", "12:00", "attraction", "a2"),
    ]
    issues = validate_itinerary(it, req, rb)
    pace = [i for i in issues if i.code == "pace"]
    assert pace and pace[0].severity == "warning"
    assert any(i.code == "duplicate_attraction" for i in issues)


def test_activity_schema_rejects_reversed_time() -> None:
    import pytest
    from pydantic import ValidationError

    from trip_agents.schemas import Activity

    with pytest.raises(ValidationError):
        Activity(start_time=time(10), end_time=time(9), kind="rest", title="t", est_cost=0)
    with pytest.raises(ValidationError):
        Activity(start_time=time(9), end_time=time(10), kind="meal", title="t", est_cost=0)  # missing ref_id
