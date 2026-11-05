"""The mock's rule-based extractor drives the offline demo and eval, so it is tested too."""

from __future__ import annotations

from datetime import date

import pytest

from trip_agents.graph.nodes.human_select import default_selection, sanitize_selection
from trip_agents.llm.mock_responders import cn_to_int, extract_request
from trip_agents.schemas import RequestExtraction, Selection

TODAY = date(2026, 11, 2)


@pytest.mark.parametrize("s, n", [("3", 3), ("三", 3), ("两", 2), ("十", 10), ("十二", 12), ("二十", 20)])
def test_cn_to_int(s: str, n: int) -> None:
    assert cn_to_int(s) == n


def test_extract_full_request() -> None:
    r = extract_request(["一家三口想2027-01-05去北京玩五天，人均预算3000，不吃海鲜，一定要去故宫博物院，别太累"], TODAY, None)
    assert r["destination"] == "北京" and r["start_date"] == "2027-01-05" and r["days"] == 5
    assert r["travelers"] == 3 and r["budget_total"] == 9000
    assert r["dietary_restrictions"] == ["no_seafood"] and r["must_visit"] == ["故宫博物院"] and r["pace"] == "relaxed"


def test_later_turns_override_and_alias() -> None:
    r = extract_request(["去西安玩3天，预算5000", "预算改成8000，想去兵马俑"], TODAY, None)
    assert r["budget_total"] == 8000 and r["must_visit"] == ["秦始皇帝陵博物院(兵马俑)"]


def test_month_day_rolls_to_next_year() -> None:
    r = extract_request(["3月1日去杭州"], TODAY, None)
    assert r["start_date"] == "2027-03-01"


def test_missing_required() -> None:
    ex = RequestExtraction(destination="杭州", days=2)
    assert ex.missing_required() == ["start_date", "budget_total"]


def test_default_selection_prefers_constraints(world) -> None:  # type: ignore[no-untyped-def]
    req, rb = world
    sel = default_selection(req, rb)
    assert sel.restaurant_ids[0] == "r1"  # diet-compliant first
    assert sel.attraction_ids[0] == "a1"  # must-visit first
    assert sanitize_selection({"hotel_id": "bad"}, rb, sel) == sel
    assert sanitize_selection(Selection(attraction_ids=["a2"]), rb, sel).attraction_ids == ["a2"]
