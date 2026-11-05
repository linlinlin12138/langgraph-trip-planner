"""End-to-end graph behaviour with the mock model: interrupts, persistence, re-planning."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from trip_agents.graph.deps import GraphOptions

from .conftest import REQUEST_TEXT


def cfg(tid: str = "t1") -> dict:  # type: ignore[type-arg]
    return {"configurable": {"thread_id": tid}}


def start(text: str = REQUEST_TEXT) -> dict:  # type: ignore[type-arg]
    return {"messages": [HumanMessage(content=text)], "user_id": "u1"}


def test_clarify_then_select_then_confirm(make_graph) -> None:  # type: ignore[no-untyped-def]
    g = make_graph()
    out = g.invoke(start("想去成都玩，喜欢美食"), cfg())
    intr = out["__interrupt__"][0].value
    assert intr["type"] == "clarify" and set(intr["missing"]) == {"start_date", "days", "budget_total"}

    out = g.invoke(Command(resume="12月1日出发，玩2天，预算3000"), cfg())
    intr = out["__interrupt__"][0].value
    assert intr["type"] == "select_candidates" and intr["hotels"] and intr["attractions"]

    out = g.invoke(Command(resume=intr["defaults"]), cfg())
    assert out["__interrupt__"][0].value["type"] == "confirm"
    out = g.invoke(Command(resume={"confirmed": True}), cfg())
    assert out["status"] == "confirmed" and out["final"].passed_validation


def test_resume_after_process_restart(make_graph) -> None:  # type: ignore[no-untyped-def]
    g1 = make_graph(db="shared.sqlite")
    intr = g1.invoke(start(), cfg("persist"))["__interrupt__"][0].value
    assert intr["type"] == "select_candidates"
    del g1

    g2 = make_graph(db="shared.sqlite")  # fresh graph + fresh connection, same file
    snap = g2.get_state(cfg("persist"))
    assert snap.next == ("human_select",)
    chosen = {**intr["defaults"], "attraction_ids": intr["defaults"]["attraction_ids"][:4]}
    out = g2.invoke(Command(resume=chosen), cfg("persist"))
    assert out["__interrupt__"][0].value["type"] == "confirm"
    used = {a.ref_id for d in out["itinerary"].days for a in d.activities if a.kind == "attraction"}
    assert used <= set(chosen["attraction_ids"])  # planner respected the user's choice


def test_user_selection_is_sanitised(make_graph) -> None:  # type: ignore[no-untyped-def]
    g = make_graph()
    intr = g.invoke(start(), cfg())["__interrupt__"][0].value
    out = g.invoke(Command(resume={"hotel_id": "evil", "attraction_ids": ["nope"], "restaurant_ids": []}), cfg())
    sel = out["selection"]
    assert sel.hotel_id == intr["defaults"]["hotel_id"] and sel.attraction_ids == intr["defaults"]["attraction_ids"]


def test_validation_failure_triggers_replan(make_graph) -> None:  # type: ignore[no-untyped-def]
    g = make_graph(faults={"itinerary_overlap_once"}, options=GraphOptions(human_in_loop=False))
    out = g.invoke(start(), cfg())
    assert out["revision"] == 1
    first, last = out["issue_history"][0], out["issue_history"][-1]
    assert any(i.code == "time_overlap" and i.day_index == 0 for i in first)
    assert not any(i.severity == "error" for i in last)
    assert out["final"].passed_validation


def test_replan_is_capped(make_graph) -> None:  # type: ignore[no-untyped-def]
    g = make_graph(faults={"itinerary_overlap_always"}, options=GraphOptions(human_in_loop=False, max_revisions=2))
    out = g.invoke(start(), cfg())
    assert out["revision"] == 2 and len(out["issue_history"]) == 3
    assert not out["final"].passed_validation and out["final"].remaining_issues


def test_no_revision_ablation(make_graph) -> None:  # type: ignore[no-untyped-def]
    g = make_graph(faults={"itinerary_overlap_once"}, options=GraphOptions(human_in_loop=False, enable_revision=False))
    out = g.invoke(start(), cfg())
    assert out["revision"] == 0 and not out["final"].passed_validation


def test_infeasible_budget_skips_replanning(make_graph) -> None:  # type: ignore[no-untyped-def]
    g = make_graph(options=GraphOptions(human_in_loop=False))
    out = g.invoke(start("11月20日去上海玩4天，4个人，预算800"), cfg())
    final = out["final"]
    assert not final.feasible and out["revision"] == 0
    assert final.budget.suggestions and "预算不足" in final.message


def test_bad_tool_args_are_recorded_and_recovered(make_graph) -> None:  # type: ignore[no-untyped-def]
    g = make_graph(faults={"bad_tool_args_once"}, options=GraphOptions(human_in_loop=False))
    out = g.invoke(start(), cfg())
    weather = [t for t in out["tool_calls"] if t["name"] == "get_weather"]
    assert [t["ok"] for t in weather] == [False, True]
    assert out["research"].weather


def test_budget_beats_hotel_tier_preference(make_graph) -> None:  # type: ignore[no-untyped-def]
    """Regression found by the eval: luxury preference + modest budget used to end over budget."""
    g = make_graph(options=GraphOptions(human_in_loop=False))
    out = g.invoke(start("我一个人2027-03-21去西安四日游，总预算4800元，想住高档一点的五星酒店，喜欢文化艺术，别太累"), cfg())
    hotel_searches = [t for t in out["tool_calls"] if t["name"] == "search_hotels"]
    assert any("tier" not in t["args"] for t in hotel_searches)  # fell back to a price-capped search
    assert out["final"].passed_validation and out["budget"].total <= 4800


def test_research_top_up_fills_sparse_categories() -> None:
    """s020-like: 3 packed days, only 'shopping' searched -> 1 attraction; top-up must fill the plan."""
    from datetime import date

    from trip_agents.graph.deps import LocalToolExecutor
    from trip_agents.graph.nodes.research import assemble_bundle, top_up
    from trip_agents.schemas import TripRequest

    tools = LocalToolExecutor()
    req = TripRequest.model_validate(
        {"hard": {"destination": "杭州", "start_date": date(2027, 1, 14), "days": 3, "budget_total": 6000, "travelers": 2,
                  "dietary_restrictions": ["halal"]},
         "soft": {"pace": "packed", "interests": ["shopping"]}}
    )  # fmt: skip
    first = tools.call("search_attractions", {"city": "杭州", "categories": ["shopping"]})
    rest = tools.call("search_restaurants", {"city": "杭州", "dietary": ["halal"], "cuisine": "西湖菜"})
    results = [{"name": "search_attractions", "data": first.data}, {"name": "search_restaurants", "data": rest.data}]
    assert len(assemble_bundle(results).attractions) <= 2
    extra, notes = top_up(results, req, tools)
    b = assemble_bundle(results + extra)
    assert len(b.attractions) == 15 and len(notes) == 2
    assert b.restaurants and all("halal" in r.dietary_tags for r in b.restaurants)  # hard constraint kept
    again, _ = top_up(results + extra, req, tools)
    assert all(x["name"] != "search_attractions" for x in again)  # 15 >= 3*4+2, no second top-up
