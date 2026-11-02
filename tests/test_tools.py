from __future__ import annotations

from langchain_core.messages import AIMessage

from trip_agents.graph.deps import AgentDeps, LocalToolExecutor
from trip_agents.graph.nodes.research import assemble_bundle, build_research_subgraph
from trip_agents.tools.executor import execute_tool

from .conftest import TODAY, ScriptedChatModel


def test_valid_call_and_filters() -> None:
    res = execute_tool("search_restaurants", {"city": "成都", "dietary": ["halal"]})
    assert res.ok and res.data and all("halal" in r["dietary_tags"] for r in res.data)
    res = execute_tool("search_hotels", {"city": "北京", "tier": "luxury", "limit": 2})
    assert res.ok and len(res.data) <= 2 and all(h["tier"] == "luxury" for h in res.data)


def test_errors_are_observations_not_exceptions() -> None:
    bad = execute_tool("get_weather", {"city": "杭州", "start_date": "2026-11-20", "end_date": "下周五"})
    assert not bad.ok and "end_date" in (bad.error or "")
    assert bad.to_observation().startswith("ERROR")
    assert "不支持的城市" in (execute_tool("search_hotels", {"city": "广州"}).error or "")
    assert "未知工具" in (execute_tool("book_flight", {}).error or "")
    reversed_range = execute_tool("get_weather", {"city": "杭州", "start_date": "2026-11-22", "end_date": "2026-11-20"})
    assert "end_date" in (reversed_range.error or "")


def test_route_accepts_ids_and_names() -> None:
    r = execute_tool("get_route", {"city": "西安", "origin": "xa-a01", "destination": "回民街"})
    assert r.ok and r.data["mode"] == "walk"
    assert not execute_tool("get_route", {"city": "西安", "origin": "不存在", "destination": "回民街"}).ok


def _call(name: str, args: dict, i: int) -> dict:  # type: ignore[type-arg]
    return {"name": name, "args": args, "id": f"c{i}", "type": "tool_call"}


def test_research_loop_dedup_and_round_limit() -> None:
    same = {"city": "杭州"}
    script = [AIMessage(content="", tool_calls=[_call("search_hotels", {"city": "杭州", "limit": k}, k)]) for k in (1, 2, 3, 4)]
    script.insert(1, AIMessage(content="", tool_calls=[_call("search_hotels", {"city": "杭州", "limit": 1}, 9)]))  # duplicate
    script.append(AIMessage(content="", tool_calls=[_call("search_attractions", same, 10)]))
    script.append(AIMessage(content="done"))
    deps = AgentDeps(
        model=ScriptedChatModel(script=script),
        tools=LocalToolExecutor(),
        today=TODAY,
        research_max_steps=10,
        research_max_rounds_per_tool=3,
    )
    out = build_research_subgraph(deps).invoke({"messages": []})
    recs = out["records"]
    assert [r["duplicate"] for r in recs] == [False, True, False, False, False, False]
    assert recs[4]["error"] == "round_limit"  # 4th distinct hotel search refused
    assert recs[5]["name"] == "search_attractions" and recs[5]["ok"]
    bundle = assemble_bundle(out["results"])
    assert bundle.hotels and bundle.attractions


def test_research_step_limit() -> None:
    loop = AIMessage(content="", tool_calls=[_call("get_weather", {"city": "杭州", "start_date": "x", "end_date": "y"}, 0)])
    deps = AgentDeps(model=ScriptedChatModel(script=[loop]), tools=LocalToolExecutor(), today=TODAY, research_max_steps=4)
    out = build_research_subgraph(deps).invoke({"messages": []})
    assert out["step"] == 4
    assert "最大步数" in out["messages"][-1].content
