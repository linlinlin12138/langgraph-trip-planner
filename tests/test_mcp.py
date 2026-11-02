"""The MCP path must be a drop-in for the in-process tools."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from langchain_core.messages import HumanMessage

from trip_agents.graph.builder import build_graph
from trip_agents.graph.deps import AgentDeps, GraphOptions, LocalToolExecutor
from trip_agents.llm.mock import MockChatModel
from trip_agents.tools.mcp_client import McpToolExecutor
from trip_agents.tools.mcp_server import build_server

from .conftest import REQUEST_TEXT, TODAY

CALLS = [
    ("get_weather", {"city": "杭州", "start_date": "2026-11-20", "end_date": "2026-11-22"}),
    ("search_hotels", {"city": "北京", "tier": "budget"}),
    ("search_restaurants", {"city": "成都", "dietary": ["halal"]}),
    ("search_attractions", {"city": "西安", "categories": ["history"], "limit": 3}),
    ("get_route", {"city": "西安", "origin": "xa-a01", "destination": "回民街"}),
]


@pytest.fixture(scope="module")
def mcp_tools() -> Iterator[McpToolExecutor]:
    ex = McpToolExecutor.in_process(build_server())
    yield ex
    ex.close()


def test_discovery_matches_local_specs(mcp_tools: McpToolExecutor) -> None:
    local = {s["function"]["name"]: s["function"]["parameters"] for s in LocalToolExecutor().tool_schemas()}
    remote = {s["function"]["name"]: s["function"]["parameters"] for s in mcp_tools.tool_schemas()}
    assert remote == local


@pytest.mark.parametrize("name, args", CALLS)
def test_results_identical(mcp_tools: McpToolExecutor, name: str, args: dict) -> None:  # type: ignore[type-arg]
    local, remote = LocalToolExecutor().call(name, args), mcp_tools.call(name, args)
    assert remote.ok and remote.data == local.data


def test_errors_come_back_as_observations(mcp_tools: McpToolExecutor) -> None:
    bad = mcp_tools.call("get_weather", {"city": "杭州", "start_date": "2026-11-20", "end_date": "下周"})
    assert not bad.ok and "end_date" in (bad.error or "")
    assert "不支持的城市" in (mcp_tools.call("search_hotels", {"city": "广州"}).error or "")
    assert "未知工具" in (mcp_tools.call("book_flight", {}).error or "")


def test_graph_over_mcp_equals_local(mcp_tools: McpToolExecutor) -> None:
    def run(tools):  # type: ignore[no-untyped-def]
        g = build_graph(AgentDeps(model=MockChatModel(), tools=tools, today=TODAY), GraphOptions(human_in_loop=False))
        return g.invoke({"messages": [HumanMessage(content=REQUEST_TEXT)]})

    a, b = run(LocalToolExecutor()), run(mcp_tools)
    assert a["final"].itinerary == b["final"].itinerary
    assert [t["name"] for t in a["tool_calls"]] == [t["name"] for t in b["tool_calls"]]


def test_stdio_subprocess_smoke() -> None:
    with McpToolExecutor.stdio() as ex:
        assert len(ex.tool_schemas()) == 5
        assert ex.call(*CALLS[2]).ok
