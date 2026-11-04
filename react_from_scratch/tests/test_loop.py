"""Acceptance tests for YOUR run_react / parse_tool_calls.

They are skipped while loop.py still raises NotImplementedError, so CI stays green.
Once you implement the loop they run automatically; to see what is pending:

    uv run pytest react_from_scratch -rs
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, Field

from react_from_scratch.client import ScriptedClient, TransientModelError
from react_from_scratch.loop import parse_tool_calls, run_react
from react_from_scratch.registry import ToolRegistry
from react_from_scratch.types import ModelTurn, RawToolCall, ReactConfig, Usage


def _implemented() -> bool:
    try:
        run_react(ScriptedClient([ModelTurn(text="ok")]), ToolRegistry(), "ping")
        parse_tool_calls(ModelTurn(text=None))
    except NotImplementedError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _implemented(), reason="react_from_scratch/loop.py not implemented yet (your exercise)")


# ----------------------------------------------------------------------------- fixtures
class AddIn(BaseModel):
    a: int
    b: int


class EchoIn(BaseModel):
    text: str = Field(min_length=1)


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    executed: list[tuple[str, Any]] = []

    def add(x: BaseModel) -> dict[str, int]:
        assert isinstance(x, AddIn)
        executed.append(("add", x))
        return {"sum": x.a + x.b}

    def echo(x: BaseModel) -> dict[str, str]:
        assert isinstance(x, EchoIn)
        executed.append(("echo", x))
        return {"text": x.text}

    reg.register("add", "两数相加", AddIn, add)
    reg.register("echo", "原样返回", EchoIn, echo)
    reg.executed = executed  # type: ignore[attr-defined]
    return reg


def call(i: int, name: str, args: dict[str, Any] | str) -> RawToolCall:
    return RawToolCall(id=f"c{i}", name=name, arguments=args if isinstance(args, str) else json.dumps(args))


def turn(*calls: RawToolCall, text: str | None = None, tokens: tuple[int, int] = (10, 2)) -> ModelTurn:
    return ModelTurn(text=text, tool_calls=list(calls), usage=Usage(*tokens))


def final(text: str = "完成") -> ModelTurn:
    return ModelTurn(text=text, usage=Usage(5, 1))


def tool_msgs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in messages if m["role"] == "tool"]


# ----------------------------------------------------------------------------- parsing
def test_parse_tool_calls() -> None:
    parsed = parse_tool_calls(turn(call(1, "add", {"a": 1, "b": 2}), call(2, "add", ""), call(3, "add", "{bad"), call(4, "add", "[1]")))
    assert [p.id for p in parsed] == ["c1", "c2", "c3", "c4"]
    assert parsed[0].args == {"a": 1, "b": 2} and parsed[0].parse_error is None
    assert parsed[1].args == {}
    assert parsed[2].args is None and parsed[2].parse_error
    assert parsed[3].args is None and parsed[3].parse_error  # valid JSON but not an object


# ----------------------------------------------------------------------------- loop
def test_immediate_final_answer(registry: ToolRegistry) -> None:
    client = ScriptedClient([final("你好")])
    r = run_react(client, registry, "打个招呼", system_prompt="SYS")
    assert r.stopped_reason == "final_answer" and r.final_text == "你好" and r.steps == 1
    assert [m["role"] for m in r.messages] == ["system", "user", "assistant"]
    assert r.messages[0]["content"] == "SYS" and r.messages[1]["content"] == "打个招呼"
    assert client.requests[0][-1]["content"] == "打个招呼"


def test_tool_call_then_answer(registry: ToolRegistry) -> None:
    client = ScriptedClient([turn(call(1, "add", {"a": 2, "b": 3})), final("5")])
    r = run_react(client, registry, "2+3")
    assert r.final_text == "5" and r.steps == 2
    assert [m["role"] for m in r.messages] == ["system", "user", "assistant", "tool", "assistant"]
    t = tool_msgs(r.messages)[0]
    assert t["tool_call_id"] == "c1" and json.loads(t["content"]) == {"sum": 5}
    assert client.requests[1][-1]["role"] == "tool"  # the observation was sent back to the model
    assert r.tool_log[0].name == "add" and r.tool_log[0].ok and r.tool_log[0].step == 1


def test_parallel_calls_each_get_one_tool_message(registry: ToolRegistry) -> None:
    r = run_react(ScriptedClient([turn(call(1, "add", {"a": 1, "b": 1}), call(2, "echo", {"text": "hi"})), final()]), registry, "x")
    assert [m["tool_call_id"] for m in tool_msgs(r.messages)] == ["c1", "c2"]
    assert [log.name for log in r.tool_log] == ["add", "echo"]


def test_invalid_args_become_error_observation_and_model_can_retry(registry: ToolRegistry) -> None:
    client = ScriptedClient([turn(call(1, "add", {"a": "x"})), turn(call(2, "add", {"a": 1, "b": 2})), final()])
    r = run_react(client, registry, "x")
    first, second = tool_msgs(r.messages)
    assert first["content"].startswith("ERROR") and "b" in first["content"]
    assert not r.tool_log[0].ok and r.tool_log[0].error
    assert json.loads(second["content"]) == {"sum": 3} and r.tool_log[1].ok
    assert len(registry.executed) == 1  # type: ignore[attr-defined]  # invalid call never reached the function


def test_malformed_json_and_unknown_tool(registry: ToolRegistry) -> None:
    r = run_react(ScriptedClient([turn(call(1, "add", "{oops"), call(2, "fly", {})), final()]), registry, "x")
    a, b = tool_msgs(r.messages)
    assert a["content"].startswith("ERROR") and b["content"].startswith("ERROR") and "fly" in b["content"]
    assert [log.ok for log in r.tool_log] == [False, False]
    assert r.tool_log[0].args is None


def test_duplicate_call_is_not_executed_again(registry: ToolRegistry) -> None:
    same = {"a": 1, "b": 2}
    r = run_react(ScriptedClient([turn(call(1, "add", same)), turn(call(2, "add", {"b": 2, "a": 1})), final()]), registry, "x")
    assert len(registry.executed) == 1  # type: ignore[attr-defined]
    assert [log.duplicate for log in r.tool_log] == [False, True]
    dup = tool_msgs(r.messages)[1]
    assert dup["tool_call_id"] == "c2" and '"sum": 3' in dup["content"]  # previous result returned


def test_failed_call_is_not_cached_as_duplicate(registry: ToolRegistry) -> None:
    bad = {"a": 1}
    run = run_react(ScriptedClient([turn(call(1, "add", bad)), turn(call(2, "add", bad)), final()]), registry, "x")
    assert [log.duplicate for log in run.tool_log] == [False, False]


def test_per_tool_round_limit(registry: ToolRegistry) -> None:
    script = [turn(call(i, "add", {"a": i, "b": 0})) for i in range(4)] + [turn(call(9, "echo", {"text": "t"})), final()]
    r = run_react(ScriptedClient(script), registry, "x", config=ReactConfig(max_steps=10, max_rounds_per_tool=3))
    assert [log.ok for log in r.tool_log] == [True, True, True, False, True]
    assert tool_msgs(r.messages)[3]["content"].startswith("ERROR")
    assert len(registry.executed) == 4  # type: ignore[attr-defined]  # 3 adds + 1 echo


def test_max_steps(registry: ToolRegistry) -> None:
    loop = [turn(call(i, "echo", {"text": f"t{i}"})) for i in range(10)]
    r = run_react(ScriptedClient(loop), registry, "x", config=ReactConfig(max_steps=3, max_rounds_per_tool=99))
    assert r.stopped_reason == "max_steps" and r.steps == 3 and len(r.tool_log) == 3


def test_transient_model_errors_are_retried(registry: ToolRegistry) -> None:
    client = ScriptedClient([TransientModelError("429"), final("ok")])
    r = run_react(client, registry, "x", config=ReactConfig(max_model_retries=2))
    assert r.final_text == "ok" and r.model_retries == 1 and r.steps == 1


def test_model_error_after_retries(registry: ToolRegistry) -> None:
    r = run_react(ScriptedClient([TransientModelError("boom")]), registry, "x", config=ReactConfig(max_model_retries=2))
    assert r.stopped_reason == "model_error" and "boom" in r.final_text and r.model_retries == 2


def test_usage_is_accumulated(registry: ToolRegistry) -> None:
    r = run_react(ScriptedClient([turn(call(1, "echo", {"text": "a"}), tokens=(100, 20)), final()]), registry, "x")
    assert (r.usage.input_tokens, r.usage.output_tokens) == (105, 21)


def test_trip_tools_end_to_end_with_mock_policy() -> None:
    """Your loop driving the real travel tools with the same mock policy the LangGraph mock uses."""
    from react_from_scratch.client import MockResearchClient
    from react_from_scratch.registry import from_trip_tools
    from trip_agents.graph.prompts import RESEARCH_SYSTEM
    from trip_agents.llm.context import render_context

    req = {
        "hard": {
            "destination": "杭州",
            "start_date": "2026-11-20",
            "days": 2,
            "budget_total": 3000,
            "travelers": 2,
            "dietary_restrictions": [],
            "must_visit": [],
        },
        "soft": {"pace": "moderate", "hotel_tier": None, "interests": []},
    }
    r = run_react(
        MockResearchClient(),
        from_trip_tools(),
        "请为以下需求检索信息：\n" + render_context({"task": "research", "request": req}),
        system_prompt=RESEARCH_SYSTEM.format(max_rounds=3),
    )
    assert r.stopped_reason == "final_answer"
    assert {log.name for log in r.tool_log} >= {"get_weather", "search_hotels", "search_restaurants", "search_attractions"}
