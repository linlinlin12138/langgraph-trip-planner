from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from trip_agents.llm.mock import MockChatModel
from trip_agents.llm.structured import clean_json_text, invoke_structured, parse_and_validate
from trip_agents.schemas import RequestExtraction


class Point(BaseModel):
    x: int
    y: int = Field(ge=0)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"x": 1, "y": 2}', '{"x": 1, "y": 2}'),
        ('```json\n{"x": 1, "y": 2}\n```', '{"x": 1, "y": 2}'),
        ('好的，结果如下：{"x": 1, "y": 2} 希望有帮助', '{"x": 1, "y": 2}'),
        ('{"x": 1, "y": 2,}', '{"x": 1, "y": 2}'),
        ("[1, 2, 3,]", "[1, 2, 3]"),
    ],
)
def test_clean_json_text(raw: str, expected: str) -> None:
    assert clean_json_text(raw) == expected


def test_parse_and_validate_reports_field_errors() -> None:
    assert parse_and_validate('```{"x": 1, "y": 3}```', Point) == Point(x=1, y=3)
    with pytest.raises(ValueError, match="不是合法 JSON"):
        parse_and_validate("{x: 1", Point)
    with pytest.raises(ValueError, match="y"):
        parse_and_validate('{"x": 1, "y": -1}', Point)


class SequenceModel(BaseChatModel):
    """Structured model returning given raw texts in order; records the prompts it saw."""

    outputs: list[str]
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "seq"

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.outputs[len(self.seen) - 1]))])

    def with_structured_output(self, schema: Any, *, include_raw: bool = False, **kw: Any) -> Any:  # type: ignore[override]
        return self | RunnableLambda(lambda m: {"raw": m, "parsed": None, "parsing_error": "native failed"})


def test_retry_feeds_back_error_then_succeeds() -> None:
    model = SequenceModel(outputs=["not json", '{"x": 1, "y": -5}', '{"x": 1, "y": 5}'])
    res = invoke_structured(model, Point, [HumanMessage(content="give point")], max_retries=3)
    assert res.ok and res.value == Point(x=1, y=5) and res.attempts == 3
    assert len(res.errors) == 2
    # the second prompt contains the first error, the third contains the schema error
    assert "不是合法 JSON" in model.seen[1][-1].content  # type: ignore[operator]
    assert "y" in model.seen[2][-1].content  # type: ignore[operator]


def test_gives_up_after_max_retries() -> None:
    model = SequenceModel(outputs=["x", "y", "z"])
    res = invoke_structured(model, Point, [HumanMessage(content="p")], max_retries=3)
    assert not res.ok and res.attempts == 3 and len(res.errors) == 3


def test_extra_check_failure_is_retried() -> None:
    model = SequenceModel(outputs=['{"x": 1, "y": 1}', '{"x": 2, "y": 1}'])
    res = invoke_structured(model, Point, [HumanMessage(content="p")], extra_check=lambda p: [] if p.x == 2 else ["x 必须为 2"])
    assert res.value == Point(x=2, y=1)
    assert "x 必须为 2" in model.seen[1][-1].content  # type: ignore[operator]


def test_mock_bad_json_fault_recovers() -> None:
    m = MockChatModel(faults={"bad_json_once"})
    res = invoke_structured(m, RequestExtraction, [HumanMessage(content="去杭州玩3天")])
    assert res.ok and res.attempts == 2 and res.value and res.value.destination == "杭州"


def test_timezone_suffixed_times_become_local() -> None:
    """Gemini sometimes returns "09:00:00Z"; comparing aware and naive times raised TypeError in validation."""
    from datetime import time

    from trip_agents.schemas import Activity

    a = Activity.model_validate(
        {"start_time": "09:00:00Z", "end_time": "10:30+08:00", "kind": "attraction", "title": "灵隐寺", "ref_id": "a1", "est_cost": 0}
    )
    assert a.start_time == time(9, 0) and a.end_time == time(10, 30)
    assert a.start_time.tzinfo is None and a.end_time < time(23, 30)
