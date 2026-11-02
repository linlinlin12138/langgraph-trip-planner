"""A LangChain-compatible chat model that needs no API key.

It is a real `BaseChatModel`, so it goes through the same code paths as a
provider model: `bind_tools`, `with_structured_output(include_raw=True)`,
callbacks (token usage is reported for the tracer) and LangGraph streaming.

Fault injection (`faults=`) lets tests and the demo deterministically trigger
the error-handling paths:
  * ``bad_json_once``          first structured call returns broken JSON
  * ``itinerary_overlap_once`` first day plan contains overlapping activities
  * ``itinerary_overlap_always`` every day plan overlaps (tests the re-plan cap)
  * ``bad_tool_args_once``     first research step passes an invalid date to a tool
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field, PrivateAttr

from trip_agents.llm import mock_responders as R
from trip_agents.llm.context import extract_context
from trip_agents.llm.structured import parse_and_validate


def _estimate_tokens(text: str) -> int:
    # ~1 token per CJK char, ~4 chars per token otherwise. Good enough for a mock.
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return max(1, cjk + (len(text) - cjk) // 4)


class MockChatModel(BaseChatModel):
    model_name: str = "mock-planner-v1"
    faults: set[str] = Field(default_factory=set)
    _calls: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "mock"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    # ------------------------------------------------------------------ core
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._calls += 1
        ctx = extract_context(messages)
        schema: type[BaseModel] | None = kwargs.get("mock_schema")
        if schema is not None:
            msg = AIMessage(content=self._structured(schema, messages, ctx))
        elif kwargs.get("tools"):
            msg = R.respond_research(messages, ctx, self.faults)
        else:
            msg = AIMessage(content="好的。")

        prompt_text = "".join(m.content if isinstance(m.content, str) else str(m.content) for m in messages)
        out_text = msg.content if isinstance(msg.content, str) else ""
        out_text += json.dumps([tc["args"] for tc in msg.tool_calls], ensure_ascii=False)
        usage = {"input_tokens": _estimate_tokens(prompt_text), "output_tokens": _estimate_tokens(out_text)}
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        msg.usage_metadata = usage  # type: ignore[assignment]
        msg.response_metadata = {"model_name": self.model_name}
        return ChatResult(generations=[ChatGeneration(message=msg)], llm_output={"model_name": self.model_name})

    def _structured(self, schema: type[BaseModel], messages: list[BaseMessage], ctx: dict[str, Any]) -> str:
        if "bad_json_once" in self.faults:
            self.faults.discard("bad_json_once")
            return '```json\n{"this is": "not valid", \n```'
        responders: dict[str, Callable[[], dict[str, Any]]] = {
            "RequestExtraction": lambda: R.respond_extract(messages, ctx),
            "DayPlan": lambda: R.respond_plan_day(ctx, self.faults),
            "BudgetAdvice": lambda: R.respond_budget_advice(ctx),
            "Itinerary": lambda: R.respond_baseline(messages, ctx),
            "JudgeScores": lambda: R.respond_judge(ctx),
        }
        fn = responders.get(schema.__name__)
        if fn is None:
            raise NotImplementedError(f"MockChatModel has no responder for schema {schema.__name__}")
        return json.dumps(fn(), ensure_ascii=False)

    # ------------------------------------------------------------------ interface parity
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        assert isinstance(schema, type) and issubclass(schema, BaseModel), "mock supports Pydantic schemas only"

        def parse(msg: AIMessage) -> dict[str, Any] | BaseModel:
            try:
                parsed: BaseModel | None = parse_and_validate(str(msg.content), schema)
                err = None
            except ValueError as e:
                parsed, err = None, e
            if include_raw:
                return {"raw": msg, "parsed": parsed, "parsing_error": err}
            if err:
                raise err
            assert parsed is not None
            return parsed

        return self.bind(mock_schema=schema) | RunnableLambda(parse)
