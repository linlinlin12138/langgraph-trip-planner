"""The ReAct loop -- YOUR exercise. Only `types.py`, `registry.py` and `client.py` are provided.

Run the tests (they are skipped until `run_react` stops raising NotImplementedError):

    uv run pytest react_from_scratch -rs

Contract (the tests check every point):

    messages = [system, user(task)]
    for step in 1..config.max_steps:
        turn = client.complete(messages, registry.schemas())
            - on TransientModelError retry up to config.max_model_retries times
              (count retries in result.model_retries); if still failing ->
              stopped_reason="model_error", final_text=str(error)
            - accumulate turn.usage
        append turn.to_message()
        if no tool calls -> stopped_reason="final_answer", final_text=turn.text or ""
        for each call, in order:
            parse arguments (parse_tool_calls)          malformed JSON -> error observation
            duplicate? (same name + same args as an earlier *successful* call)
                -> do NOT execute; observation = notice + the earlier observation; log duplicate=True
            round limit? (> max_rounds_per_tool non-duplicate calls of this tool)
                -> do NOT execute; error observation
            else registry.execute(name, args)            validation errors come back as observations
            append {"role": "tool", "tool_call_id": call.id, "content": observation}
            append a ToolCallLog (step = current step number, starting at 1)
    loop exhausted -> stopped_reason="max_steps"

Observations that are errors must start with "ERROR" so the model (and the
Anthropic adapter) can tell. Every tool_call id must get exactly one tool message.
"""

from __future__ import annotations

import json
from typing import Any

from react_from_scratch.client import ChatClient, TransientModelError  # noqa: F401  (you will need these)
from react_from_scratch.registry import ToolRegistry
from react_from_scratch.types import ModelTurn, ParsedCall, ReactConfig, ReactResult, ToolCallLog, Usage  # noqa: F401

DEFAULT_SYSTEM = "你是一个会使用工具的助手。需要信息时调用工具；信息足够时直接回答，不要再调用工具。"


def call_key(name: str, args: dict[str, Any]) -> str:
    """Canonical identity of a call, for duplicate detection (provided)."""
    return name + ":" + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def parse_tool_calls(turn: ModelTurn) -> list[ParsedCall]:
    """Turn the model's raw tool calls into ParsedCall objects.

    TODO:
      - json.loads each `arguments` string; an empty string means {}
      - arguments that are not valid JSON, or not a JSON object, -> args=None and a
        parse_error message the model can act on (include the JSON error)
      - keep the original order and ids
    """
    raise NotImplementedError("TODO: implement parse_tool_calls")


def run_react(
    client: ChatClient,
    registry: ToolRegistry,
    task: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM,
    config: ReactConfig | None = None,
) -> ReactResult:
    """Run the ReAct loop described in the module docstring and return a ReactResult.

    TODO: implement. Suggested shape (~60-80 lines):
      1. init messages, usage, tool_log, model_retries, successful-call cache, per-tool counters
      2. step loop with the model call + retry
      3. final-answer exit
      4. per-call handling: parse -> duplicate -> round limit -> execute -> append tool message + log
      5. max_steps exit
    """
    raise NotImplementedError("TODO: implement run_react")
