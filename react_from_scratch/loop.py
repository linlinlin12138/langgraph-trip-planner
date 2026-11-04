"""A framework-free ReAct loop: only the model SDK (via `client.py`) and the tool registry.

    uv run pytest react_from_scratch            # acceptance tests
    make compare-react                          # vs the LangGraph Research subgraph

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

Observations that are errors start with "ERROR" so the model (and the Anthropic
adapter) can tell. Every tool_call id gets exactly one tool message -- providers
reject a history where an assistant tool call has no matching result.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from react_from_scratch.client import ChatClient, TransientModelError
from react_from_scratch.registry import ToolRegistry
from react_from_scratch.types import Message, ModelTurn, ParsedCall, ReactConfig, ReactResult, ToolCallLog, Usage

DEFAULT_SYSTEM = "你是一个会使用工具的助手。需要信息时调用工具；信息足够时直接回答，不要再调用工具。"
DUPLICATE_NOTICE = "（重复调用：与之前参数完全相同，未再次执行，以下是之前的结果。请基于已有信息继续，不要重复调用）\n"


def call_key(name: str, args: dict[str, Any]) -> str:
    """Canonical identity of a call, for duplicate detection: key order doesn't matter."""
    return name + ":" + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def parse_tool_calls(turn: ModelTurn) -> list[ParsedCall]:
    """Raw tool calls -> ParsedCall. Never raises: bad arguments become `parse_error`,
    which the loop turns into an observation so the model can fix its own call."""
    parsed: list[ParsedCall] = []
    for c in turn.tool_calls:
        raw = (c.arguments or "").strip()
        if not raw:
            parsed.append(ParsedCall(c.id, c.name, {}))
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as e:
            parsed.append(ParsedCall(c.id, c.name, None, f"参数不是合法 JSON（{e.msg}，第 {e.colno} 列），请输出一个 JSON 对象"))
            continue
        if not isinstance(value, dict):
            parsed.append(ParsedCall(c.id, c.name, None, f"参数必须是 JSON 对象，实际是 {type(value).__name__}"))
            continue
        parsed.append(ParsedCall(c.id, c.name, value))
    return parsed


@dataclass
class _LoopState:
    """Everything the loop accumulates; kept in one place so run_react reads top to bottom."""

    messages: list[Message]
    usage: Usage = field(default_factory=Usage)
    tool_log: list[ToolCallLog] = field(default_factory=list)
    model_retries: int = 0
    steps: int = 0
    successes: dict[str, str] = field(default_factory=dict)  # call_key -> observation of the successful call
    rounds: dict[str, int] = field(default_factory=dict)  # tool name -> non-duplicate calls so far

    def result(self, final_text: str, reason: Any) -> ReactResult:
        return ReactResult(final_text, reason, self.steps, self.messages, self.tool_log, self.usage, self.model_retries)


def _complete_with_retry(client: ChatClient, registry: ToolRegistry, st: _LoopState, cfg: ReactConfig) -> ModelTurn:
    """One model call; retries only TransientModelError (rate limit / timeout / 5xx).
    Anything else (bad request, auth) propagates: retrying would not help."""
    attempt = 0
    while True:
        try:
            return client.complete(st.messages, registry.schemas())
        except TransientModelError:
            if attempt >= cfg.max_model_retries:
                raise
            attempt += 1
            st.model_retries += 1


def _handle_call(pc: ParsedCall, registry: ToolRegistry, st: _LoopState, cfg: ReactConfig) -> tuple[str, ToolCallLog]:
    """Decide what happens to one tool call; returns (observation, log entry)."""
    t0 = time.perf_counter()

    def log(ok: bool, error: str | None = None, duplicate: bool = False) -> ToolCallLog:
        return ToolCallLog(st.steps, pc.name, pc.args, ok, error, duplicate, round((time.perf_counter() - t0) * 1000, 2))

    if pc.args is None:  # malformed arguments: tell the model exactly what was wrong
        return f"ERROR: {pc.parse_error}", log(False, pc.parse_error)

    key = call_key(pc.name, pc.args)
    if key in st.successes:  # same call already succeeded: don't spend another execution
        return DUPLICATE_NOTICE + st.successes[key], log(True, duplicate=True)

    if st.rounds.get(pc.name, 0) >= cfg.max_rounds_per_tool:
        msg = f"工具 {pc.name} 已达到最多 {cfg.max_rounds_per_tool} 轮的调用上限，请基于已有信息继续"
        return f"ERROR: {msg}", log(False, "round_limit")

    st.rounds[pc.name] = st.rounds.get(pc.name, 0) + 1  # failed calls count too: they cost a round
    obs = registry.execute(pc.name, pc.args)  # validation / unknown tool / tool exceptions -> observation
    if obs.ok:
        st.successes[key] = obs.content  # only successes are cached, so a corrected retry is allowed
    return obs.content, log(obs.ok, obs.error)


def run_react(
    client: ChatClient,
    registry: ToolRegistry,
    task: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM,
    config: ReactConfig | None = None,
) -> ReactResult:
    """Run the ReAct loop described in the module docstring."""
    cfg = config or ReactConfig()
    st = _LoopState(messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": task}])

    for _ in range(cfg.max_steps):
        # 1. think: ask the model what to do next
        try:
            turn = _complete_with_retry(client, registry, st, cfg)
        except TransientModelError as e:
            return st.result(str(e), "model_error")
        st.steps += 1
        st.usage = st.usage + turn.usage
        st.messages.append(turn.to_message())

        # 2. no tool call means the model is done
        if not turn.tool_calls:
            return st.result(turn.text or "", "final_answer")

        # 3. act + observe: every call gets exactly one tool message, in order
        for pc in parse_tool_calls(turn):
            observation, entry = _handle_call(pc, registry, st, cfg)
            st.messages.append({"role": "tool", "tool_call_id": pc.id, "content": observation})
            st.tool_log.append(entry)

    return st.result("", "max_steps")
