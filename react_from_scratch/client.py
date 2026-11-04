"""Thin wrappers over the raw model SDKs, normalised to `ModelTurn`. (Provided.)

The loop only depends on the `ChatClient` protocol:

    turn = client.complete(messages, tools)      # -> ModelTurn(text, tool_calls, usage)

Transient failures (rate limit, timeout, 5xx) are raised as `TransientModelError`
so the loop can decide whether to retry; anything else propagates.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any, Protocol

from react_from_scratch.types import Message, ModelTurn, RawToolCall, Usage


class TransientModelError(RuntimeError):
    """Retryable model-call failure."""


class ChatClient(Protocol):
    model: str

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelTurn: ...


# --------------------------------------------------------------------------- OpenAI (and compatible)
class OpenAIClient:
    def __init__(self, model: str, *, base_url: str | None = None, api_key: str | None = None, temperature: float = 0.2) -> None:
        import openai

        self.model, self.temperature = model, temperature
        self._openai = openai
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelTurn:
        o = self._openai
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
        try:
            r = self._client.chat.completions.create(**kwargs)
        except (o.RateLimitError, o.APITimeoutError, o.APIConnectionError, o.InternalServerError) as e:
            raise TransientModelError(str(e)) from e
        msg = r.choices[0].message
        calls = [RawToolCall(c.id, c.function.name, c.function.arguments) for c in (msg.tool_calls or []) if c.type == "function"]
        usage = Usage(r.usage.prompt_tokens, r.usage.completion_tokens) if r.usage else Usage()
        return ModelTurn(text=msg.content, tool_calls=calls, usage=usage, raw=r)


def GeminiClient(model: str) -> OpenAIClient:
    """Gemini through its OpenAI-compatible endpoint."""
    return OpenAIClient(
        model, base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GOOGLE_API_KEY")
    )


# --------------------------------------------------------------------------- Anthropic
def to_anthropic(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """OpenAI-style history -> (system, Anthropic messages). Consecutive tool results are
    merged into one user turn of tool_result blocks, as the Messages API requires."""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        if role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            blocks: list[dict[str, Any]] = [{"type": "text", "text": m["content"]}] if m.get("content") else []
            for c in m.get("tool_calls", []):
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": c["function"]["arguments"]}
                blocks.append({"type": "tool_use", "id": c["id"], "name": c["function"]["name"], "input": args})
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m["content"],
                "is_error": str(m["content"]).startswith("ERROR"),
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return system, out


class AnthropicClient:
    def __init__(self, model: str, *, max_tokens: int = 2048, temperature: float = 0.2) -> None:
        import anthropic

        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelTurn:
        a = self._anthropic
        system, msgs = to_anthropic(messages)
        atools = [
            {"name": t["function"]["name"], "description": t["function"]["description"], "input_schema": t["function"]["parameters"]}
            for t in tools
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": msgs,
            "tools": atools,
        }
        try:
            r = self._client.messages.create(**kwargs)
        except (a.RateLimitError, a.APITimeoutError, a.APIConnectionError, a.InternalServerError) as e:
            raise TransientModelError(str(e)) from e
        text = "".join(b.text for b in r.content if b.type == "text") or None
        calls = [RawToolCall(b.id, b.name, json.dumps(b.input, ensure_ascii=False)) for b in r.content if b.type == "tool_use"]
        return ModelTurn(text=text, tool_calls=calls, usage=Usage(r.usage.input_tokens, r.usage.output_tokens), raw=r)


# --------------------------------------------------------------------------- offline clients
class ScriptedClient:
    """Plays back a script of turns (or exceptions). Records every request it receives."""

    model = "scripted"

    def __init__(self, script: Sequence[ModelTurn | Exception]) -> None:
        self.script = list(script)
        self.requests: list[list[Message]] = []

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelTurn:
        self.requests.append([dict(m) for m in messages])
        item = self.script[min(len(self.requests) - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


class MockResearchClient:
    """The same deterministic research policy as the LangGraph mock, behind the SDK-level interface,
    so both implementations can be compared offline on identical behaviour."""

    model = "mock-planner-v1"

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelTurn:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        from trip_agents.llm import mock_responders as R
        from trip_agents.llm.context import extract_context
        from trip_agents.llm.mock import _estimate_tokens

        lc: list[Any] = []
        for m in messages:
            if m["role"] == "system":
                lc.append(SystemMessage(content=m["content"]))
            elif m["role"] == "user":
                lc.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                lc_calls = [
                    {
                        "id": c["id"],
                        "name": c["function"]["name"],
                        "args": json.loads(c["function"]["arguments"] or "{}"),
                        "type": "tool_call",
                    }
                    for c in m.get("tool_calls", [])
                ]
                lc.append(AIMessage(content=m.get("content") or "", tool_calls=lc_calls))
            elif m["role"] == "tool":
                lc.append(ToolMessage(content=m["content"], tool_call_id=m["tool_call_id"]))
        ai = R.respond_research(lc, extract_context(lc), set())
        calls = [RawToolCall(tc["id"] or "", tc["name"], json.dumps(tc["args"], ensure_ascii=False)) for tc in ai.tool_calls]
        prompt = "".join(str(m.get("content") or "") for m in messages)
        out = (ai.content if isinstance(ai.content, str) else "") + "".join(c.arguments for c in calls)
        return ModelTurn(text=ai.content or None, tool_calls=calls, usage=Usage(_estimate_tokens(prompt), _estimate_tokens(out)))  # type: ignore[arg-type]


def make_client(provider: str, model: str | None = None) -> ChatClient:
    from trip_agents.llm.factory import DEFAULT_MODELS

    model = model or DEFAULT_MODELS[provider]
    if provider == "mock":
        return MockResearchClient()
    if provider == "openai":
        return OpenAIClient(model)
    if provider == "anthropic":
        return AnthropicClient(model)
    if provider == "gemini":
        return GeminiClient(model)
    raise ValueError(provider)
