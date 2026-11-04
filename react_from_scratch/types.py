"""Data types shared by the hand-written ReAct loop. No framework, just dataclasses.

Message format (OpenAI-style dicts, the de-facto lingua franca; client.py converts
for other providers):

    {"role": "system",    "content": "..."}
    {"role": "user",      "content": "..."}
    {"role": "assistant", "content": "..." | None,
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\\"city\\": \\"杭州\\"}"}}]}
    {"role": "tool",      "tool_call_id": "call_1", "content": "<observation text>"}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Message = dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens)


@dataclass
class RawToolCall:
    """A tool call exactly as the model produced it. `arguments` is the raw JSON string --
    it may be malformed, and parsing it is the loop's job."""

    id: str
    name: str
    arguments: str


@dataclass
class ModelTurn:
    text: str | None
    tool_calls: list[RawToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: Any = None

    def to_message(self) -> Message:
        """The assistant message to append to the conversation."""
        msg: Message = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in self.tool_calls
            ]
        return msg


@dataclass
class ParsedCall:
    id: str
    name: str
    args: dict[str, Any] | None  # None if the arguments could not be parsed
    parse_error: str | None = None


@dataclass
class Observation:
    ok: bool
    content: str  # what goes back to the model
    error: str | None = None
    data: Any = None


@dataclass
class ToolCallLog:
    step: int
    name: str
    args: dict[str, Any] | None
    ok: bool
    error: str | None = None
    duplicate: bool = False
    latency_ms: float = 0.0


@dataclass
class ReactConfig:
    max_steps: int = 8  # model calls
    max_rounds_per_tool: int = 3  # non-duplicate calls per tool name
    max_model_retries: int = 2  # retries of a single model call on TransientModelError


StopReason = Literal["final_answer", "max_steps", "model_error"]


@dataclass
class ReactResult:
    final_text: str
    stopped_reason: StopReason
    steps: int  # number of model calls that returned
    messages: list[Message]
    tool_log: list[ToolCallLog] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model_retries: int = 0
