"""Robust structured output: native schema mode first, then clean -> parse -> validate,
retrying with the concrete error message, and giving up gracefully.

Two layers of checks:
* schema validation (Pydantic) -- types, ranges, field validators;
* `extra_check` -- caller-supplied business checks that need context the schema
  doesn't have (e.g. "ref_id must be one of the selected attractions").
Both kinds of failure are fed back to the model verbatim on the next attempt.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

STRUCTURED_EVENT = "structured_output"


@dataclass
class StructuredResult(Generic[T]):
    value: T | None
    attempts: int
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None


# --------------------------------------------------------------------------- cleaning
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def clean_json_text(text: str) -> str:
    """Best-effort repair of typical LLM JSON noise: code fences, prose around the
    object, trailing commas, smart quotes. Does not attempt to invent content."""
    t = text.strip()
    m = _FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()
    start = min((i for i in (t.find("{"), t.find("[")) if i >= 0), default=-1)
    if start > 0:
        t = t[start:]
    if t and t[0] in "{[":
        closing = "}" if t[0] == "{" else "]"
        end = t.rfind(closing)
        if end >= 0:
            t = t[: end + 1]
    t = t.replace("“", '"').replace("”", '"')
    t = _TRAILING_COMMA_RE.sub(r"\1", t)
    return t


def format_validation_error(e: ValidationError) -> str:
    lines = []
    for err in e.errors()[:10]:
        loc = ".".join(str(x) for x in err["loc"]) or "(root)"
        lines.append(f"- {loc}: {err['msg']}")
    return "\n".join(lines)


def parse_and_validate(text: str, schema: type[T]) -> T:
    """Raises ValueError with a model-readable message on failure."""
    cleaned = clean_json_text(text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"输出不是合法 JSON: {e.msg} (line {e.lineno} col {e.colno})") from e
    try:
        return schema.model_validate(obj)
    except ValidationError as e:
        raise ValueError("JSON 不符合 schema:\n" + format_validation_error(e)) from e


# --------------------------------------------------------------------------- invoke
def _emit(name: str, data: dict[str, Any], config: RunnableConfig | None) -> None:
    """Report to tracer callbacks; silently no-op when not running inside a Runnable."""
    with contextlib.suppress(RuntimeError):
        dispatch_custom_event(name, data, config=config)


def _raw_text(raw: Any) -> str:
    content = getattr(raw, "content", raw)
    if isinstance(content, list):  # anthropic-style content blocks
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content or "")


def invoke_structured(
    model: BaseChatModel,
    schema: type[T],
    messages: Sequence[BaseMessage],
    *,
    max_retries: int = 3,
    extra_check: Callable[[T], list[str]] | None = None,
    stage: str = "",
    item: str = "",
    config: RunnableConfig | None = None,
) -> StructuredResult[T]:
    runnable = model.with_structured_output(schema, include_raw=True, method="json_schema")
    history = list(messages)
    errors: list[str] = []

    for attempt in range(1, max_retries + 1):
        error: str | None = None
        value: T | None = None
        raw_text = ""
        try:
            out: dict[str, Any] = runnable.invoke(history, config=config)  # type: ignore[assignment]
            parsed, parsing_error, raw = out["parsed"], out.get("parsing_error"), out.get("raw")
            raw_text = _raw_text(raw)
            if isinstance(parsed, schema):
                value = parsed
            elif isinstance(parsed, dict):
                value = schema.model_validate(parsed)
            elif raw_text:
                # Native parsing failed (or provider returned prose): fall back to our own cleaner.
                value = parse_and_validate(raw_text, schema)
            else:
                error = f"模型未返回结构化结果: {parsing_error}"
        except ValidationError as e:
            error = "JSON 不符合 schema:\n" + format_validation_error(e)
        except ValueError as e:
            error = str(e)

        if value is not None and extra_check is not None:
            problems = extra_check(value)
            if problems:
                error = "内容校验失败:\n" + "\n".join(f"- {p}" for p in problems)
                value = None

        if value is not None:
            _emit(STRUCTURED_EVENT, {"stage": stage, "item": item, "attempt": attempt, "ok": True}, config)
            return StructuredResult(value=value, attempts=attempt, errors=errors)

        assert error is not None
        errors.append(error)
        _emit(STRUCTURED_EVENT, {"stage": stage, "item": item, "attempt": attempt, "ok": False, "error": error}, config)
        prev = f"你上一次的输出:\n{raw_text[:2000]}\n\n" if raw_text else ""
        history.append(HumanMessage(content=f"{prev}存在以下问题，请修正后重新输出完整结果:\n{error}"))

    return StructuredResult(value=None, attempts=max_retries, errors=errors)
