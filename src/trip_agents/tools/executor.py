"""Single entry point for executing a tool call: validate -> run -> serialise.

Never raises for model-caused problems: bad arguments or unsupported inputs come
back as `ToolResult(ok=False, error=...)` so the error text can be fed back to the
model for self-correction (the ReAct "observation").
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from trip_agents.tools.mock_backend import MockBackend, ToolInputError
from trip_agents.tools.specs import TOOL_SPECS


class ToolBackend(Protocol):
    name: str

    def get_weather(self, inp: Any) -> Any: ...
    def search_hotels(self, inp: Any) -> Any: ...
    def search_restaurants(self, inp: Any) -> Any: ...
    def search_attractions(self, inp: Any) -> Any: ...
    def get_route(self, inp: Any) -> Any: ...


class ToolResult(BaseModel):
    name: str
    args: dict[str, Any]
    ok: bool
    data: Any = None
    error: str | None = None
    latency_ms: float = 0.0

    def to_observation(self) -> str:
        """Text returned to the model."""
        if self.ok:
            return json.dumps(self.data, ensure_ascii=False, default=str)
        return f"ERROR: {self.error}"


def format_validation_error(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "(root)"
        parts.append(f"{loc}: {err['msg']}")
    return "参数校验失败 -> " + "; ".join(parts)


@lru_cache
def get_backend(kind: str = "mock") -> ToolBackend:
    if kind == "mock":
        return MockBackend()
    if kind == "real":  # pragma: no cover - wired up in M3
        raise NotImplementedError("real backend arrives in M3")
    raise ValueError(f"unknown tool backend {kind!r}")


def execute_tool(name: str, args: dict[str, Any], backend: ToolBackend | None = None) -> ToolResult:
    backend = backend or get_backend()
    t0 = time.perf_counter()

    def done(**kw: Any) -> ToolResult:
        return ToolResult(name=name, args=args, latency_ms=round((time.perf_counter() - t0) * 1000, 2), **kw)

    spec = TOOL_SPECS.get(name)
    if spec is None:
        return done(ok=False, error=f"未知工具 '{name}'，可用工具: {', '.join(TOOL_SPECS)}")
    try:
        inp = spec.input_model.model_validate(args)
    except ValidationError as e:
        return done(ok=False, error=format_validation_error(e))
    try:
        out = getattr(backend, name)(inp)
    except ToolInputError as e:
        return done(ok=False, error=str(e))
    except Exception as e:  # backend/network failure: still an observation, not a crash
        return done(ok=False, error=f"工具执行异常: {type(e).__name__}: {e}")
    if isinstance(out, list):
        data: Any = [o.model_dump(mode="json") for o in out]
    else:
        data = out.model_dump(mode="json")
    return done(ok=True, data=data)
