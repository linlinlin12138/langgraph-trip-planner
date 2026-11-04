"""Tool registry: register Python functions with a Pydantic input schema, export JSON
schemas for the model, validate arguments, execute. (Provided -- not part of the exercise.)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from react_from_scratch.types import Observation


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[BaseModel], Any]


def _fmt_validation(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(x) for x in err['loc']) or '(root)'}: {err['msg']}" for err in e.errors())


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, name: str, description: str, input_model: type[BaseModel], fn: Callable[[BaseModel], Any]) -> None:
        if name in self._tools:
            raise ValueError(f"tool {name!r} already registered")
        self._tools[name] = RegisteredTool(name, description, input_model, fn)

    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    # ---------------------------------------------------------------- schemas for the model
    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI "tools" format; client.py converts for other providers."""
        return [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.input_model.model_json_schema()},
            }
            for t in self._tools.values()
        ]

    # ---------------------------------------------------------------- validation / execution
    def validate(self, name: str, args: dict[str, Any]) -> tuple[BaseModel | None, str | None]:
        """(validated_input, None) or (None, error message for the model)."""
        tool = self._tools.get(name)
        if tool is None:
            return None, f"未知工具 '{name}'，可用工具: {', '.join(self._tools)}"
        try:
            return tool.input_model.model_validate(args), None
        except ValidationError as e:
            return None, f"参数校验失败 -> {_fmt_validation(e)}"

    def execute(self, name: str, args: dict[str, Any]) -> Observation:
        """Validate then run. Never raises for model-caused problems: errors become observations."""
        inp, err = self.validate(name, args)
        if inp is None:
            return Observation(ok=False, content=f"ERROR: {err}", error=err)
        try:
            out = self._tools[name].fn(inp)
        except Exception as e:  # tool bug / backend failure is still an observation
            msg = f"工具执行异常: {type(e).__name__}: {e}"
            return Observation(ok=False, content=f"ERROR: {msg}", error=msg)
        if isinstance(out, Observation):
            return out
        return Observation(ok=True, content=json.dumps(out, ensure_ascii=False, default=str), data=out)


def from_trip_tools(executor: Any = None) -> ToolRegistry:
    """The same five travel tools the LangGraph version uses (in-process executor by default)."""
    from trip_agents.graph.deps import LocalToolExecutor
    from trip_agents.tools.specs import TOOL_SPECS

    ex = executor or LocalToolExecutor()
    reg = ToolRegistry()
    for spec in TOOL_SPECS.values():

        def run(inp: BaseModel, _name: str = spec.name) -> Observation:
            res = ex.call(_name, inp.model_dump(mode="json", exclude_none=True))
            return Observation(ok=res.ok, content=res.to_observation(), error=res.error, data=res.data)

        reg.register(spec.name, spec.description, spec.input_model, run)
    return reg
