"""Dependencies and switches injected into the graph (no globals inside nodes).

Injection keeps nodes unit-testable (pass a MockChatModel with faults) and lets
the eval harness build ablated graphs from the same code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel

from trip_agents.cache.semantic import SemanticCache
from trip_agents.config import get_settings
from trip_agents.tools.executor import ToolBackend, ToolResult, execute_tool, get_backend
from trip_agents.tools.specs import TOOL_SPECS


class ToolExecutor(Protocol):
    """How the Research agent sees tools. M1: in-process. M3: via an MCP client."""

    def tool_schemas(self) -> list[dict[str, Any]]: ...
    def call(self, name: str, args: dict[str, Any]) -> ToolResult: ...


class LocalToolExecutor:
    def __init__(self, backend: ToolBackend | None = None) -> None:
        self.backend = backend or get_backend(get_settings().tool_backend)

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {"name": s.name, "description": s.description, "parameters": s.input_model.model_json_schema()},
            }
            for s in TOOL_SPECS.values()
        ]

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        return execute_tool(name, args, self.backend)


def make_tool_executor(transport: str | None = None) -> ToolExecutor:
    transport = transport or get_settings().tool_transport
    if transport == "mcp":
        from trip_agents.tools.mcp_client import McpToolExecutor

        return McpToolExecutor.stdio()
    return LocalToolExecutor()


@dataclass
class GraphOptions:
    """Behaviour switches. Ablations in the eval harness flip these."""

    enable_revision: bool = True  # conditional edge validate -> itinerary
    max_revisions: int = field(default_factory=lambda: get_settings().max_revisions)
    human_in_loop: bool = True  # False: auto-pick defaults, never interrupt (evals)
    max_clarify_rounds: int = 3
    budget_advice: bool = True  # LLM advice when over budget
    enable_memory: bool = True  # read/write long-term memory (needs the graph compiled with a store)


@dataclass
class AgentDeps:
    model: BaseChatModel
    tools: ToolExecutor
    today: date = field(default_factory=date.today)
    clock: Callable[[], datetime] = datetime.now
    cache: SemanticCache | None = None  # optional semantic cache of confirmed plans
    structured_max_retries: int = field(default_factory=lambda: get_settings().structured_max_retries)
    research_max_steps: int = field(default_factory=lambda: get_settings().research_max_steps)
    research_max_rounds_per_tool: int = field(default_factory=lambda: get_settings().research_max_rounds_per_tool)
