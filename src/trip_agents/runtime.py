"""Thin session API over the compiled graph, shared by the CLI, Streamlit app and tests.

Hides LangGraph plumbing (thread config, `Command(resume=...)`, reading pending
interrupts) behind `start / resume / stream`.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Command

from trip_agents.cache.semantic import SemanticCache
from trip_agents.config import get_settings
from trip_agents.graph.builder import build_graph, sqlite_checkpointer
from trip_agents.graph.deps import AgentDeps, GraphOptions, make_tool_executor
from trip_agents.llm.factory import get_chat_model
from trip_agents.memory.store import sqlite_store


def create_planner(
    *,
    provider: str | None = None,
    model: str | None = None,
    faults: set[str] | None = None,
    options: GraphOptions | None = None,
    checkpoint_path: str | None = None,
    store: BaseStore | None = None,
    today: date | None = None,
    semantic_cache: bool | None = None,
) -> CompiledStateGraph[Any]:
    s = get_settings()
    kwargs: dict[str, Any] = {"faults": set(faults)} if faults and (provider or s.llm_provider) == "mock" else {}
    llm = get_chat_model(provider, model, **kwargs)  # type: ignore[arg-type]
    use_cache = s.semantic_cache if semantic_cache is None else semantic_cache
    cache = SemanticCache(s.cache_db, threshold=s.cache_threshold, ttl_s=s.cache_ttl_days * 86400) if use_cache else None
    deps = AgentDeps(model=llm, tools=make_tool_executor(), today=today or date.today(), cache=cache)
    cp = sqlite_checkpointer(checkpoint_path or s.checkpoint_db)
    return build_graph(deps, options, checkpointer=cp, store=store if store is not None else sqlite_store(s.store_db))


@dataclass
class Turn:
    """What the caller needs after each step."""

    new_messages: list[str] = field(default_factory=list)
    interrupt: dict[str, Any] | None = None
    nodes: list[str] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.interrupt is None


class PlannerSession:
    def __init__(
        self,
        graph: CompiledStateGraph[Any],
        thread_id: str | None = None,
        user_id: str = "demo-user",
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> None:
        self.graph = graph
        self.callbacks = callbacks or []
        self.thread_id = thread_id or uuid.uuid4().hex[:12]
        self.user_id = user_id

    @property
    def config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.thread_id, "user_id": self.user_id}, "callbacks": self.callbacks}

    # ---------------------------------------------------------------- driving
    def stream(self, payload: Any) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield (node_name, update) as each node finishes; for live UIs."""
        for chunk in self.graph.stream(payload, self.config, stream_mode="updates"):  # type: ignore[call-overload]
            for node, update in chunk.items():
                if node != "__interrupt__":
                    yield node, update or {}

    def _run(self, payload: Any) -> Turn:
        turn = Turn()
        for node, update in self.stream(payload):
            turn.nodes.append(node)
            for m in update.get("messages", []):
                if isinstance(m, AIMessage) and isinstance(m.content, str):
                    turn.new_messages.append(m.content)
        turn.interrupt = self.pending_interrupt()
        return turn

    def start(self, text: str) -> Turn:
        return self._run({"messages": [HumanMessage(content=text)], "user_id": self.user_id})

    def resume(self, value: Any) -> Turn:
        return self._run(Command(resume=value))

    # ---------------------------------------------------------------- inspection
    def pending_interrupt(self) -> dict[str, Any] | None:
        snap = self.graph.get_state(self.config)  # type: ignore[arg-type]
        for task in snap.tasks:
            for intr in task.interrupts:
                value: dict[str, Any] = intr.value
                return value
        return None

    @property
    def values(self) -> dict[str, Any]:
        return dict(self.graph.get_state(self.config).values)  # type: ignore[arg-type]
