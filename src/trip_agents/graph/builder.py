"""Assemble the StateGraph.

START → chat ─┬─(missing info / memory defaults)→ clarify ⟲ chat   [interrupt #1, optional]
              └─(complete)→ cache_lookup ─(hit)→ budget (re-validate the cached plan)
                                         └─(miss)→ research → human_select   [interrupt #2: pick candidates]
                 → itinerary → budget → validate ─┬─(errors & retries left)→ itinerary
                                                  └─→ finalize → confirm → END   [interrupt #3]
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from trip_agents.graph.deps import AgentDeps, GraphOptions
from trip_agents.graph.nodes.budget import make_budget_node
from trip_agents.graph.nodes.cache import make_cache_lookup_node, route_after_cache
from trip_agents.graph.nodes.chat import make_chat_node, make_clarify_node, route_after_chat
from trip_agents.graph.nodes.finalize import finalize, make_confirm_node
from trip_agents.graph.nodes.human_select import make_human_select_node
from trip_agents.graph.nodes.itinerary import make_itinerary_node
from trip_agents.graph.nodes.research import make_research_node
from trip_agents.graph.nodes.validate import route_after_validate, validate
from trip_agents.graph.state import TripState
from trip_agents.schemas import __all__ as _schema_names


def checkpoint_serializer() -> JsonPlusSerializer:
    """Explicitly allow our own Pydantic state types to be deserialised from checkpoints
    (LangGraph blocks unregistered types by default for safety)."""
    import trip_agents.schemas as schemas

    allowed = [(getattr(schemas, n).__module__, n) for n in _schema_names]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def build_graph(
    deps: AgentDeps,
    options: GraphOptions | None = None,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph[Any]:
    options = options or GraphOptions()
    g = StateGraph(TripState)
    g.add_node("chat", make_chat_node(deps, options))
    g.add_node("clarify", make_clarify_node(options))
    g.add_node("cache_lookup", make_cache_lookup_node(deps.cache))
    g.add_node("research", make_research_node(deps, options))
    g.add_node("human_select", make_human_select_node(options))
    g.add_node("itinerary", make_itinerary_node(deps))
    g.add_node("budget", make_budget_node(deps, options))
    g.add_node("validate", validate)
    g.add_node("finalize", finalize)
    g.add_node("confirm", make_confirm_node(deps, options))

    g.add_edge(START, "chat")
    g.add_conditional_edges("chat", route_after_chat(options), {"clarify": "clarify", "research": "cache_lookup", END: END})
    g.add_conditional_edges("cache_lookup", route_after_cache, ["research", "budget"])
    g.add_edge("clarify", "chat")
    g.add_edge("research", "human_select")
    g.add_edge("human_select", "itinerary")
    g.add_edge("itinerary", "budget")
    g.add_edge("budget", "validate")
    g.add_conditional_edges("validate", route_after_validate(options), ["itinerary", "finalize"])
    g.add_edge("finalize", "confirm")
    g.add_edge("confirm", END)
    return g.compile(checkpointer=checkpointer, store=store)


def sqlite_checkpointer(path: str | Path) -> SqliteSaver:
    """File-backed checkpointer: a thread survives process restarts."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn, serde=checkpoint_serializer())
