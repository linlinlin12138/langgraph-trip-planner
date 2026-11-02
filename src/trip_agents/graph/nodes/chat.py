"""Chat agent: turn the conversation into hard constraints + soft preferences.

The LLM only *extracts*; deciding whether we can proceed (which required fields
are missing) and phrasing the follow-up question are deterministic, so the
routing never depends on the model's mood.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store
from langgraph.graph import END
from langgraph.types import interrupt
from pydantic import ValidationError

from trip_agents.graph.deps import AgentDeps, GraphOptions
from trip_agents.graph.prompts import CHAT_SYSTEM, FIELD_QUESTIONS
from trip_agents.graph.state import TripState
from trip_agents.llm.context import render_context
from trip_agents.llm.structured import invoke_structured
from trip_agents.memory.ops import apply_memory_defaults
from trip_agents.memory.store import MemoryManager
from trip_agents.schemas import RequestExtraction, SkippedItem, TripRequest

_LABELS = {
    "vegetarian": "素食",
    "halal": "清真",
    "no_pork": "不吃猪肉",
    "no_spicy": "不吃辣",
    "no_seafood": "不吃海鲜",
    "relaxed": "轻松",
    "moderate": "适中",
    "packed": "紧凑",
    "budget": "经济型",
    "midrange": "舒适型",
    "luxury": "高档",
}


def describe_request(req: TripRequest) -> str:
    h, s = req.hard, req.soft
    parts = [f"{h.destination} {h.days} 天（{h.start_date} 出发）", f"{h.travelers} 人", f"总预算 {h.budget_total:.0f} 元"]
    if h.dietary_restrictions:
        parts.append("饮食：" + "、".join(_LABELS[d] for d in h.dietary_restrictions))
    if h.must_visit:
        parts.append("必去：" + "、".join(h.must_visit))
    parts.append(f"节奏：{_LABELS[s.pace]}")
    if s.hotel_tier:
        parts.append(f"住宿：{_LABELS[s.hotel_tier]}")
    return "；".join(parts)


def memory_manager(options: GraphOptions) -> MemoryManager | None:
    """The graph's LangGraph Store, if memory is enabled and the graph was compiled with one."""
    if not options.enable_memory:
        return None
    try:
        store = get_store()
    except RuntimeError:
        return None
    return MemoryManager(store) if store is not None else None


def make_chat_node(deps: AgentDeps, options: GraphOptions):  # type: ignore[no-untyped-def]
    def chat(state: TripState, config: RunnableConfig) -> dict[str, Any]:
        current = state.get("extraction")
        ctx = {"task": "extract_request", "today": deps.today.isoformat(), "current": current.model_dump(mode="json") if current else None}
        convo = [m for m in state.get("messages", []) if isinstance(m, HumanMessage | AIMessage)]
        res = invoke_structured(
            deps.model,
            RequestExtraction,
            [SystemMessage(content=CHAT_SYSTEM + "\n" + render_context(ctx)), *convo],
            max_retries=deps.structured_max_retries,
            stage="chat",
            item="extraction",
            config=config,
        )
        update: dict[str, Any] = {}
        extraction = res.value or current or RequestExtraction()  # explicit statements only
        if not res.ok:
            update["skipped_items"] = [SkippedItem(stage="chat", item="extraction", errors=res.errors)]

        # long-term memory fills only what the user did not say in this conversation
        merged, applied = extraction, dict(state.get("memory_defaults", {}))
        mem = memory_manager(options)
        if mem is not None:
            user_id = state.get("user_id") or "anonymous"
            profile = mem.get_profile(user_id)
            past = mem.find_trip(user_id, extraction.past_trip_hint or extraction.destination) if extraction.refers_to_past_trip else None
            merged, new_applied = apply_memory_defaults(extraction, profile, past)
            applied = new_applied
        ask_defaults = bool(applied) and options.human_in_loop and not state.get("defaults_confirmed")

        missing = merged.missing_required()
        request: TripRequest | None = None
        if not missing:
            try:
                request = merged.to_request()
            except ValidationError as e:  # e.g. inconsistent values; ask again
                missing = sorted({str(err["loc"][-1]) for err in e.errors()})
        if request is None or ask_defaults:
            parts = []
            if ask_defaults:
                parts.append("根据你的历史偏好，我先默认：" + "；".join(applied.values()) + "。直接回复“可以”确认，或告诉我要改什么。")
            if missing:
                parts.append("还需要确认几件事：" + " ".join(FIELD_QUESTIONS.get(f, f) for f in missing))
            return {
                **update,
                "extraction": extraction,
                "missing_fields": missing,
                "memory_defaults": applied,
                "defaults_pending": ask_defaults,
                "defaults_confirmed": state.get("defaults_confirmed", False) or ask_defaults,
                "status": "clarifying",
                "messages": [AIMessage(content="\n".join(parts), name="chat")],
            }
        note = f"（沿用记忆：{'；'.join(applied.values())}）" if applied else ""
        return {
            **update,
            "extraction": extraction,
            "missing_fields": [],
            "memory_defaults": applied,
            "defaults_pending": False,
            "request": request,
            "status": "researching",
            "messages": [AIMessage(content=f"已确认需求：{describe_request(request)}{note}。开始检索信息……", name="chat")],
        }

    return chat


def make_clarify_node(options: GraphOptions):  # type: ignore[no-untyped-def]
    def clarify(state: TripState) -> dict[str, Any]:
        question = next((m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)), "")
        answer = interrupt({"type": "clarify", "question": question, "missing": state.get("missing_fields", []),
                            "memory_defaults": state.get("memory_defaults", {})})  # fmt: skip
        return {"messages": [HumanMessage(content=str(answer))], "clarify_rounds": state.get("clarify_rounds", 0) + 1}

    return clarify


def route_after_chat(options: GraphOptions):  # type: ignore[no-untyped-def]
    def route(state: TripState) -> str:
        if state.get("request") is not None and not state.get("missing_fields") and not state.get("defaults_pending"):
            return "research"
        if state.get("clarify_rounds", 0) >= options.max_clarify_rounds or not options.human_in_loop:
            return END
        return "clarify"

    return route
