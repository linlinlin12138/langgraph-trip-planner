"""How the graph reads and writes long-term memory.

Read  (Chat node, start of planning): profile values and -- if the user refers to a
       past trip -- that trip's settings fill only fields the user did NOT state.
       Precedence: explicit in this conversation > referenced past trip > profile.
       Every filled value is listed back to the user for confirmation.
Write (Confirm node, after the user accepts the plan): an LLM extracts preferences
       with verbatim evidence; `merge_profile` drops anything the user didn't
       literally say. The trip summary is built deterministically from the final state.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from trip_agents.llm.context import render_context
from trip_agents.llm.structured import invoke_structured
from trip_agents.memory.profile import FIELD_NAMES, ProfileExtraction, TripSummary, UserProfile, merge_profile, pretty
from trip_agents.memory.store import MemoryManager
from trip_agents.schemas import RequestExtraction

PROFILE_SYSTEM = """你负责从用户在本次对话中的发言里提取「长期偏好」，用于以后的旅行规划。
可提取的字段（固定）：
- dietary_restrictions: 饮食禁忌列表（vegetarian/halal/no_pork/no_spicy/no_seafood）
- hotel_tier: 住宿档次（budget/midrange/luxury）
- pace: 节奏（relaxed/moderate/packed）
- budget_per_person_per_day: 人均每天预算（元），由用户说的预算、人数、天数换算
- companions: 同行人，如“女朋友”“家人（3人）”“独自”

规则：
1. 只提取用户**明确说过**的内容；不要根据行程内容、助手的话或常识推断。
2. evidence 必须是用户原话中的一段，逐字复制。
3. 没有把握就不要输出该字段。"""

_COUPLE = {"女朋友", "男朋友", "老婆", "老公", "妻子", "丈夫", "对象", "伴侣", "朋友", "同事"}


def companions_to_travelers(companions: str | None) -> int | None:
    if not companions:
        return None
    if m := re.search(r"(\d+)\s*人", companions):
        return int(m[1])
    if companions in ("独自", "自己", "一个人"):
        return 1
    if companions in _COUPLE:
        return 2
    return None


def apply_memory_defaults(
    ex: RequestExtraction, profile: UserProfile | None, past: TripSummary | None
) -> tuple[RequestExtraction, dict[str, str]]:
    """Fill unstated fields. Returns (merged extraction, {field: human-readable note})."""
    merged = ex.model_copy(deep=True)
    applied: dict[str, str] = {}

    def fill(field: str, value: Any, source: str) -> None:
        if value in (None, [], "") or field in applied:
            return
        current = getattr(merged, field)
        if current in (None, []):
            if field == "dietary_restrictions" and ex.dietary_none:
                return
            setattr(merged, field, value)
            applied[field] = f"{FIELD_NAMES[field]}：{pretty(field, value)}（来自{source}）"

    if past is not None:
        src = f"上次的{past.destination}行程"
        fill("destination", past.destination, src)
        fill("days", past.days, src)
        fill("travelers", past.travelers, src)
        fill("pace", past.pace, src)
        fill("hotel_tier", past.hotel_tier, src)
        fill("interests", past.interests, src)
        fill("dietary_restrictions", past.dietary_restrictions, src)
        if merged.budget_total is None and merged.days:
            per = past.budget_per_person_per_day
            fill("budget_total", round(per * (merged.travelers or 1) * merged.days, -2), f"{src}（人均每天约 {per:.0f} 元）")
    if profile is not None:
        fill("dietary_restrictions", profile.get("dietary_restrictions"), "用户画像")
        fill("hotel_tier", profile.get("hotel_tier"), "用户画像")
        fill("pace", profile.get("pace"), "用户画像")
        fill("travelers", companions_to_travelers(profile.get("companions")), f"用户画像（同行：{profile.get('companions')}）")
        per = profile.get("budget_per_person_per_day")
        if per and merged.budget_total is None and merged.days:
            fill("budget_total", round(per * (merged.travelers or 1) * merged.days, -2), f"用户画像（人均每天约 {per:.0f} 元）")
    return merged, applied


def remember_trip(
    mem: MemoryManager,
    model: BaseChatModel,
    state: dict[str, Any],
    *,
    thread_id: str,
    now: datetime,
    config: RunnableConfig | None = None,
    max_retries: int = 3,
) -> list[str]:
    """Update profile + store trip summary. Returns human-readable change notes."""
    user_id = state.get("user_id") or "anonymous"
    texts = [m.content for m in state.get("messages", []) if isinstance(m, HumanMessage) and isinstance(m.content, str)]
    res = invoke_structured(
        model, ProfileExtraction,
        [SystemMessage(content=PROFILE_SYSTEM), HumanMessage(content=render_context({"task": "extract_profile", "user_messages": texts}))],
        max_retries=max_retries, stage="memory", item="profile", config=config,
    )  # fmt: skip
    profile = mem.get_profile(user_id) or UserProfile(user_id=user_id)
    notes: list[str] = []
    if res.value is not None:
        profile, changes, _rejected = merge_profile(profile, res.value.preferences, user_texts=texts, thread_id=thread_id, now=now)
        mem.put_profile(profile)
        notes += [f"更新画像 {c}" for c in changes]

    req, final, research = state["request"], state["final"], state["research"]
    it = final.itinerary
    hotel = research.hotel(it.hotel_id) if it.hotel_id else None
    highlights = [a.title for d in it.days for a in d.activities if a.kind == "attraction"]
    h, soft = req.hard, req.soft
    trip = TripSummary(
        thread_id=thread_id,
        created_at=now,
        destination=h.destination,
        start_date=h.start_date,
        days=h.days,
        travelers=h.travelers,
        budget_total=h.budget_total,
        pace=soft.pace,
        hotel_tier=soft.hotel_tier,
        hotel_name=hotel.name if hotel else None,
        interests=list(soft.interests),
        dietary_restrictions=list(h.dietary_restrictions),
        highlights=highlights,
        total_cost=final.budget.total,
    )
    evicted = mem.add_trip(user_id, trip)
    notes.append(f"保存行程摘要：{trip.destination} {trip.days} 天" + (f"（淘汰最旧 {len(evicted)} 条）" if evicted else ""))
    return notes
