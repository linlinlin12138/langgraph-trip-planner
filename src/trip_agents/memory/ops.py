"""How the graph reads and writes long-term memory.

Read  (Chat node, start of planning): profile values and -- if the user refers to a
       past trip -- that trip's settings fill only fields the user did NOT state.
       Precedence: explicit in this conversation > referenced past trip > profile.
       Exception: dietary restrictions are a safety constraint, so a restriction stated now is
       *added* to the remembered ones (union) unless the user says there are none this time.
       Party size comes from the profile's stored number, else from parsing ``companions``;
       if the profile names companions but neither gives a number, the user is asked
       instead of silently planning for one person.
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
from trip_agents.memory.profile import (
    FIELD_NAMES,
    HISTORY_LIMIT,
    ProfileExtraction,
    TimedValue,
    TripSummary,
    UserProfile,
    merge_profile,
    pretty,
)
from trip_agents.memory.store import MemoryManager
from trip_agents.schemas import RequestExtraction

PROFILE_SYSTEM = """你负责从用户在本次对话中的发言里提取「长期偏好」，用于以后的旅行规划。
可提取的字段（固定）：
- dietary_restrictions: 饮食禁忌列表（no_spicy/no_seafood）
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


def profile_travelers(profile: UserProfile | None) -> int | None:
    """Party size the profile implies: the stored number, else parsed from ``companions``."""
    if profile is None:
        return None
    n = profile.get("travelers")
    return int(n) if n else companions_to_travelers(profile.get("companions"))


def travelers_unresolved(merged: RequestExtraction, profile: UserProfile | None) -> bool:
    """The profile says the user travels with someone, but no party size could be derived -> ask."""
    return merged.travelers is None and profile is not None and bool(profile.get("companions")) and profile_travelers(profile) is None


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
        if field == "dietary_restrictions" and current and not ex.dietary_none:
            extra = sorted(set(value) - set(current))
            if extra:  # union: a new restriction never drops a remembered one
                setattr(merged, field, sorted({*current, *extra}))
                applied[field] = f"{FIELD_NAMES[field]}：另加 {pretty(field, extra)}（来自{source}）"
            return
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
        who = f"同行：{profile.get('companions')}" if profile.get("companions") else "上次确认的人数"
        fill("travelers", profile_travelers(profile), f"用户画像（{who}）")
        per = profile.get("budget_per_person_per_day")
        if per and merged.budget_total is None and merged.days:
            fill("budget_total", round(per * (merged.travelers or 1) * merged.days, -2), f"用户画像（人均每天约 {per:.0f} 元）")
    return merged, applied


def remember_travelers(profile: UserProfile, state: dict[str, Any], *, thread_id: str, now: datetime) -> tuple[UserProfile, str | None]:
    """Store the confirmed party size, but only if the user stated it in this conversation (not a default)."""
    ex: RequestExtraction | None = state.get("extraction")
    if ex is None or ex.travelers is None:
        return profile, None
    n = state["request"].hard.travelers
    old: TimedValue | None = profile.travelers
    if old is not None and old.value == n:
        old.updated_at, old.source_thread = now, thread_id
        return profile, None
    history = ([*old.history, {"value": old.value, "updated_at": old.updated_at.isoformat()}] if old else [])[-HISTORY_LIMIT:]
    tv = TimedValue(value=n, updated_at=now, source_thread=thread_id, evidence=f"本次确认：{n} 人", history=history)
    out = profile.model_copy(update={"travelers": tv})
    return out, f"更新画像 人数：{old.value} → {n}" if old else f"更新画像 人数：{n}"


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
        model,
        ProfileExtraction,
        [SystemMessage(content=PROFILE_SYSTEM), HumanMessage(content=render_context({"task": "extract_profile", "user_messages": texts}))],
        max_retries=max_retries,
        stage="memory",
        item="profile",
        config=config,
    )
    profile = mem.get_profile(user_id) or UserProfile(user_id=user_id)
    notes: list[str] = []
    if res.value is not None:
        profile, changes, _rejected = merge_profile(profile, res.value.preferences, user_texts=texts, thread_id=thread_id, now=now)
        notes += [f"更新画像 {c}" for c in changes]
    profile, note = remember_travelers(profile, state, thread_id=thread_id, now=now)
    notes += [note] if note else []
    mem.put_profile(profile)

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
