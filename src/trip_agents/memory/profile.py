"""Long-term memory schemas and the pure merge logic.

Two kinds of memory, on purpose:
* UserProfile  -- a *fixed* set of fields, each a single current value with time,
  evidence and a short history. Cheap to load, easy to show/edit, conflicts are
  well-defined (newest wins).
* TripSummary  -- one compact record per confirmed trip, capped in number,
  retrieved when the user says "像上次那样".
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from trip_agents.schemas.request import DietaryRestriction, HotelTier, Pace

ProfileField = Literal["dietary_restrictions", "hotel_tier", "pace", "budget_per_person_per_day", "companions"]
PROFILE_FIELDS: tuple[str, ...] = get_args(ProfileField)
FIELD_LABELS = {
    "dietary_restrictions": "饮食禁忌",
    "hotel_tier": "住宿档次",
    "pace": "节奏",
    "budget_per_person_per_day": "预算习惯（人均每天）",
    "companions": "同行人",
}
_VALIDATORS: dict[str, TypeAdapter[Any]] = {
    "dietary_restrictions": TypeAdapter(list[DietaryRestriction]),
    "hotel_tier": TypeAdapter(HotelTier),
    "pace": TypeAdapter(Pace),
    "budget_per_person_per_day": TypeAdapter(float),
    "companions": TypeAdapter(str),
}
HISTORY_LIMIT = 5
VALUE_LABELS = {
    "no_spicy": "不吃辣",
    "no_seafood": "不吃海鲜",
    "relaxed": "轻松",
    "moderate": "适中",
    "packed": "紧凑",
    "budget": "经济型",
    "midrange": "舒适型",
    "luxury": "高档",
    "history": "历史",
    "nature": "自然",
    "museum": "博物馆",
    "food": "美食",
    "shopping": "购物",
    "nightlife": "夜景",
    "culture": "文化",
}
FIELD_NAMES = {
    "destination": "目的地",
    "days": "天数",
    "travelers": "人数",
    "pace": "节奏",
    "hotel_tier": "住宿",
    "interests": "兴趣",
    "dietary_restrictions": "饮食",
    "budget_total": "总预算",
}


def pretty(field: str, value: Any) -> str:
    if isinstance(value, list):
        return "、".join(VALUE_LABELS.get(v, str(v)) for v in value)
    if field == "budget_total":
        return f"{value:.0f} 元"
    return VALUE_LABELS.get(value, str(value)) if isinstance(value, str) else str(value)


class TimedValue(BaseModel):
    value: Any
    updated_at: datetime
    source_thread: str
    evidence: str = Field(description="The user's own words this value was taken from")
    history: list[dict[str, Any]] = Field(default_factory=list, description="Previous values, newest last")


class UserProfile(BaseModel):
    user_id: str
    dietary_restrictions: TimedValue | None = None
    hotel_tier: TimedValue | None = None
    pace: TimedValue | None = None
    budget_per_person_per_day: TimedValue | None = None
    companions: TimedValue | None = None

    def get(self, name: str) -> Any:
        tv: TimedValue | None = getattr(self, name)
        return None if tv is None else tv.value

    def describe(self) -> list[str]:
        out = []
        for f in PROFILE_FIELDS:
            tv: TimedValue | None = getattr(self, f)
            if tv is not None:
                value = f"{tv.value:.0f} 元" if f == "budget_per_person_per_day" else pretty(f, tv.value)
                out.append(f"{FIELD_LABELS[f]}：{value or '无'}（{tv.updated_at:%Y-%m-%d} 更新）")
        return out


# ----------------------------------------------------------------------------- LLM extraction output
class ExtractedPreference(BaseModel):
    field: ProfileField
    # explicit union (not Any) so every provider's structured-output mode accepts the schema
    value: list[str] | float | str
    evidence: str = Field(description="用户原话中的片段，必须逐字出现在用户消息里")


class ProfileExtraction(BaseModel):
    preferences: list[ExtractedPreference] = Field(default_factory=list)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def merge_profile(
    profile: UserProfile,
    extracted: list[ExtractedPreference],
    *,
    user_texts: list[str],
    thread_id: str,
    now: datetime,
) -> tuple[UserProfile, list[str], list[str]]:
    """Apply extracted preferences. Returns (new_profile, changes, rejected).

    * evidence must literally appear in what the user wrote -> only explicit preferences are stored
    * value must validate against the field type
    * same value -> refresh timestamp; different value -> newest wins, old value moved to history
    """
    out = profile.model_copy(deep=True)
    corpus = _norm("\n".join(user_texts))
    changes, rejected = [], []
    for p in extracted:
        label = FIELD_LABELS[p.field]
        if not p.evidence.strip() or _norm(p.evidence) not in corpus:
            rejected.append(f"{label}：依据「{p.evidence}」不是用户原话")
            continue
        try:
            value = _VALIDATORS[p.field].validate_python(p.value)
        except ValidationError:
            rejected.append(f"{label}：取值 {p.value!r} 不合法")
            continue
        if p.field == "dietary_restrictions":
            value = sorted(set(value))
        if p.field == "budget_per_person_per_day":
            value = round(float(value), -1)
        old: TimedValue | None = getattr(out, p.field)
        if old is not None and old.value == value:
            old.updated_at, old.evidence, old.source_thread = now, p.evidence, thread_id
            continue
        history = ([*old.history, {"value": old.value, "updated_at": old.updated_at.isoformat()}] if old else [])[-HISTORY_LIMIT:]
        setattr(out, p.field, TimedValue(value=value, updated_at=now, source_thread=thread_id, evidence=p.evidence, history=history))
        changes.append(f"{label}：{old.value} → {value}" if old else f"{label}：{value}")
    return out, changes, rejected


# ----------------------------------------------------------------------------- trip summaries
class TripSummary(BaseModel):
    thread_id: str
    created_at: datetime
    destination: str
    start_date: date
    days: int
    travelers: int
    budget_total: float
    pace: Pace
    hotel_tier: HotelTier | None = None
    hotel_name: str | None = None
    interests: list[str] = Field(default_factory=list)
    dietary_restrictions: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list, description="Attraction names actually in the plan")
    total_cost: float = 0.0

    @property
    def budget_per_person_per_day(self) -> float:
        return self.budget_total / max(1, self.travelers * self.days)

    def describe(self) -> str:
        return (
            f"{self.start_date:%Y-%m} {self.destination} {self.days} 天，{self.travelers} 人，预算 {self.budget_total:.0f} 元，"
            f"节奏 {self.pace}，亮点：{'、'.join(self.highlights[:4])}"
        )
