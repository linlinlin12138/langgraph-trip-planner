"""LLM-as-Judge: three 0-100 dimensions, scored by the model, weighted by code.

* The model only produces the three sub-scores (+ rationale). The weighted total
  is computed in code so the weighting is explicit, versioned and calibratable
  (see ``calibrate.py``) instead of being an opaque number the model invents.
* Anchored rubric with worked examples reduces score drift between runs/models.
* Hard constraints are NOT judged here -- they are checked by rules. The judge is
  given the rule violations as context so it doesn't re-litigate them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from trip_agents.evals.world import normalize_refs, world_bundle
from trip_agents.llm.context import render_context
from trip_agents.llm.structured import invoke_structured
from trip_agents.schemas import Itinerary, TripRequest

DEFAULT_WEIGHTS = {"richness": 0.3, "preference_fit": 0.4, "reasonableness": 0.3}


class JudgeScores(BaseModel):
    richness: int = Field(ge=0, le=100, description="行程丰富度")
    preference_fit: int = Field(ge=0, le=100, description="偏好符合度")
    reasonableness: int = Field(ge=0, le=100, description="行程合理性")
    rationale: str = Field(description="每个维度一句话理由")


JUDGE_SYSTEM = """你是严格的旅行行程评审。根据用户需求（request）评价行程（plan），给出三个 0-100 的整数分：

1. richness 行程丰富度：每天内容是否充实、类型是否多样（自然/历史/美食/文化等）、有无亮点。
   - 90+：每天 2-4 个高质量且类型互补的安排，餐饮有地方特色
   - 60-75：内容基本够，但类型单一或有明显空档
   - <40：大量空白或几乎只有吃饭
2. preference_fit 偏好符合度：兴趣、节奏、住宿档次、菜系喜好是否被体现（硬约束已由规则检查，不在此评）。
   - 90+：几乎所有安排都对应用户兴趣，节奏与要求一致
   - 60-75：一半左右对应兴趣，或节奏略有偏差
   - <40：基本无视偏好
3. reasonableness 行程合理性：顺序和动线是否顺路、时长是否合理、是否考虑天气、是否留出交通时间。
   context.violations 是规则检查发现的问题，可作为参考。
   - 90+：动线顺、时长合适、雨天有室内备选
   - 60-75：可执行但有绕路或过赶
   - <40：明显不可执行

示例（仅用于校准尺度）：
- 用户“喜欢历史，节奏轻松”，计划每天 2 个历史景点 + 两顿当地菜、景点同区域 → richness 78, preference_fit 92, reasonableness 88
- 用户“喜欢美食和夜景，紧凑”，计划每天 1 个博物馆、无夜间安排 → richness 55, preference_fit 30, reasonableness 75

只依据给定内容打分，不要因为篇幅长就给高分。"""


@dataclass
class JudgeResult:
    scores: JudgeScores | None
    total: float | None
    attempts: int
    error: str | None = None


def render_plan(it: Itinerary, gold: TripRequest) -> dict[str, Any]:
    """Resolve ids to names/categories so the judge reads what a human would read."""
    world = world_bundle(gold.hard.destination)
    norm, _, _ = normalize_refs(it, world)
    out_days = []
    for d in norm.days:
        acts = []
        for a in d.activities:
            item: dict[str, Any] = {
                "time": f"{a.start_time:%H:%M}-{a.end_time:%H:%M}",
                "kind": a.kind,
                "title": a.title,
                "cost": a.est_cost,
            }
            if a.kind == "attraction" and (x := world.attraction(a.ref_id or "")):
                item |= {"category": x.category, "area": x.area}
            if a.kind == "meal" and (r := world.restaurant(a.ref_id or "")):
                item |= {"cuisine": r.cuisine, "area": r.area}
            acts.append(item)
        out_days.append({"date": d.date.isoformat(), "activities": acts})
    hotel = world.hotel(norm.hotel_id or "")
    return {"hotel": hotel.model_dump(include={"name", "tier", "area", "price_per_night"}) if hotel else None, "days": out_days}


def weighted_total(s: JudgeScores, weights: dict[str, float] | None = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    norm = sum(w.values())
    return round(sum(float(getattr(s, k)) * v for k, v in w.items()) / norm, 2)


def judge_plan(
    model: BaseChatModel, it: Itinerary, gold: TripRequest, violations: list[str], weights: dict[str, float] | None = None
) -> JudgeResult:
    ctx = render_context(
        {"task": "judge", "request": gold.model_dump(mode="json"), "plan": render_plan(it, gold), "violations": violations}
    )
    res = invoke_structured(
        model, JudgeScores, [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=ctx)], stage="judge", item="plan"
    )
    if res.value is None:
        return JudgeResult(None, None, res.attempts, res.errors[-1] if res.errors else "unknown")
    return JudgeResult(res.value, weighted_total(res.value, weights), res.attempts)
