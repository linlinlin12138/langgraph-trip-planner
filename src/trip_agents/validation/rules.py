"""Rule-based validation of hard constraints. Pure functions, no LLM.

Each rule returns a list of `ValidationIssue`. `error` issues trigger a re-plan
(if retries remain); `warning` issues are reported but never block.
Issues carry `day_index` when local to one day so the Itinerary agent can
repair only the affected days.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import time
from itertools import pairwise

from trip_agents.schemas import BudgetReport, Itinerary, ResearchBundle, TripRequest, ValidationIssue

EARLIEST = time(7, 0)
LATEST = time(23, 30)
MAX_ATTRACTIONS_RELAXED = 2

Rule = Callable[[Itinerary, TripRequest, ResearchBundle, BudgetReport | None], list[ValidationIssue]]


def check_days(it: Itinerary, req: TripRequest, rb: ResearchBundle, br: BudgetReport | None) -> list[ValidationIssue]:
    issues = []
    expected = req.hard.dates
    got = {d.day_index: d for d in it.days}
    for i, dt in enumerate(expected):
        d = got.get(i)
        if d is None:
            issues.append(ValidationIssue(code="missing_day", message=f"缺少第 {i + 1} 天（{dt}）的行程", day_index=i))
        elif d.date != dt:
            issues.append(ValidationIssue(code="wrong_date", message=f"第 {i + 1} 天日期应为 {dt}，实际为 {d.date}", day_index=i))
    for idx in got:
        if idx >= len(expected):
            issues.append(
                ValidationIssue(code="extra_day", message=f"行程多出了第 {idx + 1} 天，总天数应为 {len(expected)}", day_index=idx)
            )
    return issues


def check_time_conflicts(it: Itinerary, req: TripRequest, rb: ResearchBundle, br: BudgetReport | None) -> list[ValidationIssue]:
    issues = []
    for d in it.days:
        acts = d.activities
        for a in acts:
            if a.start_time < EARLIEST or a.end_time > LATEST:
                issues.append(
                    ValidationIssue(
                        code="out_of_hours",
                        day_index=d.day_index,
                        message=f"第 {d.day_index + 1} 天「{a.title}」{a.start_time}-{a.end_time} 超出 07:00-23:30",
                    )
                )
        for prev, nxt in pairwise(acts):
            if nxt.start_time < prev.end_time:
                issues.append(
                    ValidationIssue(
                        code="time_overlap",
                        day_index=d.day_index,
                        message=f"第 {d.day_index + 1} 天「{prev.title}」({prev.start_time}-{prev.end_time}) 与"
                        f"「{nxt.title}」({nxt.start_time}-{nxt.end_time}) 时间重叠或顺序错误",
                    )
                )
    return issues


def check_references(it: Itinerary, req: TripRequest, rb: ResearchBundle, br: BudgetReport | None) -> list[ValidationIssue]:
    issues = []
    if it.hotel_id and rb.hotel(it.hotel_id) is None:
        issues.append(ValidationIssue(code="unknown_hotel", message=f"酒店 {it.hotel_id} 不在检索结果中"))
    if req.hard.days > 1 and not it.hotel_id:
        issues.append(ValidationIssue(code="missing_hotel", message="多日行程未选择酒店"))
    for d in it.days:
        for a in d.activities:
            if a.kind == "attraction":
                attr = rb.attraction(a.ref_id or "")
                if attr is None:
                    issues.append(
                        ValidationIssue(code="unknown_ref", day_index=d.day_index, message=f"景点「{a.title}」({a.ref_id}) 不在检索结果中")
                    )
                elif a.start_time < attr.open_time or a.end_time > attr.close_time:
                    issues.append(
                        ValidationIssue(
                            code="closed",
                            day_index=d.day_index,
                            message=f"第 {d.day_index + 1} 天「{attr.name}」安排在 {a.start_time}-{a.end_time}，"
                            f"但开放时间为 {attr.open_time}-{attr.close_time}",
                        )
                    )
            elif a.kind == "meal" and rb.restaurant(a.ref_id or "") is None:
                issues.append(
                    ValidationIssue(code="unknown_ref", day_index=d.day_index, message=f"餐厅「{a.title}」({a.ref_id}) 不在检索结果中")
                )
    return issues


def check_dietary(it: Itinerary, req: TripRequest, rb: ResearchBundle, br: BudgetReport | None) -> list[ValidationIssue]:
    need = set(req.hard.dietary_restrictions)
    if not need:
        return []
    issues = []
    for d in it.days:
        for a in d.activities:
            r = rb.restaurant(a.ref_id or "") if a.kind == "meal" else None
            if r and not need <= set(r.dietary_tags):
                issues.append(
                    ValidationIssue(
                        code="dietary",
                        day_index=d.day_index,
                        message=f"第 {d.day_index + 1} 天餐厅「{r.name}」不满足饮食限制 {sorted(need - set(r.dietary_tags))}",
                    )
                )
    return issues


def check_must_visit(it: Itinerary, req: TripRequest, rb: ResearchBundle, br: BudgetReport | None) -> list[ValidationIssue]:
    found = (rb.attraction(a.ref_id or "") for d in it.days for a in d.activities if a.kind == "attraction")
    visited = {x.name for x in found if x is not None}
    return [ValidationIssue(code="must_visit", message=f"必去景点「{m}」未安排") for m in req.hard.must_visit if m not in visited]


def check_budget(it: Itinerary, req: TripRequest, rb: ResearchBundle, br: BudgetReport | None) -> list[ValidationIssue]:
    if br is None:
        return []
    if br.total > br.budget:
        return [
            ValidationIssue(code="over_budget", message=f"总花费 {br.total:.0f} 元超出预算 {br.budget:.0f} 元（超 {br.over_by:.0f} 元）")
        ]
    if br.total > 0.95 * br.budget:
        return [ValidationIssue(code="near_budget", severity="warning", message=f"总花费 {br.total:.0f} 元已接近预算上限")]
    return []


def check_soft(it: Itinerary, req: TripRequest, rb: ResearchBundle, br: BudgetReport | None) -> list[ValidationIssue]:
    issues = []
    counts = Counter(a.ref_id for d in it.days for a in d.activities if a.kind == "attraction")
    for ref, c in counts.items():
        if c > 1:
            issues.append(ValidationIssue(code="duplicate_attraction", severity="warning", message=f"景点 {ref} 被安排了 {c} 次"))
    if req.soft.pace == "relaxed":
        for d in it.days:
            n = sum(a.kind == "attraction" for a in d.activities)
            if n > MAX_ATTRACTIONS_RELAXED:
                issues.append(
                    ValidationIssue(
                        code="pace",
                        severity="warning",
                        day_index=d.day_index,
                        message=f"第 {d.day_index + 1} 天安排了 {n} 个景点，与“轻松”节奏不符",
                    )
                )
    return issues


ALL_RULES: list[Rule] = [check_days, check_time_conflicts, check_references, check_dietary, check_must_visit, check_budget, check_soft]


def validate_itinerary(it: Itinerary, req: TripRequest, rb: ResearchBundle, br: BudgetReport | None = None) -> list[ValidationIssue]:
    return [issue for rule in ALL_RULES for issue in rule(it, req, rb, br)]


def has_errors(issues: list[ValidationIssue]) -> bool:
    return any(i.severity == "error" for i in issues)
