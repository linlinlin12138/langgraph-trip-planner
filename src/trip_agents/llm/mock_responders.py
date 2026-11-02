"""Deterministic 'brains' behind MockChatModel.

These are deliberately simple rule-based stand-ins so the whole graph (including
retries, re-planning and human-in-the-loop) runs offline. They are NOT meant to be
good planners -- quality numbers only mean something with a real model.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, time, timedelta
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from trip_agents.tools.mock_backend import _load_all

# =========================================================================== chat
_CN_DIGITS = {"零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_NUM = r"(\d+|[一二两三四五六七八九十]+)"


def cn_to_int(s: str) -> int:
    if s.isdigit():
        return int(s)
    if "十" in s:
        left, _, right = s.partition("十")
        return (_CN_DIGITS.get(left, 1) if left else 1) * 10 + (_CN_DIGITS.get(right, 0) if right else 0)
    return _CN_DIGITS.get(s, 0)


_DIET_RULES = [
    (r"素食|吃素|素食者", "vegetarian"),
    (r"清真", "halal"),
    (r"不吃猪肉|不能吃猪肉", "no_pork"),
    (r"不吃辣|不能吃辣|不要辣|怕辣|吃不了辣", "no_spicy"),
    (r"海鲜过敏|不吃海鲜|不能吃海鲜", "no_seafood"),
]
_PACE_RULES = [(r"轻松|悠闲|慢节奏|别太累|不想太累|休闲", "relaxed"), (r"紧凑|特种兵|多去几个|充实|暴走", "packed")]
_TIER_RULES = [(r"经济|便宜|穷游|青旅|省钱", "budget"), (r"舒适型|中档|中等", "midrange"), (r"豪华|五星|高端|奢华", "luxury")]
_INTEREST_RULES = [
    (r"历史|古迹|古建", "history"),
    (r"自然|爬山|风景|山水|公园", "nature"),
    (r"博物馆|展览", "museum"),
    (r"美食|小吃|好吃的|吃货", "food"),
    (r"购物|逛街|买买买", "shopping"),
    (r"夜景|夜生活|酒吧", "nightlife"),
    (r"文化|艺术|寺庙|演出", "culture"),
]
_COMPANION_2 = r"我和(?:女朋友|男朋友|老婆|老公|妻子|丈夫|朋友|对象|爸爸|妈妈|同事)|情侣|两口子|二人"


def _parse_date(text: str, today: date) -> date | None:
    if m := re.search(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", text):
        return date(int(m[1]), int(m[2]), int(m[3]))
    if m := re.search(r"(\d{1,2})月(\d{1,2})[日号]", text):
        d = date(today.year, int(m[1]), int(m[2]))
        return d if d >= today else date(today.year + 1, d.month, d.day)
    if "后天" in text:
        return today + timedelta(days=2)
    if "明天" in text:
        return today + timedelta(days=1)
    return None


def _attraction_aliases() -> dict[str, str]:
    """alias -> canonical name, e.g. '兵马俑' -> '秦始皇帝陵博物院(兵马俑)'."""
    out: dict[str, str] = {}
    for city in _load_all().values():
        for a in city["attractions"]:
            out[a["name"]] = a["name"]
            if m := re.search(r"\((.+?)\)", a["name"]):
                out[m[1]] = a["name"]
    return out


def extract_request(texts: list[str], today: date, current: dict[str, Any] | None) -> dict[str, Any]:
    r: dict[str, Any] = {k: v for k, v in (current or {}).items() if v not in (None, [], "")}
    cities = list(_load_all())
    aliases = _attraction_aliases()
    per_person_budget: float | None = None
    for text in texts:
        if found := [c for c in cities if c in text]:
            r["destination"] = found[0]
        if d := _parse_date(text, today):
            r["start_date"] = d.isoformat()
        if m := re.search(_NUM + r"\s*(?:天|日游)", text):
            r["days"] = cn_to_int(m[1])
        elif "周末" in text and "days" not in r:
            r["days"] = 2
        if m := re.search(r"预算[^\d]{0,6}(\d+(?:\.\d+)?)\s*(万|w|W|千|k|K)?", text) or re.search(
            r"(\d+(?:\.\d+)?)\s*(万|千)?\s*(?:元|块)", text
        ):
            amount = float(m[1]) * {"万": 10000, "w": 10000, "W": 10000, "千": 1000, "k": 1000, "K": 1000}.get(m[2] or "", 1)
            if re.search(r"人均|每人|一个人", text[max(0, m.start() - 6) : m.end()]):
                per_person_budget = amount
            else:
                r["budget_total"] = amount
        if (m := re.search(r"一家(" + "|".join(_CN_DIGITS) + r")口", text)) or (m := re.search(_NUM + r"\s*(?:个)?(?:人|位)(?!均)", text)):
            r["travelers"] = cn_to_int(m[1])
        elif re.search(_COMPANION_2, text):
            r["travelers"] = 2
        elif re.search(r"独自|一个人去|自己去", text):
            r["travelers"] = 1
        for pat, tag in _DIET_RULES:
            if re.search(pat, text):
                r.setdefault("dietary_restrictions", [])
                if tag not in r["dietary_restrictions"]:
                    r["dietary_restrictions"].append(tag)
        for pat, pace in _PACE_RULES:
            if re.search(pat, text):
                r["pace"] = pace
        for pat, tier in _TIER_RULES:
            if re.search(pat, text):
                r["hotel_tier"] = tier
        bare = text
        for alias in sorted(aliases, key=len, reverse=True):  # "天坛公园" must not count as interest "公园"
            bare = bare.replace(alias, "")
        for pat, interest in _INTEREST_RULES:
            if re.search(pat, bare):
                r.setdefault("interests", [])
                if interest not in r["interests"]:
                    r["interests"].append(interest)
        for alias, name in sorted(aliases.items(), key=lambda kv: -len(kv[0])):
            if alias in text and name not in r.get("must_visit", []) and re.search(r"一定|必须|想去|要去|必去", text):
                r.setdefault("must_visit", []).append(name)
    if per_person_budget is not None:
        r["budget_total"] = per_person_budget * int(r.get("travelers", 1))
    return r


def respond_extract(messages: list[BaseMessage], ctx: dict[str, Any]) -> dict[str, Any]:
    texts = [m.content for m in messages if isinstance(m, HumanMessage) and isinstance(m.content, str) and "<context>" not in m.content]
    today = date.fromisoformat(ctx.get("today", date.today().isoformat()))
    return extract_request(texts, today, ctx.get("current"))


# =========================================================================== research
def _tool_history(messages: list[BaseMessage]) -> list[tuple[str, dict[str, Any], bool, Any]]:
    """[(tool_name, args, ok, data)] in call order."""
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    out: list[tuple[str, dict[str, Any], bool, Any]] = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                calls[tc["id"] or ""] = (tc["name"], tc["args"])
        elif isinstance(m, ToolMessage) and m.tool_call_id in calls:
            name, args = calls[m.tool_call_id]
            text = m.content if isinstance(m.content, str) else str(m.content)
            ok = not text.startswith("ERROR")
            out.append((name, args, ok, json.loads(text) if ok else text))
    return out


def respond_research(messages: list[BaseMessage], ctx: dict[str, Any], faults: set[str]) -> AIMessage:
    req = ctx["request"]
    hard, soft = req["hard"], req["soft"]
    city, days = hard["destination"], hard["days"]
    start = date.fromisoformat(hard["start_date"])
    end = start + timedelta(days=days - 1)
    hist = _tool_history(messages)
    last: dict[str, tuple[dict[str, Any], bool, Any]] = {}
    for name, args, ok, data in hist:
        last[name] = (args, ok, data)

    nights = max(days - 1, 1)
    rooms = (hard["travelers"] + 1) // 2
    hotel_cap = round(hard["budget_total"] * 0.45 / nights / rooms, -1)

    calls: list[dict[str, Any]] = []
    if not hist:
        bad_dates = "bad_tool_args_once" in faults
        calls = [
            {
                "name": "get_weather",
                "args": {"city": city, "start_date": start.isoformat(), "end_date": "下周五" if bad_dates else end.isoformat()},
            },
            {"name": "search_hotels", "args": {"city": city, "tier": soft.get("hotel_tier"), "max_price_per_night": hotel_cap}},
            {"name": "search_restaurants", "args": {"city": city, "dietary": hard["dietary_restrictions"], "limit": 10}},
            {"name": "search_attractions", "args": {"city": city, "categories": soft.get("interests", []), "limit": 12}},
        ]
        if bad_dates:
            faults.discard("bad_tool_args_once")
    else:
        if "get_weather" in last and not last["get_weather"][1]:
            calls.append({"name": "get_weather", "args": {"city": city, "start_date": start.isoformat(), "end_date": end.isoformat()}})
        h_args, h_ok, h_data = last["search_hotels"]
        if h_ok and not h_data:  # nothing matched: relax progressively
            relaxed = {"city": city, "tier": h_args.get("tier"), "max_price_per_night": None}
            if h_args.get("max_price_per_night") is None:
                relaxed["tier"] = None
            if {k: v for k, v in relaxed.items() if v is not None} != h_args:
                calls.append({"name": "search_hotels", "args": relaxed})
        a_args, a_ok, a_data = last["search_attractions"]
        found = {a["name"] for a in (a_data or [])} if a_ok else set()
        if (set(hard["must_visit"]) - found or (a_ok and len(a_data) < days * 2)) and a_args.get("categories"):
            calls.append({"name": "search_attractions", "args": {"city": city, "categories": [], "limit": 15}})
        if not calls and "get_route" not in last:
            hotels = h_data if h_ok and h_data else []
            attrs = a_data if a_ok and a_data else []
            if hotels and attrs:
                calls.append({"name": "get_route", "args": {"city": city, "origin": hotels[0]["id"], "destination": attrs[0]["id"]}})
    if calls:
        n = len(hist)
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": c["name"],
                    "args": {k: v for k, v in c["args"].items() if v is not None},
                    "id": f"call_{n + i}",
                    "type": "tool_call",
                }
                for i, c in enumerate(calls)
            ],
        )
    return AIMessage(content="检索完成：已获得天气、酒店、餐厅、景点和交通信息。")


# =========================================================================== itinerary
def _t(s: str) -> time:
    return time.fromisoformat(s)


def _add(t: time, minutes: float) -> time:
    dt = datetime.combine(date(2000, 1, 1), t) + timedelta(minutes=minutes)
    return dt.time() if dt.date() == date(2000, 1, 1) else time(23, 59)


def _mins(a: time, b: time) -> float:
    return (datetime.combine(date(2000, 1, 1), b) - datetime.combine(date(2000, 1, 1), a)).total_seconds() / 60


_PER_DAY = {"relaxed": 2, "moderate": 3, "packed": 4}


def respond_plan_day(ctx: dict[str, Any], faults: set[str]) -> dict[str, Any]:
    if ctx.get("no_candidates"):
        return naive_day(ctx["request"]["hard"], ctx["day_index"], ctx["date"], set(ctx.get("used_names", [])))
    req = ctx["request"]
    hard, soft = req["hard"], req["soft"]
    n = hard["travelers"]
    diet = set(hard["dietary_restrictions"])
    budget_per_day = float(ctx.get("budget_per_day") or 1e9)
    tight = any("预算" in f for f in ctx.get("feedback", []))

    attrs = list(ctx.get("attractions", []))
    must = set(ctx.get("must_visit_remaining", []))
    interests = set(soft.get("interests", []))
    # must-visit first, then interest match, then rating. Spread must-visits over remaining days.
    attrs.sort(key=lambda a: (a["name"] not in must, a["category"] not in interests, -a["rating"]))
    k = _PER_DAY[soft.get("pace", "moderate")]
    if tight:
        k = max(2, k - 1)
    days_left = max(1, int(ctx.get("days_remaining", 1)))
    quota_must = -(-len(must) // days_left)  # ceil
    picked: list[dict[str, Any]] = []
    for a in attrs:
        if len(picked) >= k:
            break
        is_must = a["name"] in must
        if is_must and sum(p["name"] in must for p in picked) >= quota_must:
            continue
        if tight and not is_must and a["ticket_price"] > 0:
            continue
        if not is_must and a["recommended_hours"] > 5 and picked:
            continue
        picked.append(a)
    day_attrs = [a for a in picked if a["category"] != "nightlife"]
    evening = [a for a in picked if a["category"] == "nightlife"][:1]
    day_attrs.sort(key=lambda a: _t(a["close_time"]))

    rests = [r for r in ctx.get("restaurants", []) if diet <= set(r["dietary_tags"])] or ctx.get("restaurants", [])
    used = set(ctx.get("used_restaurant_ids", []))
    rests = sorted(rests, key=lambda r: (r["id"] in used, -r["rating"]))
    if tight or sum(r["avg_price_per_person"] for r in rests[:2]) * n > budget_per_day * 0.5:
        rests = sorted(rests, key=lambda r: (r["id"] in used, r["avg_price_per_person"]))

    acts: list[dict[str, Any]] = []
    cursor = _t("09:00")

    def visit(a: dict[str, Any]) -> None:
        nonlocal cursor
        start = max(cursor, _t(a["open_time"]))
        dur = min(a["recommended_hours"], 4.5 if len(day_attrs) > 1 else 8) * 60
        end = min(_add(start, dur), _t(a["close_time"]))
        if _mins(start, end) < 45:
            return
        acts.append(
            {
                "start_time": start.isoformat("minutes"),
                "end_time": end.isoformat("minutes"),
                "kind": "attraction",
                "title": a["name"],
                "ref_id": a["id"],
                "est_cost": a["ticket_price"] * n,
            }
        )
        cursor = end
        acts.append(
            {
                "start_time": cursor.isoformat("minutes"),
                "end_time": _add(cursor, 30).isoformat("minutes"),
                "kind": "transit",
                "title": "前往下一站",
                "ref_id": None,
                "est_cost": 4.0 * n,
            }
        )
        cursor = _add(cursor, 30)

    def meal(label: str, earliest: str, minutes: int, r: dict[str, Any] | None) -> None:
        nonlocal cursor
        if r is None:
            return
        start = max(cursor, _t(earliest))
        acts.append(
            {
                "start_time": start.isoformat("minutes"),
                "end_time": _add(start, minutes).isoformat("minutes"),
                "kind": "meal",
                "title": f"{label}：{r['name']}",
                "ref_id": r["id"],
                "est_cost": r["avg_price_per_person"] * n,
            }
        )
        cursor = _add(start, minutes)

    lunch_r = rests[0] if rests else None
    dinner_r = rests[1] if len(rests) > 1 else lunch_r
    if day_attrs:
        visit(day_attrs[0])
    meal("午餐", "11:30", 60, lunch_r)
    for a in day_attrs[1:]:
        if _mins(max(cursor, _t(a["open_time"])), _t(a["close_time"])) >= 60 and cursor < _t("17:00"):
            visit(a)
    meal("晚餐", "17:30", 90, dinner_r)
    for a in evening:
        if cursor < _t(a["close_time"]):
            visit(a)
    if acts and acts[-1]["kind"] == "transit":
        acts.pop()

    if ({"itinerary_overlap_once", "itinerary_overlap_always"} & faults) and len(acts) >= 2:
        faults.discard("itinerary_overlap_once")
        acts[1]["start_time"] = _add(_t(acts[0]["start_time"]), 30).isoformat("minutes")  # deliberate overlap

    weather = ctx.get("weather") or {}
    theme = "、".join(a["title"] for a in acts if a["kind"] == "attraction")[:30] or "自由活动"
    if weather.get("condition") == "rain":
        theme += "（有雨，备好雨具）"
    return {"day_index": ctx["day_index"], "date": ctx["date"], "theme": theme, "activities": acts}


# =========================================================================== budget
def respond_budget_advice(ctx: dict[str, Any]) -> dict[str, Any]:
    rep = ctx["report"]
    lines = sorted(rep["lines"], key=lambda li: -li["amount"])
    tips = []
    for li in lines[:3]:
        if li["category"] == "hotel":
            tips.append("换成更低档次或非核心商圈的酒店，住宿通常是最大开销。")
        elif li["category"] == "meals":
            tips.append("部分正餐换成当地小吃或人均更低的餐厅。")
        elif li["category"] == "tickets":
            tips.append("用免费景点（公园、街区、博物馆）替换部分收费景点。")
        elif li["category"] == "transit":
            tips.append("优先选择地铁公交，景点按区域集中安排减少打车。")
    if rep["min_feasible_cost"] > rep["budget"]:
        tips.insert(0, f"即使选最便宜的组合也需约 {rep['min_feasible_cost']:.0f} 元，建议提高预算或缩短天数。")
    return {"suggestions": tips[:5] or ["当前方案已接近最省，建议适当提高预算。"]}


# =========================================================================== no-retrieval planning
# Emulates a model planning from "parametric knowledge": it knows the city's famous
# places (the catalog) but not live prices / opening hours / dietary tags, so it
# ignores them. This is what baseline and the no-research ablation are compared on.
_NAIVE_SLOTS = [("09:00", "12:00"), ("13:30", "16:30"), ("19:30", "21:00")]


def _catalog(city: str) -> dict[str, Any]:
    return _load_all().get(city, {"hotels": [], "restaurants": [], "attractions": []})


def naive_day(hard: dict[str, Any], day_index: int, day: str, used: set[str]) -> dict[str, Any]:
    cat = _catalog(hard["destination"])
    n = hard["travelers"]
    must = [a for a in cat["attractions"] if a["name"] in hard.get("must_visit", []) and a["name"] not in used]
    rest = sorted((a for a in cat["attractions"] if a["name"] not in used and a not in must), key=lambda a: -a["rating"])
    picks = (must + rest)[:3]
    rests = sorted(cat["restaurants"], key=lambda r: -r["rating"])
    lunch, dinner = rests[(2 * day_index) % len(rests)], rests[(2 * day_index + 1) % len(rests)]
    acts: list[dict[str, Any]] = []
    for (s, e), a in zip(_NAIVE_SLOTS, picks, strict=False):
        acts.append(
            {
                "start_time": s,
                "end_time": e,
                "kind": "attraction",
                "title": a["name"],
                "ref_id": a["name"],
                "est_cost": a["ticket_price"] * n,
            }
        )
        used.add(a["name"])
    acts.append(
        {
            "start_time": "12:00",
            "end_time": "13:30",
            "kind": "meal",
            "title": f"午餐：{lunch['name']}",
            "ref_id": lunch["name"],
            "est_cost": lunch["avg_price_per_person"] * n,
        }
    )
    acts.append(
        {
            "start_time": "18:00",
            "end_time": "19:30",
            "kind": "meal",
            "title": f"晚餐：{dinner['name']}",
            "ref_id": dinner["name"],
            "est_cost": dinner["avg_price_per_person"] * n,
        }
    )
    acts.sort(key=lambda a: a["start_time"])
    return {"day_index": day_index, "date": day, "theme": "、".join(p["name"] for p in picks), "activities": acts}


def respond_baseline(messages: list[BaseMessage], ctx: dict[str, Any]) -> dict[str, Any]:
    """Single-shot: read the raw request, emit a whole Itinerary, no tools, no checks."""
    today = date.fromisoformat(ctx.get("today", date.today().isoformat()))
    r = extract_request([ctx.get("request_text", "")], today, None)
    city = r.get("destination") or next(iter(_load_all()))
    start = date.fromisoformat(r.get("start_date") or today.isoformat())
    hard = {"destination": city, "travelers": r.get("travelers", 1), "must_visit": r.get("must_visit", [])}
    used: set[str] = set()
    days = [naive_day(hard, i, (start + timedelta(days=i)).isoformat(), used) for i in range(int(r.get("days") or 1))]
    hotels = sorted(_catalog(city)["hotels"], key=lambda h: -h["rating"])
    return {"hotel_id": hotels[0]["name"] if hotels and len(days) > 1 else None, "days": days, "summary": ""}


# =========================================================================== judge
def respond_judge(ctx: dict[str, Any]) -> dict[str, Any]:
    """Heuristic stand-in for the LLM judge (scores are NOT meaningful quality numbers)."""
    soft, hard = ctx["request"]["soft"], ctx["request"]["hard"]
    days = ctx["plan"]["days"]
    attrs = [a for d in days for a in d["activities"] if a["kind"] == "attraction"]
    cats = {a.get("category") for a in attrs if a.get("category")}
    per_day = len(attrs) / max(1, hard["days"])
    richness = min(95, 40 + 12 * per_day + 5 * len(cats))
    interests = set(soft.get("interests", []))
    match = sum(a.get("category") in interests for a in attrs) / len(attrs) if attrs and interests else 0.6
    pace_ok = {"relaxed": per_day <= 2.2, "moderate": 1.5 <= per_day <= 3.5, "packed": per_day >= 2.5}[soft.get("pace", "moderate")]
    preference = round(45 + 40 * match + (10 if pace_ok else -10))
    violations = len(ctx.get("violations", []))
    reasonable = max(10, 90 - 12 * violations)
    return {
        "richness": round(richness),
        "preference_fit": max(0, min(100, preference)),
        "reasonableness": reasonable,
        "rationale": f"mock judge: {len(attrs)} attractions, interest match {match:.0%}, {violations} rule violations",
    }
