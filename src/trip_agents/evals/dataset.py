"""Evaluation dataset: structured constraint combos -> natural-language requests.

    uv run python -m trip_agents.evals.dataset --n 200 --seed 42 --rewrite template
    # LLM paraphrases: use trip_agents.evals.rewrite (back-checked against gold), not --rewrite llm

Each sample keeps the structured constraints it was generated from (``gold``) as the
answer key, plus the tools a reasonable agent should use. Difficulty is decided by
the generator (number of constraints, budget tightness), not by model behaviour.
Feasibility is computed against the full catalog: if even the cheapest compliant
plan exceeds the budget the sample is labelled infeasible.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel

from trip_agents.config import PROJECT_ROOT
from trip_agents.evals.world import min_feasible_cost, world_bundle
from trip_agents.schemas import HardConstraints, SoftPreferences, TripRequest
from trip_agents.tools.mock_backend import _load_all, supported_cities

DATASET_DIR = PROJECT_ROOT / "evals" / "datasets"
EVAL_TODAY = date(2026, 11, 2)  # fixed "today" so relative dates in requests are reproducible

Difficulty = Literal["easy", "medium", "hard"]
DIETS = ["no_spicy", "no_seafood"]
INTERESTS = ["history", "nature", "museum", "food", "shopping", "nightlife", "culture"]


class Sample(BaseModel):
    id: str
    difficulty: Difficulty
    feasible: bool
    gold: TripRequest
    expected_tools: list[str]
    request_text: str
    rewrite: Literal["template", "llm", "llm_challenge"]  # llm_challenge: evals.challenge, human-reviewed, no back-check
    min_feasible_cost: float
    typical_cost: float
    # provenance of an LLM rewrite (rewriter / checker models, attempts); see evals/rewrite.py
    rewrite_info: dict[str, Any] | None = None


# ----------------------------------------------------------------------------- constraints
def typical_cost(req: TripRequest) -> float:
    """What a normal mid-range trip costs; budgets are drawn relative to this."""
    w = world_bundle(req.hard.destination)
    h = req.hard
    mid_hotels = sorted(x.price_per_night for x in w.hotels if x.tier == "midrange")
    meal = sorted(r.avg_price_per_person for r in w.restaurants)
    hotel = mid_hotels[len(mid_hotels) // 2] * max(h.days - 1, 0) * ((h.travelers + 1) // 2)
    meals = meal[len(meal) // 2] * 2 * h.days * h.travelers
    tickets = 50 * 2 * h.days * h.travelers
    return round(hotel + meals + tickets, -1)


def _compliant_restaurants(city: str, diet: list[str]) -> int:
    return sum(set(diet) <= set(r.dietary_tags) for r in world_bundle(city).restaurants)


def sample_constraints(rng: random.Random, difficulty: Difficulty, infeasible: bool) -> TripRequest:
    city = rng.choice(supported_cities())
    spec: dict[str, Any] = {
        "easy": {"days": (1, 3), "trav": (1, 2), "diet": (0, 0), "must": (0, 0), "int": (0, 1), "budget": (1.6, 2.2)},
        "medium": {"days": (2, 4), "trav": (1, 3), "diet": (0, 1), "must": (0, 1), "int": (1, 2), "budget": (1.15, 1.5)},
        "hard": {
            "days": (3, 5),
            "trav": (2, 4),
            "diet": (1, 2),
            "must": (1, 2),
            "int": (2, 3),
            "budget": (1.1, 1.6),
        },  # hard: x min feasible cost
    }[difficulty]
    days = rng.randint(*spec["days"])
    travelers = rng.randint(*spec["trav"])
    for _ in range(50):  # resample diet combos until the city can actually serve them
        diet = rng.sample(DIETS, rng.randint(*spec["diet"]))
        if _compliant_restaurants(city, diet) >= 2:
            break
    # must-visit: only the hand-curated landmarks people actually name (not every mosque / mall in the catalog)
    landmarks = [a["name"] for a in _load_all()[city]["attractions"] if a.get("source", "curated") == "curated"]
    must = rng.sample(landmarks, rng.randint(*spec["must"]))
    start = EVAL_TODAY + timedelta(days=rng.randint(20, 170))
    soft = SoftPreferences(
        pace=rng.choice(["relaxed", "moderate", "moderate", "packed"]),
        hotel_tier=rng.choice([None, None, "budget", "midrange", "luxury"]) if days > 1 else None,
        interests=rng.sample(INTERESTS, rng.randint(*spec["int"])),
    )
    hard = HardConstraints(
        destination=city, start_date=start, days=days, budget_total=1, travelers=travelers, dietary_restrictions=diet, must_visit=must
    )
    req = TripRequest(hard=hard, soft=soft)
    if infeasible:
        budget = min_feasible_cost(req) * rng.uniform(0.4, 0.7)
    elif difficulty == "hard":  # tight: pricier picks overshoot, so budget checks and revisions matter
        budget = min_feasible_cost(req) * rng.uniform(*spec["budget"])
    else:
        budget = max(typical_cost(req) * rng.uniform(*spec["budget"]), min_feasible_cost(req) * 1.25)
    req.hard.budget_total = float(max(100, round(budget, -2)))
    return req


def expected_tools(req: TripRequest) -> list[str]:
    tools = ["get_weather", "search_attractions", "search_restaurants"]
    if req.hard.days > 1:
        tools += ["search_hotels", "get_route"]
    return sorted(tools)


# ----------------------------------------------------------------------------- rewriting
_CN = "零一两三四五六七八九十"
_PHRASES: dict[str, dict[str, Any]] = {
    "diet": {
        "no_spicy": ["不能吃辣", "完全吃不了辣"],
        "no_seafood": ["海鲜过敏", "不吃海鲜"],
    },
    "pace": {"relaxed": ["节奏轻松点", "别太累"], "packed": ["行程紧凑一点", "想特种兵式多去几个地方"], "moderate": [""]},
    "tier": {"budget": ["住经济型就行", "住便宜点的"], "midrange": ["住舒适型酒店"], "luxury": ["想住高档一点的五星酒店"]},
    "interest": {
        "history": "历史古迹",
        "nature": "自然风景",
        "museum": "博物馆",
        "food": "美食小吃",
        "shopping": "逛街购物",
        "nightlife": "夜景",
        "culture": "文化艺术",
    },
}


def _cn(n: int) -> str:
    return _CN[n] if n <= 10 else str(n)


def template_rewrite(req: TripRequest, rng: random.Random) -> str:
    """Deterministic colloquial rendering. Every phrase is one the rule-based mock extractor understands."""
    h, s = req.hard, req.soft
    options = {
        1: ["我一个人", "我自己"],
        2: ["我和女朋友", "我和一个朋友", "我们两个人"],  # not "我和朋友": could be several friends
        3: ["一家三口", "我们3个人"],
        4: ["我们4个人", "我们四个人"],
    }
    who = rng.choice(options.get(h.travelers, [f"我们{h.travelers}个人"]))
    if h.travelers == 1 and who == "我自己":
        who = "我自己一个人"
    when = rng.choice(
        [f"{h.start_date.month}月{h.start_date.day}日", f"{h.start_date.month}月{h.start_date.day}号", h.start_date.isoformat()]
    )
    length = rng.choice([f"玩{h.days}天", f"玩{_cn(h.days)}天", f"{_cn(h.days)}日游"])
    per_person = h.travelers > 1 and h.budget_total % h.travelers == 0 and rng.random() < 0.3
    budget = (
        f"人均预算{h.budget_total / h.travelers:.0f}元"
        if per_person
        else rng.choice([f"预算{h.budget_total:.0f}", f"总预算{h.budget_total:.0f}元"])
    )
    parts = [f"{who}{when}去{h.destination}{length}", budget]
    parts += [rng.choice(_PHRASES["diet"][d]) for d in h.dietary_restrictions]
    if s.interests:
        parts.append("喜欢" + "和".join(_PHRASES["interest"][i] for i in s.interests))
    parts += [f"一定要去{m}" for m in h.must_visit]
    pace = rng.choice(_PHRASES["pace"][s.pace])
    if pace:
        parts.append(pace)
    if s.hotel_tier:
        parts.append(rng.choice(_PHRASES["tier"][s.hotel_tier]))
    head, tail = parts[:1], parts[1:]
    rng.shuffle(tail)
    return "，".join(head + tail) + rng.choice(["。", "，帮我规划一下。", "，有什么推荐？"])


REWRITE_PROMPT = """把下面的旅行需求改写成一句自然、口语化的中文请求，像真实用户在聊天框里打字。
要求：必须包含全部信息且数值准确（日期、天数、人数、总预算或人均预算、饮食限制、必去景点、兴趣、节奏、住宿档次），
不要添加任何额外的约束，不要使用列表格式。只输出改写后的句子。

需求（JSON）：
{gold}"""


def llm_rewrite(req: TripRequest, model: Any) -> str:
    from langchain_core.messages import HumanMessage

    msg = model.invoke([HumanMessage(content=REWRITE_PROMPT.format(gold=req.model_dump_json(indent=1)))])
    return str(msg.content).strip()


# ----------------------------------------------------------------------------- build
def build_dataset(n: int, seed: int, rewrite: Literal["template", "llm"] = "template", model: Any = None) -> list[Sample]:
    rng = random.Random(seed)
    n_easy, n_medium = round(n * 0.35), round(n * 0.40)
    plan = cast(list[Difficulty], ["easy"] * n_easy + ["medium"] * n_medium + ["hard"] * (n - n_easy - n_medium))
    n_infeasible = max(1, round(n * 0.06))
    infeasible_idx = set(rng.sample([i for i, d in enumerate(plan) if d != "easy"], n_infeasible))
    samples = []
    for i, diff in enumerate(plan):
        srng = random.Random(f"{seed}-{i}")
        req = sample_constraints(srng, diff, i in infeasible_idx)
        text = template_rewrite(req, srng) if rewrite == "template" else llm_rewrite(req, model)
        mfc = min_feasible_cost(req)
        samples.append(
            Sample(
                id=f"s{i:03d}",
                difficulty=diff,
                feasible=req.hard.budget_total >= mfc,
                gold=req,
                expected_tools=expected_tools(req),
                request_text=text,
                rewrite=rewrite,
                min_feasible_cost=mfc,
                typical_cost=typical_cost(req),
            )
        )
    return samples


def split(samples: list[Sample], dev_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    """Stratified by difficulty x feasibility, so both splits keep the same mix."""
    rng = random.Random(seed)
    dev, test = [], []
    groups: dict[tuple[str, bool], list[Sample]] = {}
    for s in samples:
        groups.setdefault((s.difficulty, s.feasible), []).append(s)
    for key in sorted(groups):
        g = groups[key][:]
        rng.shuffle(g)
        k = round(len(g) * dev_ratio)
        dev += g[:k]
        test += g[k:]
    return sorted(dev, key=lambda s: s.id), sorted(test, key=lambda s: s.id)


def save(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(s.model_dump_json() + "\n" for s in samples), encoding="utf-8")


def load(path: str | Path) -> list[Sample]:
    return [Sample.model_validate_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev-ratio", type=float, default=0.3)
    ap.add_argument("--rewrite", choices=["template", "llm"], default="template")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, default=DATASET_DIR)
    args = ap.parse_args()

    model = None
    if args.rewrite == "llm":
        from trip_agents.llm.factory import get_chat_model

        model = get_chat_model(args.provider, args.model, temperature=0.9)
    samples = build_dataset(args.n, args.seed, args.rewrite, model)
    dev, test = split(samples, args.dev_ratio, args.seed)
    save(dev, args.out / "dev.jsonl")
    save(test, args.out / "test.jsonl")
    meta = {
        "n": len(samples),
        "seed": args.seed,
        "rewrite": args.rewrite,
        "dev": len(dev),
        "test": len(test),
        "difficulty": Counter(s.difficulty for s in samples),
        "infeasible": sum(not s.feasible for s in samples),
        "eval_today": EVAL_TODAY.isoformat(),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=dict), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
