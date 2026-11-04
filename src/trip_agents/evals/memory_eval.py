"""Memory ablation: two-session scenarios, with vs without long-term memory.

    uv run python -m trip_agents.evals.memory_eval --n 30 --out docs/eval/mock_memory.md

Session 1 states preferences explicitly and is confirmed (memory written).
Session 2, same user, is one of:
  inherit    only city / date / days -> preferences + budget habit should carry over
  override   explicitly different pace and "什么都吃" -> the new statement must win
  past_trip  "像上次X那样" -> destination, length, party, budget from that trip
Gold for session 2 is derived from the scenario. Runs in auto mode (no human),
so a system that lacks required info simply cannot plan -- that is the point.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from trip_agents.evals.dataset import EVAL_TODAY, template_rewrite
from trip_agents.evals.run import git_commit
from trip_agents.evals.world import evaluate_plan, world_bundle
from trip_agents.graph.builder import build_graph
from trip_agents.graph.deps import AgentDeps, GraphOptions, LocalToolExecutor
from trip_agents.llm.factory import DEFAULT_MODELS, get_chat_model
from trip_agents.observability.tracer import TraceRecorder
from trip_agents.schemas import HardConstraints, SoftPreferences, TripRequest
from trip_agents.tools.mock_backend import supported_cities

Kind = Literal["inherit", "override", "past_trip"]
FIELDS = ("destination", "days", "travelers", "budget_total", "dietary_restrictions", "pace", "hotel_tier")


@dataclass
class Scenario:
    id: str
    kind: Kind
    s1_text: str
    s2_text: str
    gold2: TripRequest


def _diet_ok(city: str, diet: list[str]) -> bool:
    return sum(set(diet) <= set(r.dietary_tags) for r in world_bundle(city).restaurants) >= 2


def build_scenarios(n: int, seed: int) -> list[Scenario]:
    rng = random.Random(seed)
    kinds: list[Kind] = ["inherit", "override", "past_trip"]
    out = []
    for i in range(n):
        kind = kinds[i % 3]
        c1, c2 = rng.sample(supported_cities(), 2)
        diet = [d for d in [rng.choice(["vegetarian", "no_pork", "no_spicy", "no_seafood"])] if _diet_ok(c1, [d]) and _diet_ok(c2, [d])]
        travelers = rng.choice([1, 2, 3])
        days1 = rng.randint(2, 4)
        per_day = rng.choice([600, 800, 1000, 1200])
        pace = rng.choice(["relaxed", "packed"])
        tier = rng.choice(["budget", "midrange"])
        d1 = EVAL_TODAY + timedelta(days=rng.randint(10, 60))
        s1 = TripRequest(
            hard=HardConstraints(
                destination=c1,
                start_date=d1,
                days=days1,
                budget_total=per_day * travelers * days1,
                travelers=travelers,
                dietary_restrictions=diet,
            ),
            soft=SoftPreferences(pace=pace, hotel_tier=tier),
        )
        s1_text = template_rewrite(s1, random.Random(f"{seed}-{i}"))
        d2 = d1 + timedelta(days=rng.randint(30, 90))
        when = f"{d2.month}月{d2.day}日"
        if kind == "past_trip":
            gold = s1.model_copy(deep=True)
            gold.hard.start_date = d2
            text = f"{when}像上次{c1}那样再去一次"
        else:
            days2 = rng.randint(2, 4)
            gold = TripRequest(
                hard=HardConstraints(
                    destination=c2,
                    start_date=d2,
                    days=days2,
                    budget_total=round(per_day * travelers * days2, -2),
                    travelers=travelers,
                    dietary_restrictions=diet,
                ),
                soft=SoftPreferences(pace=pace, hotel_tier=tier),
            )
            text = f"{when}去{c2}玩{days2}天"
            if kind == "override":
                new_pace = "packed" if pace == "relaxed" else "relaxed"
                gold.hard.dietary_restrictions, gold.soft.pace = [], new_pace  # type: ignore[assignment]
                text += "，这次什么都吃，" + ("行程紧凑一点" if new_pace == "packed" else "节奏轻松点")
        out.append(Scenario(f"m{i:02d}", kind, s1_text, text, gold))
    return out


def _field_acc(pred: TripRequest | None, gold: TripRequest) -> dict[str, bool]:
    if pred is None:
        return dict.fromkeys(FIELDS, False)
    p, g = pred, gold
    return {
        "destination": p.hard.destination == g.hard.destination,
        "days": p.hard.days == g.hard.days,
        "travelers": p.hard.travelers == g.hard.travelers,
        "budget_total": abs(p.hard.budget_total - g.hard.budget_total) <= 100,
        "dietary_restrictions": set(p.hard.dietary_restrictions) == set(g.hard.dietary_restrictions),
        "pace": p.soft.pace == g.soft.pace,
        "hotel_tier": p.soft.hotel_tier == g.soft.hotel_tier,
    }


def run_scenario(sc: Scenario, make_model: Any, with_memory: bool) -> dict[str, Any]:
    store = InMemoryStore() if with_memory else None
    deps = AgentDeps(model=make_model(), tools=LocalToolExecutor(), today=EVAL_TODAY, clock=lambda: datetime(2026, 11, 2, 12))
    g = build_graph(deps, GraphOptions(human_in_loop=False), store=store)
    g.invoke({"messages": [HumanMessage(content=sc.s1_text)], "user_id": "u"}, {"configurable": {"thread_id": "s1"}})
    tracer = TraceRecorder()
    out = g.invoke(
        {"messages": [HumanMessage(content=sc.s2_text)], "user_id": "u"}, {"configurable": {"thread_id": "s2"}, "callbacks": [tracer]}
    )
    req = out.get("request")
    final = out.get("final")
    ev = evaluate_plan(final.itinerary if final else None, sc.gold2)
    acc = _field_acc(req, sc.gold2)
    tr = tracer.finish()
    return {
        "id": sc.id,
        "kind": sc.kind,
        "memory": with_memory,
        "planned": final is not None,
        "fields": acc,
        "all_fields": all(acc.values()),
        "hard_pass": ev.hard_pass,
        "tokens": tr.input_tokens + tr.output_tokens,
    }


def report(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = ["# 记忆消融评测", ""]
    if meta["mock"]:
        lines += ["> ⚠️ **MOCK 模式**：mock 模型用规则抽取偏好，这里验证的是记忆读写与优先级逻辑，不代表真实模型的抽取能力。", ""]
    lines += ["| 项 | 值 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in meta.items() if k != "mock"] + [""]
    lines += [
        "| 场景 | 系统 | 样本数 | 无需追问即可规划 | 第二次会话字段全对 | 硬约束通过（对照第二次会话标准答案） | 平均 token |",
        "|---|---|---|---|---|---|---|",
    ]
    groups: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["kind"], r["memory"])].append(r)
    for kind in ("inherit", "override", "past_trip"):
        for mem in (True, False):
            g = groups[(kind, mem)]
            if not g:
                continue

            def pct(key: str, g: list[dict[str, Any]] = g) -> str:
                return f"{100 * sum(r[key] for r in g) / len(g):.1f}%"

            name = "full（有记忆）" if mem else "no_memory"
            lines.append(
                f"| {kind} | {name} | {len(g)} | {pct('planned')} | {pct('all_fields')} | {pct('hard_pass')} | "
                f"{sum(r['tokens'] for r in g) / len(g):.0f} |"
            )
    lines += [
        "",
        "说明：",
        "",
        "- 自动模式下不能追问用户。no_memory 缺少预算、人数等信息时无法规划，实际使用中它会转为向用户追问（多一轮交互）。",
        "- mock 的偏好写入和读取用的是同一套规则，所以 full 的准确率由构造决定，接入真实模型后才能衡量抽取质量。",
        "",
        "字段级准确率（有记忆）：",
        "",
    ]
    for kind in ("inherit", "override", "past_trip"):
        g = groups[(kind, True)]
        if g:
            per = {f: sum(r["fields"][f] for r in g) / len(g) for f in FIELDS}
            lines.append(f"- {kind}：" + "，".join(f"{f} {v * 100:.0f}%" for f, v in per.items()))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--provider", default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    scenarios = build_scenarios(args.n, args.seed)

    def make_model() -> Any:
        return get_chat_model(args.provider, args.model)

    rows = [run_scenario(sc, make_model, mem) for sc in scenarios for mem in (True, False)]
    meta = {
        "日期": datetime.now().isoformat(timespec="seconds"),
        "git commit": f"`{git_commit()}`",
        "模型": f"{args.provider} / `{args.model or DEFAULT_MODELS[args.provider]}`",
        "场景数": f"{len(scenarios)}（三类各 1/3）",
        "seed": args.seed,
        "每组运行次数": 1,
        "mock": args.provider == "mock",
    }
    text = report(rows, meta)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
