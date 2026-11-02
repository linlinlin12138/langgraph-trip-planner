"""Terminal demo.

uv run python -m trip_agents.cli                       # interactive
uv run python -m trip_agents.cli --auto "我想11月20日去杭州玩3天，预算6000"
uv run python -m trip_agents.cli --thread abc123       # resume a saved thread
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from trip_agents.runtime import PlannerSession, Turn, create_planner


def _print_turn(turn: Turn) -> None:
    print(f"  [nodes] {' → '.join(turn.nodes)}")
    for m in turn.new_messages:
        print(f"助手: {m}")


def _answer_interrupt(intr: dict[str, Any], auto: bool) -> Any:
    kind = intr["type"]
    if kind == "clarify":
        print(f"助手: {intr['question']}")
        return input("你: ")
    if kind == "select_candidates":
        d = intr["defaults"]
        print("\n候选酒店:")
        for h in intr["hotels"]:
            print(
                f"  {'*' if h['id'] == d['hotel_id'] else ' '} {h['id']} {h['name']} {h['tier']}"
                f" {h['price_per_night']:.0f}元/晚 ★{h['rating']}"
            )
        print("候选景点:")
        for a in intr["attractions"]:
            print(f"  {'*' if a['id'] in d['attraction_ids'] else ' '} {a['id']} {a['name']} [{a['category']}] 门票{a['ticket_price']:.0f}")
        print("候选餐厅:")
        for r in intr["restaurants"]:
            print(
                f"  {'*' if r['id'] in d['restaurant_ids'] else ' '} {r['id']} {r['name']}"
                f" 人均{r['avg_price_per_person']:.0f} {r['dietary_tags']}"
            )
        if auto:
            return d
        raw = input("回车接受默认（*），或输入要保留的景点 id（逗号分隔）: ").strip()
        return d if not raw else {**d, "attraction_ids": [x.strip() for x in raw.split(",")]}
    if kind == "confirm":
        return {"confirmed": True} if auto else {"confirmed": input("确认这个行程吗？[Y/n] ").strip().lower() != "n"}
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", metavar="REQUEST", help="run non-interactively with this request")
    ap.add_argument("--thread", help="resume an existing thread id")
    ap.add_argument("--faults", default="", help="mock faults, comma separated (e.g. itinerary_overlap_once)")
    args = ap.parse_args()

    graph = create_planner(faults={f for f in args.faults.split(",") if f})
    session = PlannerSession(graph, thread_id=args.thread)
    print(f"thread_id = {session.thread_id}")
    intr = session.pending_interrupt() if args.thread else None
    if intr is None:
        turn = session.start(args.auto or input("你: "))
        _print_turn(turn)
        intr = turn.interrupt
    while intr is not None:
        turn = session.resume(_answer_interrupt(intr, auto=bool(args.auto)))
        _print_turn(turn)
        intr = turn.interrupt

    final = session.values.get("final")
    if final:
        for d in final.itinerary.days:
            print(f"\n== 第 {d.day_index + 1} 天 {d.date} {d.theme}")
            for a in d.activities:
                print(f"  {a.start_time:%H:%M}-{a.end_time:%H:%M} {a.title}  ¥{a.est_cost:.0f}")
        print("\n预算明细:", json.dumps([li.model_dump() for li in final.budget.lines], ensure_ascii=False))


if __name__ == "__main__":
    main()
