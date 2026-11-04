"""Hand-written ReAct vs the LangGraph Research subgraph, on the same research tasks.

    uv run python -m react_from_scratch.compare --limit 20                     # mock
    uv run python -m react_from_scratch.compare --provider openai --model <m> --limit 20

Both get the identical system prompt, task text, tools and per-tool round limit.
Reported per implementation: success rate (got weather + attractions + restaurants
[+ hotels if multi-day]), tool precision/recall vs the sample's expected tools,
steps, invalid / duplicate calls, tokens, latency -- plus logical lines of code.
"""

from __future__ import annotations

import argparse
import ast
import random
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from react_from_scratch.client import make_client
from react_from_scratch.loop import run_react
from react_from_scratch.registry import from_trip_tools
from react_from_scratch.types import ReactConfig
from trip_agents.config import PROJECT_ROOT
from trip_agents.evals.dataset import DATASET_DIR, Sample, load
from trip_agents.evals.metrics import percentile, tool_precision_recall
from trip_agents.evals.run import git_commit
from trip_agents.graph.deps import AgentDeps, LocalToolExecutor
from trip_agents.graph.nodes.research import build_research_subgraph, hotel_budget_per_night
from trip_agents.graph.prompts import RESEARCH_SYSTEM
from trip_agents.llm.context import render_context
from trip_agents.llm.factory import DEFAULT_MODELS, get_chat_model
from trip_agents.observability.tracer import TraceRecorder

MAX_STEPS, MAX_ROUNDS = 8, 3


def logical_loc(path: Path, only: set[str] | None = None) -> int:
    """Non-blank, non-comment, non-docstring lines (optionally only inside the named top-level defs)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
            doc_lines.update(range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1))
    spans = None
    if only:
        spans = [
            (n.lineno, n.end_lineno or n.lineno) for n in tree.body if isinstance(n, ast.FunctionDef | ast.ClassDef) and n.name in only
        ]
    count = 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#") or i in doc_lines:
            continue
        if spans is not None and not any(a <= i <= b for a, b in spans):
            continue
        count += 1
    return count


def task_for(s: Sample) -> tuple[str, str]:
    ctx = render_context(
        {"task": "research", "request": s.gold.model_dump(mode="json"), "hotel_budget_per_night": round(hotel_budget_per_night(s.gold), -1)}
    )
    return RESEARCH_SYSTEM.format(max_rounds=MAX_ROUNDS), "请为以下需求检索信息：\n" + ctx


def success(used: set[str], s: Sample) -> bool:
    need = {"get_weather", "search_attractions", "search_restaurants"} | ({"search_hotels"} if s.gold.hard.days > 1 else set())
    return need <= used


def run_langgraph(s: Sample, provider: str, model: str | None) -> dict[str, Any]:
    deps = AgentDeps(
        model=get_chat_model(provider, model),  # type: ignore[arg-type]
        tools=LocalToolExecutor(),
        research_max_steps=MAX_STEPS,
        research_max_rounds_per_tool=MAX_ROUNDS,
    )
    system, task = task_for(s)
    tracer = TraceRecorder()
    t0 = time.perf_counter()
    out = build_research_subgraph(deps).invoke(
        {"messages": [SystemMessage(content=system), HumanMessage(content=task)]}, {"callbacks": [tracer], "recursion_limit": 40}
    )
    ms = (time.perf_counter() - t0) * 1000
    recs = out.get("records", [])
    ok_used = {r["name"] for r in recs if r["ok"]}
    return {
        "used": {r["name"] for r in recs},
        "ok_used": ok_used,
        "steps": out.get("step", 0),
        "calls": len(recs),
        "invalid": sum(not r["ok"] for r in recs),
        "dup": sum(r["duplicate"] for r in recs),
        "tokens": tracer.trace.input_tokens + tracer.trace.output_tokens,
        "ms": ms,
    }


def run_handwritten(s: Sample, provider: str, model: str | None) -> dict[str, Any]:
    system, task = task_for(s)
    t0 = time.perf_counter()
    r = run_react(
        make_client(provider, model),
        from_trip_tools(),
        task,
        system_prompt=system,
        config=ReactConfig(max_steps=MAX_STEPS, max_rounds_per_tool=MAX_ROUNDS),
    )
    ms = (time.perf_counter() - t0) * 1000
    return {
        "used": {x.name for x in r.tool_log},
        "ok_used": {x.name for x in r.tool_log if x.ok},
        "steps": r.steps,
        "calls": len(r.tool_log),
        "invalid": sum(not x.ok for x in r.tool_log),
        "dup": sum(x.duplicate for x in r.tool_log),
        "tokens": r.usage.input_tokens + r.usage.output_tokens,
        "ms": ms,
    }


def summarize(rows: list[dict[str, Any]], samples: list[Sample]) -> dict[str, float | None]:
    pr = [tool_precision_recall(sorted(r["used"]), s.expected_tools) for r, s in zip(rows, samples, strict=True)]
    calls = sum(r["calls"] for r in rows)
    return {
        "success": statistics.fmean(success(r["ok_used"], s) for r, s in zip(rows, samples, strict=True)),
        "precision": statistics.fmean(p for p, _ in pr if p is not None) if any(p is not None for p, _ in pr) else None,
        "recall": statistics.fmean(rc for _, rc in pr if rc is not None),
        "steps": statistics.fmean(r["steps"] for r in rows),
        "invalid_rate": sum(r["invalid"] for r in rows) / calls if calls else None,
        "dup_rate": sum(r["dup"] for r in rows) / calls if calls else None,
        "tokens": statistics.fmean(r["tokens"] for r in rows),
        "ms_mean": statistics.fmean(r["ms"] for r in rows),
        "ms_p95": percentile([r["ms"] for r in rows], 95),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    samples = load(DATASET_DIR / f"{args.split}.jsonl")
    if args.limit < len(samples):
        samples = sorted(random.Random(args.seed).sample(samples, args.limit), key=lambda s: s.id)

    lg = summarize([run_langgraph(s, args.provider, args.model) for s in samples], samples)
    try:
        hw: dict[str, float | None] | None = summarize([run_handwritten(s, args.provider, args.model) for s in samples], samples)
    except NotImplementedError:
        hw = None

    loc_hw = logical_loc(PROJECT_ROOT / "react_from_scratch" / "loop.py")
    loc_lg = logical_loc(
        PROJECT_ROOT / "src/trip_agents/graph/nodes/research.py", {"ResearchState", "call_key", "build_research_subgraph", "_obs_cache"}
    )

    def f(k: str, v: float | None) -> str:
        if v is None:
            return "n/a"
        return f"{v * 100:.1f}%" if k in ("success", "precision", "recall", "invalid_rate", "dup_rate") else f"{v:.1f}"

    labels = {
        "success": "检索成功率",
        "precision": "工具精确率",
        "recall": "工具召回率",
        "steps": "平均步数（模型调用）",
        "invalid_rate": "无效调用率",
        "dup_rate": "重复调用率",
        "tokens": "平均 token",
        "ms_mean": "平均耗时 ms",
        "ms_p95": "耗时 P95 ms",
    }
    model_name = args.model or DEFAULT_MODELS[args.provider]
    lines = ["# 手写 ReAct vs LangGraph Research 子图", ""]
    if args.provider == "mock":
        lines += ["> ⚠️ **MOCK 模式**：两边用同一套确定性检索策略，比较的是循环实现本身（正确性、开销、代码量），不是模型能力。", ""]
    lines += [
        f"- 日期 {datetime.now():%Y-%m-%d %H:%M} · commit `{git_commit()}` · 模型 {args.provider} / `{model_name}`",
        f"- 样本：{args.split} 中按 seed={args.seed} 抽取 {len(samples)} 条，各跑 1 次；"
        f"max_steps={MAX_STEPS}，每类工具最多 {MAX_ROUNDS} 轮",
        "",
        "| 指标 | 手写 ReAct | LangGraph |",
        "|---|---|---|",
    ]
    for k, label in labels.items():
        lines.append(f"| {label} | {f(k, hw[k]) if hw else '未实现'} | {f(k, lg[k])} |")
    hw_loc = f"{loc_hw}（loop.py）" if hw else "—（未实现）"
    lines += [f"| 循环代码行数（逻辑行） | {hw_loc} | {loc_lg}（research.py 子图部分，不含框架本身） |", ""]
    if hw is None:
        lines.append("手写版尚未实现（`react_from_scratch/loop.py` 仍为 TODO），只给出 LangGraph 一侧的数字。")
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
