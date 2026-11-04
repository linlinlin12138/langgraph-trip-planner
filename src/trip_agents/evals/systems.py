"""Systems under test and the code that runs one sample through one system.

* full         the complete graph (auto-selection instead of a human)
* no_revision  ablation: validator still runs, but never routes back to Itinerary
* no_research  ablation: no tool calls; agents plan from model knowledge
* baseline     one LLM call: raw request -> whole itinerary; no tools, no validation, no retries
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from trip_agents.evals.dataset import EVAL_TODAY, Sample
from trip_agents.evals.judge import judge_plan
from trip_agents.evals.world import evaluate_plan
from trip_agents.graph.builder import build_graph
from trip_agents.graph.deps import AgentDeps, GraphOptions, LocalToolExecutor
from trip_agents.llm.context import render_context
from trip_agents.llm.structured import invoke_structured
from trip_agents.observability.tracer import TraceRecorder
from trip_agents.schemas import Itinerary, TripRequest

BASELINE_SYSTEM = """你是旅行规划助手。根据用户的请求，直接输出完整的多日行程（Itinerary 结构）。
每天包含景点和午餐、晚餐，给出时间、名称（ref_id 填景点或餐厅名称）和全体同行人的预计花费；hotel_id 填酒店名称。
context.today 是今天的日期。"""


@dataclass(frozen=True)
class SystemConfig:
    name: str
    kind: Literal["graph", "single_shot"]
    options: GraphOptions = field(default_factory=lambda: GraphOptions(human_in_loop=False))
    description: str = ""


SYSTEMS: dict[str, SystemConfig] = {
    "full": SystemConfig("full", "graph", GraphOptions(human_in_loop=False), "完整系统"),
    "no_revision": SystemConfig("no_revision", "graph", GraphOptions(human_in_loop=False, enable_revision=False), "去掉回退"),
    "no_research": SystemConfig("no_research", "graph", GraphOptions(human_in_loop=False, enable_research=False), "去掉检索"),
    "baseline": SystemConfig("baseline", "single_shot", description="单次生成，无检索无校验"),
}


class RunRecord(BaseModel):
    sample_id: str
    system: str
    rep: int
    difficulty: str
    feasible_gold: bool
    completed: bool
    error: str | None = None
    # outcome
    structure_ok: bool = False
    hard_pass: bool = False
    violations: list[str] = Field(default_factory=list)
    hallucinated_refs: int = 0
    n_refs: int = 0
    extraction: dict[str, bool] | None = None  # field -> correct?
    flagged_infeasible: bool | None = None
    judge: dict[str, Any] | None = None
    judge_total: float | None = None
    plan_signature: list[list[str]] = Field(default_factory=list)
    # process
    tools_used: list[str] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    tool_calls: int = 0
    invalid_tool_calls: int = 0
    duplicate_tool_calls: int = 0
    research_steps: int = 0
    revisions: int = 0
    parse_failures: int = 0
    skipped_items: int = 0
    # efficiency
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    wall_ms: float = 0.0
    node_ms: dict[str, float] = Field(default_factory=dict)


def extraction_accuracy(pred: TripRequest | None, gold: TripRequest) -> dict[str, bool]:
    if pred is None:
        return dict.fromkeys(
            ["destination", "start_date", "days", "budget_total", "travelers", "dietary_restrictions", "must_visit"], False
        )
    p, g = pred.hard, gold.hard
    return {
        "destination": p.destination == g.destination,
        "start_date": p.start_date == g.start_date,
        "days": p.days == g.days,
        "budget_total": abs(p.budget_total - g.budget_total) <= 1,
        "travelers": p.travelers == g.travelers,
        "dietary_restrictions": set(p.dietary_restrictions) == set(g.dietary_restrictions),
        "must_visit": set(p.must_visit) == set(g.must_visit),
    }


def _signature(it: Itinerary | None) -> list[list[str]]:
    if it is None:
        return []
    return [sorted(a.ref_id or a.title for a in d.activities if a.kind == "attraction") for d in it.days]


def run_one(
    sample: Sample,
    system: SystemConfig,
    make_model: Callable[[], BaseChatModel],
    *,
    rep: int = 0,
    judge_model: BaseChatModel | None = None,
) -> RunRecord:
    rec = RunRecord(
        sample_id=sample.id,
        system=system.name,
        rep=rep,
        difficulty=sample.difficulty,
        feasible_gold=sample.feasible,
        completed=False,
        expected_tools=sample.expected_tools,
    )
    tracer = TraceRecorder()
    itinerary: Itinerary | None = None
    t0 = time.perf_counter()
    try:
        if system.kind == "graph":
            deps = AgentDeps(model=make_model(), tools=LocalToolExecutor(), today=EVAL_TODAY)
            graph = build_graph(deps, system.options)
            state = graph.invoke(
                {"messages": [HumanMessage(content=sample.request_text)], "user_id": f"eval-{sample.id}"},
                {"callbacks": [tracer], "recursion_limit": 80},
            )
            final = state.get("final")
            itinerary = final.itinerary if final else state.get("itinerary")
            rec.extraction = extraction_accuracy(state.get("request"), sample.gold)
            rec.flagged_infeasible = (not final.feasible) if final else None
            rec.revisions = state.get("revision", 0)
        else:
            ctx = render_context({"task": "baseline_plan", "today": EVAL_TODAY.isoformat(), "request_text": sample.request_text})
            res = invoke_structured(
                make_model(),
                Itinerary,
                [SystemMessage(content=BASELINE_SYSTEM), HumanMessage(content=ctx + "\n\n" + sample.request_text)],
                max_retries=1,
                stage="baseline",
                item="plan",
                config={"callbacks": [tracer]},
            )
            itinerary = res.value
        rec.completed = True
    except Exception as e:  # a crash is a result, not a reason to abort the whole eval
        rec.error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
    trace = tracer.finish()
    rec.wall_ms = round((time.perf_counter() - t0) * 1000, 2)

    ev = evaluate_plan(itinerary, sample.gold)
    rec.structure_ok, rec.hard_pass, rec.violations = ev.structure_ok, ev.hard_pass, ev.errors
    rec.n_refs, rec.hallucinated_refs = ev.n_refs, ev.hallucinated_refs
    rec.plan_signature = _signature(itinerary)

    s = trace.summary()
    rec.tools_used = sorted({t.name for t in trace.tool_calls})
    rec.tool_calls, rec.invalid_tool_calls, rec.duplicate_tool_calls = s["tool_calls"], s["tool_errors"], s["tool_duplicates"]
    rec.research_steps = sum(c.node == "research.agent" for c in trace.llm_calls)
    rec.parse_failures, rec.skipped_items = s["parse_failures"], s["skipped_items"]
    rec.llm_calls, rec.input_tokens, rec.output_tokens, rec.cost_usd = s["llm_calls"], s["input_tokens"], s["output_tokens"], s["cost_usd"]
    rec.node_ms = s["node_ms"]

    if judge_model is not None and itinerary is not None and ev.structure_ok:
        jr = judge_plan(judge_model, itinerary, sample.gold, ev.errors)
        if jr.scores:
            rec.judge, rec.judge_total = jr.scores.model_dump(), jr.total
    return rec
