"""Systems under test and the code that runs one sample through one system.

* full         the complete graph (auto-selection instead of a human)
* no_revision  ablation: validator still runs, but never routes back to Itinerary
* baseline_catalog  one LLM call: raw request + the destination's full catalog -> whole itinerary;
               no tools, no validation, no revision. Same data as `full`, without the multi-step
               pipeline and the checks: this is the comparison that measures the architecture.
* baseline     (not run by default) one LLM call without the catalog. It can only name places from
               memory, which the evaluation cannot verify against the catalog, so its hard-pass rate
               is 0% by construction; kept for reference.
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
from trip_agents.evals.world import evaluate_plan, world_bundle
from trip_agents.graph.builder import build_graph
from trip_agents.graph.deps import AgentDeps, GraphOptions, LocalToolExecutor
from trip_agents.llm.context import render_context
from trip_agents.llm.structured import invoke_structured
from trip_agents.observability.tracer import TraceRecorder
from trip_agents.schemas import BudgetReport, Itinerary, TripRequest
from trip_agents.validation.names import matches_any

BASELINE_SYSTEM = """你是旅行规划助手。根据用户的请求，直接输出完整的多日行程（Itinerary 结构）。
每天包含景点和午餐、晚餐，给出时间、名称（ref_id 填景点或餐厅名称）和全体同行人的预计花费；hotel_id 填酒店名称。
context.today 是今天的日期。"""

# Same rules as ITINERARY_SYSTEM (graph/prompts.py), for a whole trip at once: the model reads the
# request itself and picks the hotel. Nothing that `full` doesn't get.
BASELINE_CATALOG_SYSTEM = """你是旅行规划助手。根据用户的原始请求（context.request_text）和 context 中该城市的目录，
一次性输出完整的多日行程（Itinerary 结构）。

要求：
1. 自行从请求中理解出行日期、天数、人数、预算、饮食限制、必去景点和偏好；context.today 是今天的日期。
   每天的 day_index 从 0 开始，date 为当天日期。
2. 只能使用 context.hotels、context.attractions 和 context.restaurants 中的条目：hotel_id 和每个活动的 ref_id 填对应 id；
   多日行程选择一家酒店。
3. 活动按时间顺序排列，不得重叠；景点必须在开放时间内；全天在 07:00-23:30 之间。
4. 每天安排午餐和晚餐，餐厅必须满足用户的全部饮食限制。
5. 优先安排用户的必去景点；参考节奏偏好。
6. est_cost 为全体同行人的总费用（门票×人数，餐饮人均×人数）；尽量让总花费（含住宿）不超过用户的预算。
"""


@dataclass(frozen=True)
class SystemConfig:
    name: str
    kind: Literal["graph", "single_shot"]
    options: GraphOptions = field(default_factory=lambda: GraphOptions(human_in_loop=False))
    description: str = ""
    catalog_context: bool = False  # single_shot only: put the destination's full catalog in the prompt


SYSTEMS: dict[str, SystemConfig] = {
    "full": SystemConfig("full", "graph", GraphOptions(human_in_loop=False), "完整系统"),
    "no_revision": SystemConfig("no_revision", "graph", GraphOptions(human_in_loop=False, enable_revision=False), "去掉回退"),
    "baseline_catalog": SystemConfig(
        "baseline_catalog", "single_shot", description="单次生成，给完整目录，无校验无回退", catalog_context=True
    ),
    "baseline": SystemConfig("baseline", "single_shot", description="单次生成，无目录无校验（默认不跑）"),
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
    # full plan, kept so `evals.rejudge` can score it with another judge without regenerating
    itinerary: dict[str, Any] | None = None
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
    # audit of the model's own cost estimates (the budget check uses catalog prices); refs = activities
    # that resolve to the catalog
    cost_model_refs: float | None = None
    cost_catalog_refs: float | None = None
    plan_total_model: float | None = None  # lodging + the model's est_cost of every activity
    plan_total_catalog: float | None = None  # the authoritative total the evaluation uses
    budget_flip: bool | None = None  # would the over-budget verdict differ if the model's estimates were trusted?
    extracted: dict[str, Any] | None = None  # the system's extracted hard constraints (graph systems), for error listings
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
        # same rule as the must-visit check (validation/names.py): short names / aliases of the catalog name count
        "must_visit": len(p.must_visit) == len(g.must_visit)
        and all(matches_any(m, p.must_visit) for m in g.must_visit)
        and all(matches_any(m, g.must_visit) for m in p.must_visit),
    }


def _signature(it: Itinerary | None) -> list[list[str]]:
    if it is None:
        return []
    return [sorted(a.ref_id or a.title for a in d.activities if a.kind == "attraction") for d in it.days]


def set_cost_audit(rec: RunRecord, evaluated: BudgetReport | None, runtime: BudgetReport | None, budget: float) -> None:
    if evaluated is None:
        return
    src = runtime if runtime is not None and runtime.model_refs_total is not None else evaluated
    rec.cost_model_refs, rec.cost_catalog_refs = src.model_refs_total, src.catalog_refs_total
    rec.plan_total_model = src.model_plan_total
    rec.plan_total_catalog = evaluated.total
    if rec.plan_total_model is not None:
        rec.budget_flip = (rec.plan_total_model > budget) != (evaluated.total > budget)


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
    runtime_budget: BudgetReport | None = None
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
            if (req := state.get("request")) is not None:
                rec.extracted = req.hard.model_dump(mode="json")
            rec.flagged_infeasible = (not final.feasible) if final else None
            rec.revisions = state.get("revision", 0)
            runtime_budget = state.get("budget")
        else:
            if system.catalog_context:
                # The catalog is chosen by the gold destination (one field the model is given; full extracts
                # the destination ~100% correctly). Everything else must be read from request_text. No weather:
                # fetching it would reveal the dates, which the model has to read itself.
                world = world_bundle(sample.gold.hard.destination)
                ctx = render_context(
                    {
                        "task": "baseline_catalog_plan",
                        "today": EVAL_TODAY.isoformat(),
                        "request_text": sample.request_text,
                        "hotels": [h.model_dump(mode="json") for h in world.hotels],
                        "attractions": [a.model_dump(mode="json") for a in world.attractions],
                        "restaurants": [r.model_dump(mode="json") for r in world.restaurants],
                    }
                )
                system_prompt = BASELINE_CATALOG_SYSTEM
            else:
                ctx = render_context({"task": "baseline_plan", "today": EVAL_TODAY.isoformat(), "request_text": sample.request_text})
                system_prompt = BASELINE_SYSTEM
            res = invoke_structured(
                make_model(),
                Itinerary,
                [SystemMessage(content=system_prompt), HumanMessage(content=ctx + "\n\n" + sample.request_text)],
                max_retries=1,
                stage=system.name,
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
    # graph systems rewrite est_cost to catalog prices, so the model's original numbers come from the run's
    # own budget report; for the baseline the plan still carries them and the evaluation's report has them
    set_cost_audit(rec, ev.budget, runtime_budget, sample.gold.hard.budget_total)
    rec.n_refs, rec.hallucinated_refs = ev.n_refs, ev.hallucinated_refs
    rec.plan_signature = _signature(itinerary)
    rec.itinerary = itinerary.model_dump(mode="json") if itinerary is not None else None

    s = trace.summary()
    rec.tools_used = sorted({t.name for t in trace.tool_calls})
    rec.tool_calls, rec.invalid_tool_calls, rec.duplicate_tool_calls = s["tool_calls"], s["tool_errors"], s["tool_duplicates"]
    rec.research_steps = sum(c.node == "research.agent" for c in trace.llm_calls)
    rec.parse_failures, rec.skipped_items = s["parse_failures"], s["skipped_items"]
    rec.llm_calls, rec.input_tokens, rec.output_tokens, rec.cost_usd = s["llm_calls"], s["input_tokens"], s["output_tokens"], s["cost_usd"]
    rec.node_ms = s["node_ms"]

    # Judge only plans that pass every hard constraint: quality of an invalid plan is not comparable
    # (an invented landmark or a missing day reads as "rich"), and it saves the judge calls.
    if judge_model is not None and itinerary is not None and ev.hard_pass:
        jr = judge_plan(judge_model, itinerary, sample.gold, ev.errors)
        if jr.scores:
            rec.judge, rec.judge_total = jr.scores.model_dump(), jr.total
    return rec
