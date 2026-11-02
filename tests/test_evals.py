"""Dataset generation, ground-truth evaluation, tracer and the runner, end to end on mock."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, time
from pathlib import Path

from langchain_core.messages import HumanMessage

from trip_agents.evals.dataset import EVAL_TODAY, build_dataset, split
from trip_agents.evals.run import main as run_main
from trip_agents.evals.systems import SYSTEMS, run_one
from trip_agents.evals.world import evaluate_plan, normalize_refs, world_bundle
from trip_agents.graph.builder import build_graph
from trip_agents.graph.deps import AgentDeps, GraphOptions, LocalToolExecutor
from trip_agents.llm.mock import MockChatModel
from trip_agents.observability.tracer import TraceRecorder
from trip_agents.schemas import Activity, DayPlan, HardConstraints, Itinerary, TripRequest

from .conftest import TODAY


def test_dataset_is_deterministic_stratified_and_labelled() -> None:
    a, b = build_dataset(60, seed=7), build_dataset(60, seed=7)
    assert [s.model_dump() for s in a] == [s.model_dump() for s in b]
    assert Counter(s.difficulty for s in a) == {"easy": 21, "medium": 24, "hard": 15}
    infeasible = [s for s in a if not s.feasible]
    assert infeasible and all(s.gold.hard.budget_total < s.min_feasible_cost for s in infeasible)
    assert all(s.gold.hard.budget_total >= s.min_feasible_cost for s in a if s.feasible)
    assert all(("search_hotels" in s.expected_tools) == (s.gold.hard.days > 1) for s in a)
    dev, test = split(a, 0.3, seed=7)
    assert not {s.id for s in dev} & {s.id for s in test} and len(dev) + len(test) == 60
    assert {s.difficulty for s in dev} == {"easy", "medium", "hard"}


def _gold() -> TripRequest:
    return TripRequest(hard=HardConstraints(destination="杭州", start_date=date(2026, 12, 1), days=1, budget_total=2000, travelers=1,
                                            dietary_restrictions=["vegetarian"], must_visit=["灵隐寺"]))  # fmt: skip


def test_world_eval_resolves_names_and_counts_hallucinations() -> None:
    w = world_bundle("杭州")
    veg = next(r for r in w.restaurants if "vegetarian" in r.dietary_tags)
    it = Itinerary(hotel_id=None, days=[DayPlan(day_index=0, date=date(2026, 12, 1), activities=[
        Activity(start_time=time(9), end_time=time(11), kind="attraction", title="灵隐寺", ref_id="灵隐寺", est_cost=75),
        Activity(start_time=time(12), end_time=time(13), kind="meal", title="午餐", ref_id=veg.name, est_cost=50),
        Activity(start_time=time(14), end_time=time(15), kind="attraction", title="不存在的景点", ref_id="杭州火星基地", est_cost=0),
    ])])  # fmt: skip
    norm, total, unresolved = normalize_refs(it, w)
    assert norm.days[0].activities[0].ref_id == "hz-a01" and norm.days[0].activities[1].ref_id == veg.id
    assert (total, unresolved) == (3, 1)
    ev = evaluate_plan(it, _gold())
    assert ev.structure_ok and ev.hallucinated_refs == 1 and "unknown_ref" in ev.errors and not ev.hard_pass
    assert not evaluate_plan(None, _gold()).structure_ok


def test_tracer_captures_llm_tools_nodes_and_retries() -> None:
    model = MockChatModel(faults={"bad_json_once", "bad_tool_args_once"})
    g = build_graph(AgentDeps(model=model, tools=LocalToolExecutor(), today=TODAY), GraphOptions(human_in_loop=False))
    t = TraceRecorder()
    g.invoke({"messages": [HumanMessage(content="11月20日去杭州玩2天，预算3000")]}, {"callbacks": [t]})
    tr = t.finish()
    assert len(tr.llm_calls) == model._calls  # every model call recorded exactly once
    assert tr.input_tokens > 0 and all(c.model == "mock-planner-v1" for c in tr.llm_calls)
    assert {c.node for c in tr.llm_calls} >= {"chat", "research.agent", "itinerary"}
    assert tr.parse_failures == 1 and tr.skipped == 0
    assert sum(not x.ok for x in tr.tool_calls) == 1
    top = {n.node for n in tr.nodes if n.parent is None}
    assert {"chat", "research", "itinerary", "budget", "validate", "finalize"} <= top
    assert tr.cost_usd == 0.0


def test_run_one_all_systems() -> None:
    sample = build_dataset(10, seed=1)[5]
    for name, cfg in SYSTEMS.items():
        r = run_one(sample, cfg, lambda: MockChatModel(), judge_model=MockChatModel())
        assert r.completed, (name, r.error)
        assert r.structure_ok and r.judge_total is not None
        assert (r.tool_calls > 0) == (name in ("full", "no_revision"))
        assert (r.llm_calls == 1) == (name == "baseline")


def test_runner_writes_report_and_resumes(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import trip_agents.evals.run as runmod

    monkeypatch.setattr(runmod, "RUNS_DIR", tmp_path)
    args = ["--limit", "6", "--repeats", "2", "--systems", "full,baseline", "--name", "t", "--provider", "mock"]
    out = run_main(args)
    lines = (out / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6 * 2 * 2
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "MOCK 模式" in report and "硬约束通过率" in report and "full" in report
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["mock"] and meta["n_samples"] == 6 and meta["repeats"] == 2
    run_main(args)  # resume: nothing new appended
    assert len((out / "records.jsonl").read_text(encoding="utf-8").splitlines()) == len(lines)
    assert EVAL_TODAY.year == 2026
