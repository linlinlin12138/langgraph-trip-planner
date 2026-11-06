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
    return TripRequest(
        hard=HardConstraints(
            destination="杭州",
            start_date=date(2026, 12, 1),
            days=1,
            budget_total=2000,
            travelers=1,
            dietary_restrictions=["no_spicy"],
            must_visit=["灵隐寺"],
        )
    )


def test_world_eval_resolves_names_and_counts_hallucinations() -> None:
    w = world_bundle("杭州")
    veg = next(r for r in w.restaurants if "no_spicy" in r.dietary_tags)
    it = Itinerary(
        hotel_id=None,
        days=[
            DayPlan(
                day_index=0,
                date=date(2026, 12, 1),
                activities=[
                    Activity(start_time=time(9), end_time=time(11), kind="attraction", title="灵隐寺", ref_id="灵隐寺", est_cost=75),
                    Activity(start_time=time(12), end_time=time(13), kind="meal", title="午餐", ref_id=veg.name, est_cost=50),
                    Activity(
                        start_time=time(14), end_time=time(15), kind="attraction", title="不存在的景点", ref_id="杭州火星基地", est_cost=0
                    ),
                ],
            )
        ],
    )
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
        assert r.structure_ok and (r.judge_total is not None) == r.hard_pass  # only hard-passing plans are judged
        assert (r.tool_calls > 0) == (cfg.kind == "graph")
        assert (r.llm_calls == 1) == (cfg.kind == "single_shot")


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


def test_rejudge_scores_stored_plans_with_another_judge(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import trip_agents.evals.run as runmod
    from trip_agents.evals.rejudge import rejudge
    from trip_agents.evals.systems import RunRecord

    monkeypatch.setattr(runmod, "RUNS_DIR", tmp_path)
    out = run_main(["--limit", "3", "--systems", "full,baseline", "--name", "rj", "--provider", "mock", "--no-judge"])
    recs = [RunRecord.model_validate_json(x) for x in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r.itinerary is not None and r.judge_total is None for r in recs)

    done, skipped = rejudge(out, "mock", None, workers=2)
    assert (done, skipped) == (6, 0)
    recs = [RunRecord.model_validate_json(x) for x in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all((r.judge_total is not None) == r.hard_pass for r in recs)  # judged iff hard constraints pass
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "合格计划质量分" in report and "（n=" in report and "配对比较" in report
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["judge"]["provider"] == "mock"
    assert (out / "records.none.jsonl").exists()  # previous records kept


def test_real_model_runs_go_to_tracked_results_dir() -> None:
    from trip_agents.evals import run as runmod

    assert runmod.run_dir("mock", "x") == runmod.RUNS_DIR / "x"
    assert runmod.run_dir("gemini", "x") == runmod.RESULTS_DIR / "x"
    assert runmod.RESULTS_DIR.parent == runmod.RUNS_DIR.parent and runmod.RESULTS_DIR.name == "results"


def test_days_matched_by_date_not_index_numbering() -> None:
    from trip_agents.evals.world import evaluate_plan

    g = _gold()
    day = DayPlan(day_index=1, date=g.hard.start_date, activities=[  # 1-based numbering, right date
        Activity(start_time=time(9), end_time=time(11), kind="attraction", title="灵隐寺", ref_id="灵隐寺", est_cost=75),
    ])  # fmt: skip
    ev = evaluate_plan(Itinerary(hotel_id=None, days=[day]), g)
    assert not {"missing_day", "wrong_date", "extra_day"} & set(ev.errors)
    wrong = day.model_copy(update={"day_index": 0, "date": g.hard.start_date.replace(day=2)})
    assert "wrong_date" in evaluate_plan(Itinerary(hotel_id=None, days=[wrong]), g).errors


def test_retry_errors_reruns_crashed_records(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import trip_agents.evals.run as runmod
    from trip_agents.evals.systems import RunRecord

    monkeypatch.setattr(runmod, "RUNS_DIR", tmp_path)
    args = ["--limit", "2", "--systems", "baseline", "--name", "re", "--provider", "mock", "--no-judge"]
    out = run_main(args)
    path = out / "records.jsonl"
    recs = [RunRecord.model_validate_json(x) for x in path.read_text(encoding="utf-8").splitlines()]
    recs[0].error = "GoogleAPIError: 504 DEADLINE_EXCEEDED"
    path.write_text("".join(r.model_dump_json() + "\n" for r in recs), encoding="utf-8")
    run_main(args)  # plain resume keeps the crashed record
    assert sum(1 for x in path.read_text(encoding="utf-8").splitlines() if "DEADLINE" in x) == 1
    run_main([*args, "--retry-errors"])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and not any("DEADLINE" in x for x in lines)


def test_baseline_catalog_single_call_with_destination_catalog() -> None:
    from langchain_core.messages import BaseMessage

    from trip_agents.evals.world import world_bundle
    from trip_agents.llm.context import extract_context

    sample = next(s for s in build_dataset(20, seed=3) if s.feasible and s.gold.hard.days > 1)
    seen: list[list[BaseMessage]] = []

    class Spy(MockChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kw):  # type: ignore[no-untyped-def]
            seen.append(list(messages))
            return super()._generate(messages, stop, run_manager, **kw)

    r = run_one(sample, SYSTEMS["baseline_catalog"], lambda: Spy())
    assert r.completed and r.llm_calls == 1 and len(seen) == 1 and r.tool_calls == 0
    assert r.extraction is None and r.flagged_infeasible is None  # n/a for single-shot systems
    ctx = extract_context(seen[0])
    world = world_bundle(sample.gold.hard.destination)
    assert {a["id"] for a in ctx["attractions"]} == {a.id for a in world.attractions}
    assert {h["id"] for h in ctx["hotels"]} == {h.id for h in world.hotels}
    assert {x["id"] for x in ctx["restaurants"]} == {x.id for x in world.restaurants}
    assert "start_date" not in json.dumps(ctx) and "budget_total" not in json.dumps(ctx)  # nothing else from gold
    assert ctx["request_text"] == sample.request_text
    # the plan references catalog ids, so the evaluation can judge it like any other plan
    assert r.structure_ok and r.n_refs > 0 and r.hallucinated_refs == 0
    assert r.plan_total_catalog is not None


def test_uses_tools_is_decided_by_system_kind() -> None:
    from trip_agents.evals.metrics import _uses_tools
    from trip_agents.evals.systems import RunRecord

    base = {"sample_id": "s", "rep": 0, "difficulty": "easy", "feasible_gold": True, "completed": True}
    assert _uses_tools(RunRecord(system="full", **base))
    assert not _uses_tools(RunRecord(system="baseline_catalog", **base))
    assert not _uses_tools(RunRecord(system="baseline", **base))


def test_difficulty_filter(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import trip_agents.evals.run as runmod
    from trip_agents.evals.systems import RunRecord

    monkeypatch.setattr(runmod, "RUNS_DIR", tmp_path)
    out = run_main(["--difficulty", "hard", "--systems", "baseline_catalog", "--name", "h", "--provider", "mock", "--no-judge"])
    recs = [RunRecord.model_validate_json(x) for x in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert recs and all(r.difficulty == "hard" for r in recs)
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["difficulty_filter"] == "hard"
