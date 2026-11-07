"""Tooling for the stability and memory experiments (mock model, no network)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from trip_agents.evals.dataset import DATASET_DIR, load
from trip_agents.evals.stability import stratified


def test_stratified_subset_has_every_stratum_and_is_reproducible() -> None:
    test = load(DATASET_DIR / "test.jsonl")
    a, b = stratified(test, 40), stratified(test, 40)
    assert [s.id for s in a] == [s.id for s in b] and len(a) == 40
    assert set(Counter((s.difficulty, s.feasible) for s in a)) == set(Counter((s.difficulty, s.feasible) for s in test))


def test_ids_filter_and_rejudge_only_missing(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import trip_agents.evals.run as runmod
    from trip_agents.evals.rejudge import rejudge
    from trip_agents.evals.run import main as run_main
    from trip_agents.evals.systems import RunRecord

    monkeypatch.setattr(runmod, "RUNS_DIR", tmp_path)
    ids = [s.id for s in load(DATASET_DIR / "dev.jsonl")[:3]]
    (tmp_path / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    out = run_main(["--ids", str(tmp_path / "ids.json"), "--systems", "full", "--name", "st", "--provider", "mock", "--no-judge"])
    path = out / "records.jsonl"
    recs = [RunRecord.model_validate_json(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert sorted(r.sample_id for r in recs) == sorted(ids)
    recs[0].judge_total, recs[0].judge = 12.3, {"richness": 1, "preference_fit": 1, "reasonableness": 1}
    recs[0].hard_pass = True
    path.write_text("".join(r.model_dump_json() + "\n" for r in recs), encoding="utf-8")
    rejudge(out, "mock", None, only_missing=True)
    after = {r.sample_id: r for r in (RunRecord.model_validate_json(x) for x in path.read_text(encoding="utf-8").splitlines())}
    assert after[recs[0].sample_id].judge_total == 12.3  # kept
    assert all(r.judge_total is not None for r in after.values() if r.hard_pass)


def test_memory_eval_saves_rows_resumes_and_skips_session1_without_memory(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import sys

    import trip_agents.evals.memory_eval as mem

    monkeypatch.setattr(mem, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["memory_eval", "--n", "3", "--name", "m"])
    mem.main()
    rows = [json.loads(x) for x in (tmp_path / "m" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6 and (tmp_path / "m" / "report.md").exists()
    assert all(r["session1_tokens"] == 0 for r in rows if not r["memory"])  # session 1 not run without memory
    assert all(r["session1_tokens"] > 0 for r in rows if r["memory"])
    calls: list[Any] = []
    monkeypatch.setattr(mem, "run_scenario", lambda *a, **k: calls.append(a))
    mem.main()  # resume: nothing left to run
    assert calls == []


def test_memory_eval_records_errors_instead_of_crashing() -> None:
    import trip_agents.evals.memory_eval as mem

    sc = mem.build_scenarios(3, 7)[0]

    def boom() -> Any:
        raise RuntimeError("504 DEADLINE_EXCEEDED")

    row = mem.run_scenario(sc, boom, True)
    assert "504" in row["error"] and row["id"] == sc.id


def test_add_restriction_scenarios_and_both_error_directions() -> None:
    import trip_agents.evals.memory_eval as mem

    scs = mem.build_scenarios(40, 7)
    assert Counter(s.kind for s in scs) == dict.fromkeys(mem.KINDS, 10)
    adds = [s for s in scs if s.kind == "add_restriction"]
    for s in adds:
        old = set(s.gold1.hard.dietary_restrictions) if s.gold1 else set()
        new = set(s.gold2.hard.dietary_restrictions)
        assert old <= new  # nothing revoked: the remembered restriction stays in gold
        if new - old:
            assert any(p in s.s2_text for p in mem._DIET_PHRASE.values())
    row = mem.run_scenario(adds[0], lambda: __import__("trip_agents.llm.mock", fromlist=["MockChatModel"]).MockChatModel(), True)
    assert {"missed_new_restriction", "dropped_old_restriction", "stale_diet", "stale_pace"} <= row.keys()
    ov = next(s for s in scs if s.kind == "override")
    row2 = mem.run_scenario(ov, lambda: __import__("trip_agents.llm.mock", fromlist=["MockChatModel"]).MockChatModel(), True)
    assert row2["stale_diet"] is False and row2["stale_pace"] is False  # the mock applies "这次什么都吃 / 节奏" correctly
