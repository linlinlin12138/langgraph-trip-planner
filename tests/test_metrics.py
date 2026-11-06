"""Metric maths checked against hand-computed values."""

from __future__ import annotations

from typing import Any

import pytest

from trip_agents.evals.calibrate import fit, spearman
from trip_agents.evals.judge import JudgeScores, weighted_total
from trip_agents.evals.metrics import METRICS, jaccard, percentile, stability, summarize, tool_precision_recall
from trip_agents.evals.systems import RunRecord


def rec(**kw: Any) -> RunRecord:
    base: dict[str, Any] = {"sample_id": "s0", "system": "full", "rep": 0, "difficulty": "easy", "feasible_gold": True, "completed": True}
    return RunRecord(**(base | kw))


def m(key: str, rs: list[RunRecord]) -> float | None:
    return METRICS[key][1](rs)


def test_percentile_nearest_rank() -> None:
    xs = [float(i) for i in range(1, 21)]  # 1..20
    assert percentile(xs, 95) == 19.0
    assert percentile(xs, 50) == 10.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([], 95) is None


def test_tool_precision_recall() -> None:
    assert tool_precision_recall(["a", "b", "c", "d"], ["a", "b", "x"]) == (0.5, 2 / 3)
    assert tool_precision_recall([], ["a"]) == (None, 0.0)


def test_hard_pass_only_counts_feasible_and_infeasible_detection() -> None:
    rs = [
        rec(sample_id="a", hard_pass=True, structure_ok=True),
        rec(sample_id="b", hard_pass=False),
        rec(sample_id="c", feasible_gold=False, hard_pass=False, flagged_infeasible=True),
        rec(sample_id="d", feasible_gold=False, flagged_infeasible=False),
        rec(sample_id="e", flagged_infeasible=True),
    ]
    assert m("hard_pass", rs) == pytest.approx(1 / 3)  # a,b,e feasible; only a passes
    assert m("infeasible_recall", rs) == 0.5
    assert m("infeasible_false_alarm", rs) == 1.0  # only e has a non-None flag among feasible


def test_rate_metrics_are_pooled_over_calls() -> None:
    rs = [rec(tool_calls=4, invalid_tool_calls=1, duplicate_tool_calls=0), rec(tool_calls=6, invalid_tool_calls=0, duplicate_tool_calls=3)]
    assert m("invalid_call_rate", rs) == 0.1 and m("duplicate_call_rate", rs) == 0.3
    hallu = [rec(n_refs=10, hallucinated_refs=1), rec(n_refs=0)]
    assert m("hallucination_rate", hallu) == 0.1


def test_tool_metrics_skip_systems_without_tools() -> None:
    rs = [rec(system="baseline", tools_used=[], expected_tools=["a"])]
    assert m("tool_recall", rs) is None


def test_cost_is_na_if_any_price_unknown() -> None:
    assert m("cost_usd", [rec(cost_usd=0.01), rec(cost_usd=None)]) is None
    assert m("cost_usd", [rec(cost_usd=0.01), rec(cost_usd=0.03)]) == pytest.approx(0.02)


def test_summarize_mean_std_over_repeats() -> None:
    # rep 0: 1/2 structure ok; rep 1: 2/2 -> mean 0.75, sample std of (0.5, 1.0) = 0.3536
    rs = [
        rec(rep=0, structure_ok=True),
        rec(rep=0, sample_id="s1"),
        rec(rep=1, structure_ok=True),
        rec(rep=1, sample_id="s1", structure_ok=True),
    ]
    s = summarize(rs)["structure_ok"]
    assert s["mean"] == pytest.approx(0.75) and s["std"] == pytest.approx(0.35355, abs=1e-4) and s["n_reps"] == 2


def test_stability() -> None:
    rs = [
        rec(sample_id="a", rep=0, hard_pass=True, plan_signature=[["x", "y"]], judge_total=80),
        rec(sample_id="a", rep=1, hard_pass=True, plan_signature=[["x", "z"]], judge_total=70),
        rec(sample_id="b", rep=0, hard_pass=True, plan_signature=[["q"]]),
        rec(sample_id="b", rep=1, hard_pass=False, plan_signature=[["q"]]),
    ]
    st = stability(rs)
    assert st["outcome_agreement"] == 0.5
    assert st["plan_jaccard"] == pytest.approx((1 / 3 + 1) / 2)
    assert st["judge_std"] == pytest.approx(7.0711, abs=1e-3)
    assert jaccard(set(), set()) == 1.0


def test_judge_weighted_total_and_calibration() -> None:
    s = JudgeScores(richness=60, preference_fit=90, reasonableness=80, rationale="")
    assert weighted_total(s) == pytest.approx(0.3 * 60 + 0.4 * 90 + 0.3 * 80)
    assert weighted_total(s, {"richness": 1, "preference_fit": 0, "reasonableness": 0}) == 60
    # human scores driven only by reasonableness -> fitted weight should put all mass there
    pairs = [
        ({"richness": r, "preference_fit": p, "reasonableness": q}, q)
        for r, p, q in [(90, 10, 20), (10, 90, 40), (50, 50, 60), (20, 30, 80), (70, 60, 95)]
    ]
    w, corr = fit(pairs)
    assert w["reasonableness"] == pytest.approx(1.0) and corr == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_paired_judge_uses_only_samples_both_systems_pass() -> None:
    from trip_agents.evals.metrics import paired_judge

    a = [rec(sample_id="easy", judge_total=80.0), rec(sample_id="hard", judge_total=60.0), rec(sample_id="x", judge_total=None)]
    b = [rec(sample_id="easy", system="b", judge_total=85.0), rec(sample_id="hard", system="b", judge_total=None)]
    # unpaired means would be 70 vs 85 (b only passed the easy one); paired: 80 vs 85 on 1 sample
    assert paired_judge(a, b) == (80.0, 85.0, 1)
    assert paired_judge(a, []) == (None, None, 0)


def test_judge_cell_shows_n_and_na() -> None:
    from trip_agents.evals.report import judge_cell

    assert judge_cell("judge_total", {"mean": 79.0, "std": 0.0, "n_reps": 1.0}, 5) == "79.00（n=5）"
    assert judge_cell("judge_total", {"mean": None, "std": None, "n_reps": 0.0}, 0) == "N/A（n=0）"


def test_calibration_template_only_hard_passing_plans(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import csv

    from trip_agents.evals.calibrate import make_template

    recs = [rec(sample_id=f"p{i}", hard_pass=True, judge={"richness": 1}, judge_total=1.0) for i in range(4)]
    recs += [rec(sample_id=f"f{i}", hard_pass=False) for i in range(4)]
    path = tmp_path / "h.csv"
    assert make_template(recs, path, 10) == 4
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert {r["sample_id"] for r in rows} == {"p0", "p1", "p2", "p3"} and all(r["human"] == "" for r in rows)


def test_estimate_audit_metrics() -> None:
    rs = [
        rec(sample_id="a", cost_model_refs=80.0, cost_catalog_refs=100.0, budget_flip=True),  # under-quoted by 20%
        rec(sample_id="b", cost_model_refs=105.0, cost_catalog_refs=100.0, budget_flip=False),
        rec(sample_id="c"),  # no audit (e.g. crashed)
    ]
    assert m("est_deviation", rs) == pytest.approx(-0.075)
    assert m("est_underreport", rs) == 1.0
    assert m("budget_flip", rs) == 1.0
    assert m("est_deviation", [rec()]) is None
