"""Paired analysis: flips, bootstrap CI, failure attribution, plan size -- and the test-set reference values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from trip_agents.evals.analyze import analyze
from trip_agents.evals.metrics import (
    attraction_count,
    bootstrap_ci,
    failure_attribution,
    paired_judge_diffs,
    pass_flips,
    passing_attractions,
)
from trip_agents.evals.report import by_system
from trip_agents.evals.systems import RunRecord

RUN = Path(__file__).resolve().parents[1] / "evals" / "results" / "gemini_test_v1"


def rec(**kw: Any) -> RunRecord:
    base: dict[str, Any] = {"sample_id": "s", "system": "full", "rep": 0, "difficulty": "hard", "feasible_gold": True, "completed": True}
    return RunRecord(**(base | kw))


def test_pass_flips_counts_feasible_pairs_only() -> None:
    a = [rec(sample_id="1", hard_pass=True), rec(sample_id="2", hard_pass=False), rec(sample_id="3", hard_pass=True, feasible_gold=False)]
    b = [rec(sample_id="1", system="b", hard_pass=False), rec(sample_id="2", system="b", hard_pass=True),
         rec(sample_id="3", system="b", hard_pass=False, feasible_gold=False)]  # fmt: skip
    assert pass_flips(a, b) == (1, 1, 2)


def test_bootstrap_is_reproducible_and_brackets_the_mean() -> None:
    d = [1.0, 2.0, 3.0, 4.0, 10.0]
    ci = bootstrap_ci(d)
    assert ci == bootstrap_ci(d) and ci is not None and ci[0] <= 4.0 <= ci[1]
    assert bootstrap_ci([1.0]) is None


def test_failure_attribution_and_attractions() -> None:
    days = [{"activities": [{"kind": "attraction"}, {"kind": "meal"}, {"kind": "attraction"}]}]
    rs = [rec(sample_id="1", flagged_infeasible=True), rec(sample_id="2", revisions=2), rec(sample_id="3", revisions=3),
          rec(sample_id="4", revisions=0), rec(sample_id="5", hard_pass=True, itinerary={"days": days})]  # fmt: skip
    assert failure_attribution(rs) == {"误判不可行": 1, "回退耗尽": 2, "其他": 1}
    assert attraction_count(rs[-1]) == 2


@pytest.mark.skipif(not (RUN / "records.jsonl").exists(), reason="gemini_test_v1 not present")
def test_reference_values_on_the_test_run() -> None:
    import json

    records = [RunRecord.model_validate(json.loads(x)) for x in (RUN / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    g = by_system(records)
    assert pass_flips(g["full"], g["no_revision"])[:2] == (22, 0)
    assert pass_flips(g["full"], g["baseline_catalog"])[:2] == (46, 1)
    d = paired_judge_diffs(g["full"], g["baseline_catalog"])
    assert d is not None and d["n"] == 77 and d["delta"] == pytest.approx(2.9, abs=0.05)
    assert d["ci"][0] == pytest.approx(1.6, abs=0.1) and d["ci"][1] == pytest.approx(4.2, abs=0.1)
    assert d["sub"]["richness"] == pytest.approx(4.0, abs=0.1)
    assert failure_attribution(g["full"]) == {"误判不可行": 5, "回退耗尽": 3, "其他": 0}
    pa = passing_attractions(g["full"], g["baseline_catalog"])
    assert pa["a_paired"] == pytest.approx(7.3, abs=0.1) and pa["b_paired"] == pytest.approx(8.2, abs=0.1)
    text = analyze(RUN)
    assert "配对翻转" in text and "失败归因" in text and "95% CI" in text
