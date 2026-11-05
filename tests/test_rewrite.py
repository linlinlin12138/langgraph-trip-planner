"""LLM rewrite pipeline with fake rewriter/checker models (no network)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from trip_agents.evals.dataset import DATASET_DIR, load
from trip_agents.evals.rewrite import compare, rewrite_one, rule_errors, run
from trip_agents.schemas import RequestExtraction, TripRequest

SAMPLES = load(DATASET_DIR / "dev.jsonl")
WITH_MUST = next(s for s in SAMPLES if s.gold.hard.must_visit and s.gold.hard.travelers > 1)


def as_extraction(g: TripRequest, **override: object) -> RequestExtraction:
    h, s = g.hard, g.soft
    d = {
        "destination": h.destination, "start_date": h.start_date, "days": h.days, "budget_total": h.budget_total,
        "travelers": h.travelers, "dietary_restrictions": h.dietary_restrictions, "must_visit": h.must_visit,
        "pace": s.pace, "hotel_tier": s.hotel_tier, "interests": s.interests,
    }  # fmt: skip
    return RequestExtraction.model_validate(d | override)


def test_rule_gate_catches_softened_must_visit_vague_budget_and_relative_dates() -> None:
    g, name = WITH_MUST.gold, WITH_MUST.gold.hard.must_visit[0]
    assert rule_errors(WITH_MUST.request_text, g) == []  # the template text is clean
    assert any("可选" in e for e in rule_errors(f"有空的话去{name}，预算{g.hard.budget_total:.0f}", g))
    assert any("没有逐字出现" in e for e in rule_errors("随便逛逛", g))
    assert any("预算说法不精确" in e for e in rule_errors(f"一定要去{name}，预算大概六千", g))
    assert any("相对日期" in e for e in rule_errors(f"下周末去，一定要去{name}", g))


def test_compare_is_field_by_field() -> None:
    g = WITH_MUST.gold
    assert compare(as_extraction(g), g) == []
    errs = compare(as_extraction(g, budget_total=g.hard.budget_total / g.hard.travelers, must_visit=[]), g)
    assert any(e.startswith("budget_total") for e in errs) and any(e.startswith("must_visit") for e in errs)
    assert any("兴趣" in e for e in compare(as_extraction(g, interests=["shopping", "food", "nature", "museum"]), g))
    assert "缺少字段" in compare(as_extraction(g, budget_total=None), g)[0]


def test_retry_with_feedback_then_accept() -> None:
    g, prompts = WITH_MUST.gold, []
    checks = iter([as_extraction(g, days=g.hard.days + 1), as_extraction(g)])

    def rewrite(system: str, user: str) -> str:
        prompts.append(user)
        return WITH_MUST.request_text

    o = rewrite_one(WITH_MUST, rewrite, lambda t: next(checks))
    assert o.accepted and len(o.attempts) == 2
    assert "days 不一致" in prompts[1] and "上一版" in prompts[1]  # the concrete mismatch is fed back


def test_fallback_keeps_template_text(tmp_path: Path) -> None:
    subset = SAMPLES[:4]
    bad_id = subset[1].id
    by_text = {s.request_text: s for s in subset}

    def rewrite(system: str, user: str) -> str:
        sid = next(s.id for s in subset if s.gold.hard.destination in user and f"天数：{s.gold.hard.days}" in user and
                   f"{s.gold.hard.budget_total:.0f}" in user)  # fmt: skip
        return next(s.request_text for s in subset if s.id == sid)

    def check(text: str) -> RequestExtraction:
        s = by_text[text]
        return as_extraction(s.gold, days=s.gold.hard.days + 1) if s.id == bad_id else as_extraction(s.gold)

    info = {"rewriter": {"provider": "fake"}, "checker": {"provider": "fake2"}}
    meta = run({"dev": subset}, rewrite, check, info, tmp_path, workers=2, max_attempts=2)
    assert meta["accepted"] == 3 and meta["fallback_to_template"] == 1
    out = {s.id: s for s in load(tmp_path / "dev.jsonl")}
    assert out[bad_id].rewrite == "template" and out[bad_id].request_text == subset[1].request_text
    assert out[bad_id].rewrite_info and out[bad_id].rewrite_info["attempts"] == 2
    assert all(out[s.id].gold == s.gold for s in subset)  # the answer key is never modified
    rows = list(csv.reader((tmp_path / "spotcheck.csv").open(encoding="utf-8-sig")))
    assert rows[0][:2] == ["id", "difficulty"] and 1 < len(rows) <= 4
    assert json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))["n"] == 4

    def boom(system: str, user: str) -> str:
        raise AssertionError("resume must not call the rewriter again")

    assert run({"dev": subset}, boom, check, info, tmp_path)["accepted"] == 3

    other = {"rewriter": {"provider": "fake"}, "checker": {"provider": "other"}}
    try:
        run({"dev": subset}, boom, check, other, tmp_path)
    except SystemExit as e:
        assert "checker" in str(e)
    else:
        raise AssertionError("cached outcomes from another checker must not be reused")
