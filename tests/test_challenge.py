"""Challenge-set tooling: stratified selection, review application, template-vs-challenge report."""

from __future__ import annotations

import csv
from pathlib import Path

from trip_agents.evals.challenge import REVIEW_HEADER, apply_review, compare, select, write_review
from trip_agents.evals.dataset import DATASET_DIR, load, save
from trip_agents.evals.systems import RunRecord, extraction_accuracy

TEST = load(DATASET_DIR / "test.jsonl")


def test_select_is_stratified_and_reproducible() -> None:
    a, b = select(TEST), select(TEST)
    assert [s.id for s in a] == [s.id for s in b] and len(a) == 36
    for d in ("easy", "medium", "hard"):
        assert sum(s.difficulty == d for s in a) == 12
    assert 2 <= sum(not s.feasible for s in a) <= 3


def test_apply_review_edits_and_drops(tmp_path: Path) -> None:
    picked = [s.model_copy(update={"rewrite_info": {"template_text": s.request_text}}) for s in select(TEST)[:3]]
    path = tmp_path / "review.csv"
    write_review(picked, path)
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    assert list(rows[0].keys()) == REVIEW_HEADER
    rows[0]["decision(ok/edit/drop)"], rows[0]["edited_text"] = "edit", "改过的文本"
    rows[1]["decision(ok/edit/drop)"] = "drop"
    rows[2]["decision(ok/edit/drop)"] = "ok"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_HEADER)
        w.writeheader()
        w.writerows(rows)
    out, tally = apply_review(picked, path)
    assert [s.id for s in out] == [picked[0].id, picked[2].id] and out[0].request_text == "改过的文本"
    assert tally == {"edit": 1, "drop": 1, "ok": 1} and all(s.rewrite_info and s.rewrite_info["reviewed"] for s in out)


def _rec(run: Path, recs: list[RunRecord]) -> None:
    run.mkdir(parents=True)
    (run / "records.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in recs), encoding="utf-8")


def test_compare_pairs_same_ids_and_lists_extraction_errors(tmp_path: Path) -> None:
    samples = [s for s in select(TEST) if s.feasible][:2]
    ds = tmp_path / "challenge.jsonl"
    save(samples, ds)
    base = {"rep": 0, "completed": True, "structure_ok": True}
    ok = dict.fromkeys(["destination", "start_date", "days", "budget_total", "travelers", "dietary_restrictions", "must_visit"], True)
    bad = ok | {"travelers": False}
    t = [RunRecord(sample_id=s.id, system="full", difficulty=s.difficulty, feasible_gold=True, hard_pass=True, extraction=ok, **base)
         for s in samples]  # fmt: skip
    c = [
        RunRecord(sample_id=samples[0].id, system="full", difficulty=samples[0].difficulty, feasible_gold=True, hard_pass=False,
                  violations=["over_budget"], extraction=bad, extracted={"travelers": 1}, **base),
        RunRecord(sample_id=samples[1].id, system="full", difficulty=samples[1].difficulty, feasible_gold=True, hard_pass=True,
                  extraction=ok, **base),
    ]  # fmt: skip
    t.append(RunRecord(sample_id="not-in-challenge", system="full", difficulty="easy", feasible_gold=True, hard_pass=True, **base))
    _rec(tmp_path / "tmpl", t)
    _rec(tmp_path / "chal", c)
    md = compare(tmp_path / "tmpl", tmp_path / "chal", ds, systems=("full",))
    assert "配对样本 2 个" in md and "100.0%（2/2） | 50.0%（1/2）" in md
    assert f"`{samples[0].id}`" in md and "| 人数 |" in md and "over_budget" in md


def test_must_visit_extraction_uses_the_lenient_rule() -> None:
    g = TEST[0].gold.model_copy(deep=True)
    g.hard.must_visit = ["秦始皇帝陵博物院(兵马俑)", "故宫博物院"]
    p = g.model_copy(deep=True)
    p.hard.must_visit = ["兵马俑", "故宫"]  # alias in parentheses + contiguous short name
    assert extraction_accuracy(p, g)["must_visit"]
    p.hard.must_visit = ["兵马俑"]  # one missing
    assert not extraction_accuracy(p, g)["must_visit"]
    p.hard.must_visit = ["兵马俑", "熊猫基地"]  # nickname that is neither
    assert not extraction_accuracy(p, g)["must_visit"]
