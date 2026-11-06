"""Calibrate judge weights against human scores.

1. Draw ~30 plans to label. Only plans that pass every hard constraint are eligible: the
   judge never scores the others, so labelling them would calibrate nothing.
       uv run python -m trip_agents.evals.calibrate <run> human_scores.csv --make-template 30
   then fill in the ``human`` column (0-100 overall) yourself.
2. uv run python -m trip_agents.evals.calibrate <run> human_scores.csv

Grid-searches weights on the simplex (step 0.05) to maximise Spearman correlation
between the weighted judge total and the human score, and reports the default
weights' correlation for comparison. Small-sample results are noisy: treat the
output as a sanity check, and keep the weights only if the gain is clear.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from trip_agents.evals.judge import DEFAULT_WEIGHTS
from trip_agents.evals.systems import RunRecord

DIMS = ("richness", "preference_fit", "reasonableness")


def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):  # average ranks for ties
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(_rank(a), _rank(b))


def simplex(step: float = 0.05) -> list[dict[str, float]]:
    n = round(1 / step)
    return [{"richness": i / n, "preference_fit": j / n, "reasonableness": (n - i - j) / n} for i in range(n + 1) for j in range(n + 1 - i)]


def fit(pairs: list[tuple[dict[str, float], float]], step: float = 0.05) -> tuple[dict[str, float], float]:
    human = [h for _, h in pairs]
    best_w, best = DEFAULT_WEIGHTS, -2.0
    for w in simplex(step):
        pred = [sum(j[d] * w[d] for d in DIMS) for j, _ in pairs]
        r = spearman(pred, human)
        if r > best + 1e-9:
            best_w, best = w, r
    return best_w, best


def make_template(recs: list[RunRecord], path: Path, n: int, seed: int = 0) -> int:
    """Stratified (system x difficulty) draw from hard-passing, judged plans; human column left empty."""
    pool: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for r in recs:
        if r.hard_pass and r.judge is not None:
            pool[(r.system, r.difficulty)].append(r)
    rng = random.Random(seed)
    for v in pool.values():
        rng.shuffle(v)
    picked: list[RunRecord] = []
    while len(picked) < n and any(pool.values()):
        for key in sorted(pool):
            if pool[key] and len(picked) < n:
                picked.append(pool[key].pop())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "system", "rep", "difficulty", "human"])
        w.writerows([r.sample_id, r.system, r.rep, r.difficulty, ""] for r in picked)
    return len(picked)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("human_csv", type=Path)
    ap.add_argument("--make-template", type=int, default=0, metavar="N", help="write N hard-passing plans to label, then exit")
    args = ap.parse_args()
    recs: dict[tuple[str, str, int], RunRecord] = {}
    for line in (args.run / "records.jsonl").read_text(encoding="utf-8").splitlines():
        rec = RunRecord.model_validate_json(line)
        recs[(rec.sample_id, rec.system, rec.rep)] = rec
    if args.make_template:
        k = make_template(list(recs.values()), args.human_csv, args.make_template)
        print(f"wrote {k} hard-passing plans to {args.human_csv}; fill in the human column")
        return
    pairs = []
    skipped = 0
    with args.human_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            found = recs.get((row["sample_id"], row["system"], int(row.get("rep") or 0)))
            if not (row.get("human") or "").strip():
                continue
            if found is None or not found.hard_pass or found.judge is None:  # labels on failing plans are not used
                skipped += 1
                continue
            pairs.append(({d: float(found.judge[d]) for d in DIMS}, float(row["human"])))
    if len(pairs) < 5:
        raise SystemExit(f"need at least 5 matched rows with judge scores, got {len(pairs)}")
    human = [h for _, h in pairs]
    default_r = spearman([sum(j[d] * DEFAULT_WEIGHTS[d] for d in DIMS) for j, _ in pairs], human)
    w, corr = fit(pairs)
    print(
        json.dumps(
            {
                "n": len(pairs),
                "skipped_not_hard_pass_or_unjudged": skipped,
                "default_weights": DEFAULT_WEIGHTS,
                "default_spearman": round(default_r, 3),
                "fitted_weights": w,
                "fitted_spearman": round(corr, 3),
            },
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
