"""Stability experiment: run the same samples again and compare the runs.

    # 1. pick 40 test samples (difficulty x feasibility strata, fixed seed), copy their first run
    uv run python -m trip_agents.evals.stability prepare --source evals/results/gemini_test_v1 --name gemini_test_stability
    # 2. two more runs of `full` on them (rep 0 is already there and is skipped)
    make eval ARGS="--split test --ids evals/results/gemini_test_stability/ids.json --systems full --repeats 3 \\
                    --name gemini_test_stability --provider gemini ..."
    # 3. judge only the new runs (the copied run keeps its scores)
    make eval-rejudge RUN=evals/results/gemini_test_stability ARGS="--judge-provider openai --judge-model gpt-6.1-sol --only-missing"

The report then has the stability section (outcome agreement, attraction-set Jaccard, judge std across
the 3 runs of each sample). The source run is only read, never modified. Rep 0 is a copy of the source
record; since each repeat uses seed = 42 + rep, rep 0 would reproduce the source run anyway.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from trip_agents.evals.dataset import DATASET_DIR, Sample, load
from trip_agents.evals.run import RESULTS_DIR
from trip_agents.evals.systems import RunRecord

SEED = 20261108


def stratified(samples: list[Sample], n: int, seed: int = SEED) -> list[Sample]:
    """Proportional allocation over (difficulty, feasible) strata, largest remainder, >= 1 per stratum."""
    strata: dict[tuple[str, bool], list[Sample]] = defaultdict(list)
    for s in samples:
        strata[(s.difficulty, s.feasible)].append(s)
    keys = sorted(strata)
    exact = {k: n * len(strata[k]) / len(samples) for k in keys}
    alloc = {k: max(1, int(exact[k])) for k in keys}
    for k in sorted(keys, key=lambda k: -(exact[k] - int(exact[k]))):
        if sum(alloc.values()) >= n:
            break
        alloc[k] += 1
    while sum(alloc.values()) > n:  # the ">= 1" floor can overshoot: take from the largest stratum
        k = max(keys, key=lambda k: alloc[k])
        alloc[k] -= 1
    rng = random.Random(seed)
    picked = [s for k in keys for s in rng.sample(strata[k], min(alloc[k], len(strata[k])))]
    return sorted(picked, key=lambda s: s.id)


def prepare(source: Path, name: str, n: int, system: str = "full") -> Path:
    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    picked = stratified(load(DATASET_DIR / f"{meta['split']}.jsonl"), n)
    ids = [s.id for s in picked]
    (out / "ids.json").write_text(json.dumps(ids, indent=1), encoding="utf-8")
    records = [RunRecord.model_validate_json(x) for x in (source / "records.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    first = [r for r in records if r.system == system and r.sample_id in set(ids) and r.rep == 0 and not r.error]
    rec_path = out / "records.jsonl"
    if not rec_path.exists():
        rec_path.write_text("".join(r.model_dump_json() + "\n" for r in first), encoding="utf-8")
    strata: dict[str, int] = defaultdict(int)
    for s in picked:
        strata[f"{s.difficulty}/{'feasible' if s.feasible else 'infeasible'}"] += 1
    (out / "NOTE.md").write_text(
        f"# Stability subset\n\n{len(ids)} {meta['split']} samples (seed {SEED}, strata {dict(sorted(strata.items()))}), "
        f"listed in `ids.json`. Rep 0 of `{system}` is copied from `{source.name}` (read only); reps 1-2 are new runs.\n",
        encoding="utf-8",
    )
    print(f"{len(ids)} samples {dict(sorted(strata.items()))}; copied {len(first)} rep-0 records from {source} -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--system", default="full")
    args = ap.parse_args()
    prepare(args.source, args.name, args.n, args.system)


if __name__ == "__main__":
    main()
