"""Score an existing run's plans with a (different) judge, without regenerating them.

    uv run python -m trip_agents.evals.rejudge evals/runs/<name> --judge-provider openai --judge-model <m>

Plans come from `RunRecord.itinerary`; records written before that field existed cannot be
re-judged and are counted as skipped. The previous records.jsonl is kept as records.<old judge>.jsonl,
meta.json gets the new judge, and summary.json / report.md are rebuilt.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trip_agents.evals.dataset import DATASET_DIR, load
from trip_agents.evals.judge import judge_plan
from trip_agents.evals.report import write_report
from trip_agents.evals.systems import RunRecord
from trip_agents.llm.factory import DEFAULT_MODELS, get_chat_model
from trip_agents.schemas import Itinerary


def rejudge(run_dir: Path, provider: str, model: str | None, *, workers: int = 4) -> tuple[int, int]:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    rec_path = run_dir / "records.jsonl"
    records = [RunRecord.model_validate_json(x) for x in rec_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    gold = {s.id: s.gold for s in load(DATASET_DIR / f"{meta['split']}.jsonl")}
    judge = get_chat_model(provider, model, temperature=0.0)  # type: ignore[arg-type]
    model_name = model or DEFAULT_MODELS[provider]

    def score(r: RunRecord) -> bool:
        r.judge = r.judge_total = None
        if r.itinerary is None or not r.structure_ok:
            return r.itinerary is not None  # unscorable plan: judged as "nothing", like run_one does
        jr = judge_plan(judge, Itinerary.model_validate(r.itinerary), gold[r.sample_id], r.violations)
        if jr.scores:
            r.judge, r.judge_total = jr.scores.model_dump(), jr.total
        return True

    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = sum(pool.map(score, records))

    old = meta.get("judge") or {"model": "none"}
    shutil.copy(rec_path, run_dir / f"records.{re.sub(r'[^A-Za-z0-9.-]+', '_', old['model'])}.jsonl")
    rec_path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    meta["judge"] = {"provider": provider, "model": model_name}
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(run_dir, meta, records)
    return done, len(records) - done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--judge-provider", required=True)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    done, skipped = rejudge(args.run_dir, args.judge_provider, args.judge_model, workers=args.workers)
    print(f"re-judged {done} records, skipped {skipped} (no stored plan); report: {args.run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
