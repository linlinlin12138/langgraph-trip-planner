"""Run the evaluation.  One command, resumable, writes records + report.

    uv run python -m trip_agents.evals.run                                  # mock, dev split, all systems
    uv run python -m trip_agents.evals.run --split test --repeats 3 --provider openai --model <m> --judge-provider anthropic

Output: <dir>/<name>/{meta.json, records.jsonl, summary.json, report.md}, where <dir> is
evals/runs/ for mock runs (git-ignored, cheap to redo) and evals/results/ for real models
(tracked in git: generation costs money, and rejudge needs the stored plans).
Re-running with the same --name skips (sample, system, rep) triples already in records.jsonl.
To use a judge from another provider, generate with --no-judge and score later with
`python -m trip_agents.evals.rejudge evals/results/<name> --judge-provider openai --judge-model <m>`.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from trip_agents.config import PROJECT_ROOT, get_settings
from trip_agents.evals.dataset import DATASET_DIR, load
from trip_agents.evals.report import write_report
from trip_agents.evals.systems import SYSTEMS, RunRecord, run_one
from trip_agents.llm.factory import DEFAULT_MODELS, get_chat_model

# Errors that will not go away by retrying -- billing / quota / credentials. When one shows up the run
# stops instead of burning through the remaining samples as error records (gemini_test_v1 hit the
# Gemini monthly spending cap after 40 of 420 runs and wrote 380 error records). Plain per-minute rate
# limits are not here: the SDKs back off and retry those.
FATAL_ERROR_MARKERS = (
    "spending cap",  # Gemini: project monthly spend cap
    "spend cap",
    "insufficient_quota",  # OpenAI: no credit left
    "exceeded your current quota",
    "credit balance is too low",  # Anthropic
    "API_KEY_INVALID",
    "API key not valid",
    "invalid_api_key",
    "PERMISSION_DENIED",
)


def fatal_error(error: str | None) -> str | None:
    """The marker if ``error`` is one that retrying can't fix, else None."""
    if not error:
        return None
    return next((m for m in FATAL_ERROR_MARKERS if m in error), None)


RUNS_DIR = PROJECT_ROOT / "evals" / "runs"  # mock runs, git-ignored
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"  # real-model runs, committed


def run_dir(provider: str, name: str) -> Path:
    return (RUNS_DIR if provider == "mock" else RESULTS_DIR) / name


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT, check=True)
        # "dirty" = uncommitted changes to code or data that affect results (reports in docs/ don't count)
        paths = ["src", "react_from_scratch", "app", "data", "evals/datasets", "pyproject.toml", "uv.lock"]
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", *paths], capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def model_factory(provider: str, model: str | None, seed: int) -> Callable[[], BaseChatModel]:
    def make() -> BaseChatModel:
        kwargs: dict[str, Any] = {"seed": seed} if provider in ("openai", "gemini") else {}
        return get_chat_model(provider, model, **kwargs)  # type: ignore[arg-type]

    return make


def main(argv: list[str] | None = None) -> Path:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--dataset", type=Path, default=None, help="override dataset path")
    ap.add_argument("--systems", default="full,no_revision,baseline_catalog")  # baseline: opt-in
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="first N samples only (0 = all)")
    ap.add_argument("--difficulty", default=None, help="only these difficulties, e.g. hard or medium,hard")
    ap.add_argument("--ids", type=Path, default=None, help="JSON list of sample ids to run (e.g. a stability subset)")
    ap.add_argument("--provider", default=s.llm_provider)
    ap.add_argument("--model", default=s.llm_model or None)
    ap.add_argument("--judge-provider", default=None, help="default: same as --provider")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--name", default=None)
    ap.add_argument("--retry-errors", action="store_true", help="on resume, re-run records that ended in an exception (e.g. 504)")
    args = ap.parse_args(argv)

    if args.split == "test" and args.provider == "mock":
        print("note: test split with the mock model -- fine for pipeline checks, but report numbers are not model quality.")
    dataset_path = args.dataset or DATASET_DIR / f"{args.split}.jsonl"
    samples = load(dataset_path)
    if args.ids:
        wanted = set(json.loads(args.ids.read_text(encoding="utf-8")))
        samples = [x for x in samples if x.id in wanted]
    if args.difficulty:
        keep = set(args.difficulty.split(","))
        samples = [x for x in samples if x.difficulty in keep]
    if args.limit and args.limit < len(samples):  # seeded sample keeps the difficulty mix
        samples = sorted(random.Random(args.seed).sample(samples, args.limit), key=lambda x: x.id)
    systems = [SYSTEMS[x] for x in args.systems.split(",")]
    name = args.name or f"{datetime.now():%Y%m%d-%H%M%S}_{args.provider}_{args.split}"
    out = run_dir(args.provider, name)
    out.mkdir(parents=True, exist_ok=True)

    model_name = args.model or DEFAULT_MODELS[args.provider]
    judge_provider = args.judge_provider or args.provider
    judge_name = args.judge_model or DEFAULT_MODELS[judge_provider]
    meta = {
        "name": name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "dataset": str(dataset_path.relative_to(PROJECT_ROOT) if dataset_path.is_relative_to(PROJECT_ROOT) else dataset_path),
        "split": args.split,
        "n_samples": len(samples),
        "difficulty_filter": args.difficulty,
        "ids_file": str(args.ids) if args.ids else None,
        "systems": [x.name for x in systems],
        "repeats": args.repeats,
        "seed": args.seed,
        "provider": args.provider,
        "model": model_name,
        "thinking_level": s.llm_thinking_level if args.provider == "gemini" else None,
        "tool_backend": s.tool_backend,
        "judge": None if args.no_judge else {"provider": judge_provider, "model": judge_name},
        "mock": args.provider == "mock",
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    rec_path = out / "records.jsonl"
    done = set()
    if rec_path.exists():
        kept = []
        for line in rec_path.read_text(encoding="utf-8").splitlines():
            r = RunRecord.model_validate_json(line)
            if args.retry_errors and r.error:
                continue  # dropped: will be run again
            kept.append(line)
            done.add((r.sample_id, r.system, r.rep))
        if args.retry_errors:
            rec_path.write_text("".join(x + "\n" for x in kept), encoding="utf-8")
    jobs = [(smp, sys, rep) for rep in range(args.repeats) for sys in systems for smp in samples if (smp.id, sys.name, rep) not in done]
    print(f"{name}: {len(jobs)} runs to do ({len(done)} already done)")

    judge = None if args.no_judge else get_chat_model(judge_provider, args.judge_model, temperature=0.0)
    lock = threading.Lock()

    def work(job: tuple[Any, Any, int]) -> RunRecord:
        smp, sys, rep = job
        return run_one(smp, sys, model_factory(args.provider, args.model, args.seed + rep), rep=rep, judge_model=judge)

    stopped: str | None = None
    with rec_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.cancelled():
                continue
            rec = fut.result()
            marker = fatal_error(rec.error)
            if marker is not None:  # not written: a resume runs this sample again
                if stopped is None:
                    stopped = marker
                    for other in futures:
                        other.cancel()  # queued jobs; the ones already running finish and are kept
                continue
            with lock:
                f.write(rec.model_dump_json() + "\n")
                f.flush()
            if i % 20 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)}")

    records = [RunRecord.model_validate_json(line) for line in rec_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = write_report(out, meta, records)
    print(f"report: {report}")
    if stopped is not None:
        left = len(jobs) - sum(1 for r in records if (r.sample_id, r.system, r.rep) not in done)
        raise SystemExit(
            f"stopped: a non-retryable provider error ({stopped!r}) -- fix billing / quota / key, then run the same "
            f"command again to resume ({left} runs left)"
        )
    return out


if __name__ == "__main__":
    main()
