"""Connectivity check + cost estimate for a provider, before any expensive eval.

    uv run python scripts/check_provider.py --provider gemini --list-models
    uv run python scripts/check_provider.py --provider gemini --model <m> --samples 5

Steps (stops at the first failure):
  1. (gemini, --list-models) list models the key can call with generateContent
  2. structured output: one RequestExtraction call
  3. tool calling: one Research subgraph run
  4. N dev samples through `full` and `baseline`, measuring real tokens and latency
  5. extrapolate tokens (and cost, if PRICING_OVERRIDES has this model) for full eval runs
Nothing here is cached or written to the eval runs directory.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import HumanMessage, SystemMessage

from trip_agents.config import gemini_api_key, get_settings
from trip_agents.evals.dataset import DATASET_DIR, EVAL_TODAY, load
from trip_agents.evals.systems import SYSTEMS, run_one
from trip_agents.graph.deps import AgentDeps, LocalToolExecutor
from trip_agents.graph.nodes.research import build_research_subgraph
from trip_agents.graph.prompts import CHAT_SYSTEM, RESEARCH_SYSTEM
from trip_agents.llm.context import render_context
from trip_agents.llm.factory import DEFAULT_MODELS, get_chat_model
from trip_agents.llm.structured import invoke_structured
from trip_agents.observability.pricing import estimate_cost
from trip_agents.observability.tracer import TraceRecorder
from trip_agents.schemas import RequestExtraction


def list_gemini_models() -> None:
    from google import genai

    client = genai.Client(api_key=gemini_api_key())
    rows = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            rows.append((m.name, getattr(m, "display_name", ""), getattr(m, "output_token_limit", "")))
    for name, display, out_limit in sorted(rows):
        print(f"  {name:<45} {display}  (max output {out_limit})")


_HINTS = (
    ("NOT_FOUND", "model not available to this key -> pick one from `--list-models` and pass --model / set LLM_MODEL"),
    ("PERMISSION_DENIED", "key rejected or API not enabled for its project -> check GOOGLE_API_KEY in AI Studio"),
    ("API_KEY_INVALID", "key rejected -> check GOOGLE_API_KEY in AI Studio"),
    ("credits are depleted", "project billing: prepaid credits used up -> top up at https://ai.studio/projects"),
    ("RESOURCE_EXHAUSTED", "quota / rate limit (or billing) -> wait, lower concurrency, or check the project's quota"),
)


def explain(e: BaseException) -> str:
    msg = str(e)
    hint = next((h for key, h in _HINTS if key in msg), None)
    first = msg.splitlines()[0][:300] if msg else type(e).__name__
    return f"{type(e).__name__}: {first}" + (f"\n    hint: {hint}" if hint else "")


def main() -> None:
    try:
        _main()
    except KeyboardInterrupt:
        raise
    except Exception as e:  # provider errors: one actionable line instead of a 40-frame traceback
        raise SystemExit(f"FAILED  {explain(e)}") from None


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default=None)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()
    model_name = args.model or get_settings().llm_model or DEFAULT_MODELS[args.provider]
    if args.provider == "gemini" and not gemini_api_key():
        raise SystemExit("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set -- put it in .env or the environment")

    if args.list_models:
        if args.provider != "gemini":
            raise SystemExit("--list-models is implemented for gemini only")
        print("models with generateContent:")
        list_gemini_models()
        return

    def make():  # type: ignore[no-untyped-def]
        return get_chat_model(args.provider, model_name)  # type: ignore[arg-type]

    print(f"[1] provider={args.provider} model={model_name}")

    t0 = time.perf_counter()
    tr = TraceRecorder()
    ctx = render_context({"task": "extract_request", "today": EVAL_TODAY.isoformat(), "current": None})
    probe = [SystemMessage(content=CHAT_SYSTEM + "\n" + ctx), HumanMessage(content="我和女朋友12月5日去杭州玩3天，预算6000，不吃辣")]
    res = invoke_structured(make(), RequestExtraction, probe, config={"callbacks": [tr]})
    print(
        f"[2] structured output: ok={res.ok} attempts={res.attempts} {time.perf_counter() - t0:.1f}s "
        f"tokens={tr.trace.input_tokens}+{tr.trace.output_tokens}"
    )
    if not res.ok:
        print("    errors:", res.errors)
        raise SystemExit(1)
    print("    ->", res.value.model_dump(exclude_defaults=True) if res.value else None)

    samples = load(DATASET_DIR / "dev.jsonl")
    s0 = samples[0]
    tr = TraceRecorder()
    t0 = time.perf_counter()
    sub = build_research_subgraph(AgentDeps(model=make(), tools=LocalToolExecutor(), today=EVAL_TODAY))
    task = "请为以下需求检索信息：\n" + render_context({"task": "research", "request": s0.gold.model_dump(mode="json")})
    msgs = [SystemMessage(content=RESEARCH_SYSTEM.format(max_rounds=3)), HumanMessage(content=task)]
    out = sub.invoke({"messages": msgs}, {"callbacks": [tr], "recursion_limit": 40})
    recs = out.get("records", [])
    print(
        f"[3] tool calling: {len(recs)} calls in {out.get('step', 0)} steps, {time.perf_counter() - t0:.1f}s: "
        + ", ".join(f"{r['name']}{'' if r['ok'] else '(ERR)'}" for r in recs)
    )
    if not recs:
        raise SystemExit("model made no tool calls")

    picked = sorted(random.Random(0).sample(samples, min(args.samples, len(samples))), key=lambda s: s.id)
    per: dict[str, list[tuple[int, int, float]]] = {"full": [], "baseline": []}
    for s in picked:
        for name in per:
            r = run_one(s, SYSTEMS[name], make)
            per[name].append((r.input_tokens, r.output_tokens, r.wall_ms))
            err = ("ERR " + r.error[:80]) if r.error else ""
            print(
                f"[4] {s.id} {s.difficulty:<6} {name:<8} completed={r.completed} hard_pass={r.hard_pass} "
                f"tokens={r.input_tokens}+{r.output_tokens} {r.wall_ms / 1000:.1f}s {err}"
            )

    print("[5] per-sample averages and extrapolation (graph systems ~= full; baseline is one call):")
    for name, rows in per.items():
        ti = statistics.fmean(x[0] for x in rows)
        to = statistics.fmean(x[1] for x in rows)
        cost = estimate_cost(model_name, int(ti), int(to))
        print(
            f"    {name:<8} input {ti:.0f}  output {to:.0f}  latency {statistics.fmean(x[2] for x in rows) / 1000:.1f}s  "
            f"cost/sample {'n/a (set PRICING_OVERRIDES)' if cost is None else f'${cost:.5f}'}"
        )
    full_in, full_out = (statistics.fmean(x[i] for x in per["full"]) for i in (0, 1))
    base_in, base_out = (statistics.fmean(x[i] for x in per["baseline"]) for i in (0, 1))
    for split, n in (("dev", 60), ("test", 140)):
        for reps in (1, 3):
            tin = (3 * full_in + base_in) * n * reps  # full, no_revision, no_research ~ full (upper bound), + baseline
            tout = (3 * full_out + base_out) * n * reps
            cost = estimate_cost(model_name, int(tin), int(tout))
            print(
                f"    {split} x{reps} (4 systems, judge excluded): ~{tin / 1e6:.2f}M in + {tout / 1e6:.2f}M out tokens"
                f"{'' if cost is None else f'  ~${cost:.2f}'}"
            )


if __name__ == "__main__":
    main()
