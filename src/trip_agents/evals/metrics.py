"""Metric definitions. Pure functions over RunRecords -- easy to unit test.

Four families:
  outcome     structure_ok, hard_pass (feasible samples only), hallucination rate,
              extraction accuracy, infeasible detection, judge score (hard-passing plans only)
  process     tool precision / recall, invalid & duplicate call rate, research steps, revisions
  efficiency  tokens, estimated cost, end-to-end latency (mean / P95), per-node latency
  stability   across repeats of the same sample: outcome agreement, plan Jaccard, judge std

Repeats: each family is computed per repeat, then reported as mean ± std over repeats.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from itertools import combinations

from trip_agents.evals.systems import RunRecord

Metric = Callable[[Sequence[RunRecord]], float | None]


# ----------------------------------------------------------------------------- helpers
def mean(xs: Iterable[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def percentile(xs: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile (p in 0..100)."""
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, math.ceil(p / 100 * len(s)) - 1)
    return s[k]


def rate(records: Sequence[RunRecord], pred: Callable[[RunRecord], bool | None]) -> float | None:
    vals = [pred(r) for r in records]
    vals2 = [v for v in vals if v is not None]
    return sum(vals2) / len(vals2) if vals2 else None


def tool_precision_recall(used: Sequence[str], expected: Sequence[str]) -> tuple[float | None, float | None]:
    u, e = set(used), set(expected)
    precision = len(u & e) / len(u) if u else None
    recall = len(u & e) / len(e) if e else None
    return precision, recall


def jaccard(a: set[str], b: set[str]) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


# ----------------------------------------------------------------------------- metric table
def _uses_tools(r: RunRecord) -> bool:
    return r.system != "baseline"


METRICS: dict[str, tuple[str, Metric]] = {
    # outcome
    "completion_rate": ("运行完成率", lambda rs: rate(rs, lambda r: r.completed)),
    "structure_ok": ("结构正确率", lambda rs: rate(rs, lambda r: r.structure_ok)),
    "hard_pass": ("硬约束通过率(可行样本)", lambda rs: rate([r for r in rs if r.feasible_gold], lambda r: r.hard_pass)),
    "hallucination_rate": (
        "实体幻觉率",
        lambda rs: (sum(r.hallucinated_refs for r in rs) / n) if (n := sum(r.n_refs for r in rs)) else None,
    ),
    "extraction_acc": ("需求抽取字段准确率", lambda rs: mean(mean(r.extraction.values()) for r in rs if r.extraction is not None)),
    "infeasible_recall": ("不可行样本识别率", lambda rs: rate([r for r in rs if not r.feasible_gold], lambda r: r.flagged_infeasible)),
    "infeasible_false_alarm": ("可行样本误报率", lambda rs: rate([r for r in rs if r.feasible_gold], lambda r: r.flagged_infeasible)),
    # Judge scores exist only for plans that pass every hard constraint (see docs/EVAL.md); the mean is over those
    "judge_total": ("合格计划质量分（Judge）", lambda rs: mean(r.judge_total for r in rs)),
    "judge_richness": ("　丰富度", lambda rs: mean(r.judge["richness"] if r.judge else None for r in rs)),
    "judge_preference": ("　偏好符合度", lambda rs: mean(r.judge["preference_fit"] if r.judge else None for r in rs)),
    "judge_reasonable": ("　合理性", lambda rs: mean(r.judge["reasonableness"] if r.judge else None for r in rs)),
    # process
    "tool_precision": (
        "工具选择精确率",
        lambda rs: mean(tool_precision_recall(r.tools_used, r.expected_tools)[0] for r in rs if _uses_tools(r)),
    ),
    "tool_recall": (
        "工具选择召回率",
        lambda rs: mean(tool_precision_recall(r.tools_used, r.expected_tools)[1] for r in rs if _uses_tools(r)),
    ),
    "invalid_call_rate": (
        "无效调用率",
        lambda rs: (sum(r.invalid_tool_calls for r in rs) / n) if (n := sum(r.tool_calls for r in rs)) else None,
    ),
    "duplicate_call_rate": (
        "重复调用率",
        lambda rs: (sum(r.duplicate_tool_calls for r in rs) / n) if (n := sum(r.tool_calls for r in rs)) else None,
    ),
    "research_steps": ("Research 平均步数", lambda rs: mean(r.research_steps for r in rs if _uses_tools(r))),
    "revisions": ("平均回退次数", lambda rs: mean(r.revisions for r in rs)),
    "parse_failures": ("平均解析失败次数", lambda rs: mean(r.parse_failures for r in rs)),
    # efficiency
    "llm_calls": ("平均 LLM 调用次数", lambda rs: mean(r.llm_calls for r in rs)),
    "tokens": ("平均 token (输入+输出)", lambda rs: mean(r.input_tokens + r.output_tokens for r in rs)),
    "cost_usd": ("平均估算费用 (USD)", lambda rs: None if any(r.cost_usd is None for r in rs) else mean(r.cost_usd for r in rs)),
    "latency_mean_ms": ("端到端耗时均值 (ms)", lambda rs: mean(r.wall_ms for r in rs)),
    "latency_p95_ms": ("端到端耗时 P95 (ms)", lambda rs: percentile([r.wall_ms for r in rs], 95)),
}
FAMILIES = {
    "结果": [
        "completion_rate",
        "structure_ok",
        "hard_pass",
        "hallucination_rate",
        "extraction_acc",
        "infeasible_recall",
        "infeasible_false_alarm",
        "judge_total",
        "judge_richness",
        "judge_preference",
        "judge_reasonable",
    ],
    "过程": ["tool_precision", "tool_recall", "invalid_call_rate", "duplicate_call_rate", "research_steps", "revisions", "parse_failures"],
    "效率": ["llm_calls", "tokens", "cost_usd", "latency_mean_ms", "latency_p95_ms"],
}


def summarize(records: Sequence[RunRecord]) -> dict[str, dict[str, float | None]]:
    """{metric: {"mean": .., "std": .., "n_reps": ..}} -- metric computed per repeat, then aggregated."""
    by_rep: dict[int, list[RunRecord]] = defaultdict(list)
    for r in records:
        by_rep[r.rep].append(r)
    out: dict[str, dict[str, float | None]] = {}
    for key, (_, fn) in METRICS.items():
        vals = [v for rs in by_rep.values() if (v := fn(rs)) is not None]
        out[key] = {
            "mean": statistics.fmean(vals) if vals else None,
            "std": statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else None),
            "n_reps": float(len(vals)),
        }
    return out


def node_latency(records: Sequence[RunRecord]) -> dict[str, dict[str, float | None]]:
    per: dict[str, list[float]] = defaultdict(list)
    for r in records:
        for node, ms in r.node_ms.items():
            per[node].append(ms)
    return {n: {"mean": mean(v), "p95": percentile(v, 95)} for n, v in per.items()}


def stability(records: Sequence[RunRecord]) -> dict[str, float | None]:
    """Only meaningful with repeats >= 2: compares runs of the same (sample, system)."""
    groups: dict[str, list[RunRecord]] = defaultdict(list)
    for r in records:
        groups[r.sample_id].append(r)
    agree, jac, jstd = [], [], []
    for runs in groups.values():
        if len(runs) < 2:
            continue
        agree.append(float(len({r.hard_pass for r in runs}) == 1))
        sets = [{x for day in r.plan_signature for x in day} for r in runs]
        jac.append(statistics.fmean(jaccard(a, b) for a, b in combinations(sets, 2)))
        js = [r.judge_total for r in runs if r.judge_total is not None]
        if len(js) > 1:
            jstd.append(statistics.stdev(js))
    return {"outcome_agreement": mean(agree), "plan_jaccard": mean(jac), "judge_std": mean(jstd), "n_groups": float(len(agree))}


JUDGE_KEYS = ("judge_total", "judge_richness", "judge_preference", "judge_reasonable")


def judged(records: Sequence[RunRecord]) -> int:
    return sum(r.judge_total is not None for r in records)


def paired_judge(a: Sequence[RunRecord], b: Sequence[RunRecord]) -> tuple[float | None, float | None, int]:
    """Judge means of two systems over the (sample, rep) pairs where BOTH plans were judged.

    Comparing each system's mean over its own passing plans is biased: a weaker system passes
    mostly easy samples, whose plans score higher. Pairing removes that selection effect.
    """
    sa = {(r.sample_id, r.rep): r.judge_total for r in a if r.judge_total is not None}
    sb = {(r.sample_id, r.rep): r.judge_total for r in b if r.judge_total is not None}
    keys = sorted(sa.keys() & sb.keys())
    if not keys:
        return None, None, 0
    return statistics.fmean(sa[k] for k in keys), statistics.fmean(sb[k] for k in keys), len(keys)


def violation_counts(records: Sequence[RunRecord]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for r in records:
        for v in set(r.violations):
            c[v] += 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))
