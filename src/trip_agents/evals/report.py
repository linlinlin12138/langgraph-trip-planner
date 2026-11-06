"""Markdown report for one run, and a diff between two runs.

uv run python -m trip_agents.evals.report evals/runs/<A> evals/runs/<B>     # compare
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from trip_agents.evals.metrics import (
    FAMILIES,
    JUDGE_KEYS,
    METRICS,
    judged,
    node_latency,
    paired_judge,
    stability,
    summarize,
    violation_counts,
)
from trip_agents.evals.systems import SYSTEMS, RunRecord

PERCENT = {
    "completion_rate",
    "structure_ok",
    "hard_pass",
    "hallucination_rate",
    "extraction_acc",
    "infeasible_recall",
    "infeasible_false_alarm",
    "tool_precision",
    "tool_recall",
    "invalid_call_rate",
    "duplicate_call_rate",
}


def fmt(key: str, mean: float | None, std: float | None = None, n_reps: float = 1) -> str:
    if mean is None:
        return "n/a"
    if key in PERCENT:
        s = f"{mean * 100:.1f}%"
        return s + (f" ± {std * 100:.1f}" if std is not None and n_reps > 1 else "")
    if key == "cost_usd":
        s = f"{mean:.5f}"
    elif mean >= 100:
        s = f"{mean:.0f}"
    else:
        s = f"{mean:.2f}"
    return s + (f" ± {std:.2f}" if std is not None and n_reps > 1 else "")


def judge_cell(key: str, m: dict[str, float | None], n: int) -> str:
    """Quality score of hard-passing plans, with how many plans it is based on."""
    return "N/A（n=0）" if not n or m["mean"] is None else f"{fmt(key, **m)}（n={n}）"  # type: ignore[arg-type]


def by_system(records: list[RunRecord]) -> dict[str, list[RunRecord]]:
    g: dict[str, list[RunRecord]] = defaultdict(list)
    for r in records:
        g[r.system].append(r)
    order = list(SYSTEMS)
    return dict(sorted(g.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99))


def build_summary(records: list[RunRecord]) -> dict[str, Any]:
    groups = by_system(records)
    return {
        "metrics": {s: summarize(rs) for s, rs in groups.items()},
        "by_difficulty": {
            s: {d: summarize([r for r in rs if r.difficulty == d]) for d in ("easy", "medium", "hard")} for s, rs in groups.items()
        },
        "node_latency": {s: node_latency(rs) for s, rs in groups.items()},
        "stability": {s: stability(rs) for s, rs in groups.items()},
        "violations": {s: violation_counts(rs) for s, rs in groups.items()},
    }


def render(meta: dict[str, Any], summary: dict[str, Any], records: list[RunRecord]) -> str:
    systems = list(summary["metrics"])
    lines = [f"# 评测报告 · {meta['name']}", ""]
    if meta.get("mock"):
        lines += [
            "> ⚠️ **MOCK 模式**：模型为规则模拟的 `MockChatModel`，Judge 为启发式打分。以下数字只验证评测流程能跑通，",
            "> **不代表任何真实模型的规划质量**；mock 是确定性的，稳定性指标恒为 1。",
            "",
        ]
    judge = meta.get("judge")
    lines += [
        "| 项 | 值 |",
        "|---|---|",
        f"| 日期 | {meta['date']} |",
        f"| git commit | `{meta['git_commit']}` |",
        f"| 模型 | {meta['provider']} / `{meta['model']}` |",
        f"| Judge | {judge['provider'] + ' / `' + judge['model'] + '`' if judge else '未启用'} |",
        f"| 工具后端 | {meta['tool_backend']} |",
        f"| 数据集 | `{meta['dataset']}`（{meta['n_samples']} 条） |",
        f"| 每组运行次数 | {meta['repeats']}（seed={meta['seed']}，表中为各次均值 ± 标准差） |",
        f"| 记录数 | {len(records)} |",
        "",
    ]
    groups = by_system(records)
    judge_on = any(r.judge_total is not None for r in records)
    for fam, keys in FAMILIES.items():
        header = " | ".join(f"{s}（{SYSTEMS[s].description}）" if s in SYSTEMS else s for s in systems)
        lines += [f"## {fam}指标", "", f"| 指标 | {header} |", "|---|" + "---|" * len(systems)]
        for k in keys:
            if k in JUDGE_KEYS:
                if not judge_on:
                    continue
                cells = [judge_cell(k, summary["metrics"][s][k], judged(groups[s])) for s in systems]
            else:
                cells = [fmt(k, **summary["metrics"][s][k]) for s in systems]
            if all(c == "n/a" for c in cells):
                continue
            lines.append(f"| {METRICS[k][0]} | " + " | ".join(cells) + " |")
        if fam == "结果" and judge_on:
            lines += ["", "> 质量分只对硬约束全部通过的计划打分（n = 参与评分的计划数），不通过的计划不打分；"
                      "系统之间请看下面的配对比较。"]  # fmt: skip
        lines.append("")

    if judge_on and len(systems) > 1:
        lines += [
            "## 合格计划质量分：配对比较",
            "",
            "只取两个系统**都通过**硬约束的同一批样本比较，避免一方只在简单样本上通过造成的选择偏差。",
            "",
            "| 对比 (A vs B) | A | B | Δ (B−A) | 配对样本数 |",
            "|---|---|---|---|---|",
        ]
        for i, a in enumerate(systems):
            for b in systems[i + 1 :]:
                ma, mb, n = paired_judge(groups[a], groups[b])
                if not n:
                    lines.append(f"| {a} vs {b} | N/A | N/A | N/A | 0 |")
                else:
                    lines.append(f"| {a} vs {b} | {ma:.2f} | {mb:.2f} | {mb - ma:+.2f} | {n} |")  # type: ignore[operator]
        lines.append("")

    lines += [
        "## 稳定性（同一样本多次运行）",
        "",
        "| 系统 | 结果一致率 | 景点集合 Jaccard | Judge 分标准差 | 样本组数 |",
        "|---|---|---|---|---|",
    ]
    for s in systems:
        st = summary["stability"][s]
        if not st["n_groups"]:
            continue
        lines.append(
            f"| {s} | {fmt('structure_ok', st['outcome_agreement'])} | {fmt('x', st['plan_jaccard'])} | "
            f"{fmt('x', st['judge_std'])} | {st['n_groups']:.0f} |"
        )
    if all(not summary["stability"][s]["n_groups"] for s in systems):
        lines.append("| （repeats=1，未计算） | | | | |")
    lines.append("")

    lines += ["## 按难度", "", "| 系统 | 难度 | 硬约束通过率 | 合格计划质量分 | 平均 token |", "|---|---|---|---|---|"]
    for s in systems:
        for d, m in summary["by_difficulty"][s].items():
            if m["completion_rate"]["mean"] is None:
                continue
            nd = judged([r for r in groups[s] if r.difficulty == d])
            judge_txt = judge_cell("judge_total", m["judge_total"], nd) if judge_on else "n/a"
            cells = [fmt("hard_pass", **m["hard_pass"]), judge_txt, fmt("tokens", **m["tokens"])]
            lines.append(f"| {s} | {d} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## 各节点耗时（ms）", ""]
    for s in systems:
        nl = summary["node_latency"][s]
        if nl:
            lines.append(f"**{s}**：" + " · ".join(f"`{n}` {fmt('x', v['mean'])} / P95 {fmt('x', v['p95'])}" for n, v in nl.items()))
    lines += ["", "## 违规类型（出现该违规的记录数）", ""]
    for s in systems:
        vc = summary["violations"][s]
        lines.append(f"- **{s}**：" + ("、".join(f"`{k}` {v}" for k, v in list(vc.items())[:8]) or "无"))
    failures = [r for r in records if r.system == "full" and r.feasible_gold and not r.hard_pass][:5]
    if failures:
        lines += ["", "## full 系统失败样例", ""]
        lines += [
            f"- `{r.sample_id}`（{r.difficulty}，rep {r.rep}）：{', '.join(r.violations) or r.error or 'structure'}" for r in failures
        ]
    return "\n".join(lines) + "\n"


def write_report(out: Path, meta: dict[str, Any], records: list[RunRecord]) -> Path:
    summary = build_summary(records)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    path = out / "report.md"
    path.write_text(render(meta, summary, records), encoding="utf-8")
    return path


def _records(run: Path) -> list[RunRecord]:
    path = run / "records.jsonl"
    if not path.exists():
        return []
    return [RunRecord.model_validate_json(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def compare(a: Path, b: Path) -> str:
    sa, sb = (json.loads((p / "summary.json").read_text(encoding="utf-8"))["metrics"] for p in (a, b))
    ra, rb = by_system(_records(a)), by_system(_records(b))
    ma, mb = (json.loads((p / "meta.json").read_text(encoding="utf-8")) for p in (a, b))
    lines = [f"# 对比：`{a.name}` (A) vs `{b.name}` (B)", ""]
    if ma.get("mock") or mb.get("mock"):
        lines += ["> ⚠️ 至少一次运行为 **MOCK 模式**：差异只反映代码 / mock 规划器的变化，不代表真实模型质量。", ""]
    lines += ["| | A | B |", "|---|---|---|"]
    for key in ("date", "git_commit", "model", "dataset", "n_samples", "repeats"):
        lines.append(f"| {key} | {ma.get(key)} | {mb.get(key)} |")
    lines.append("")
    for system in [s for s in sa if s in sb]:
        lines += [f"## {system}", "", "| 指标 | A | B | Δ (B−A) |", "|---|---|---|---|"]
        for k, (label, _) in METRICS.items():
            if k in JUDGE_KEYS:  # unpaired means mix different sample sets; compare on samples judged in both runs
                if k == "judge_total":
                    pa, pb, n = paired_judge(ra.get(system, []), rb.get(system, []))
                    if n:
                        lines.append(f"| 合格计划质量分（两次都通过的 {n} 个样本） | {pa:.2f} | {pb:.2f} | {pb - pa:+.2f} |")  # type: ignore[operator]
                continue
            va, vb = sa[system][k]["mean"], sb[system][k]["mean"]
            if va is None and vb is None:
                continue
            delta = "n/a" if va is None or vb is None else (f"{(vb - va) * 100:+.1f} pp" if k in PERCENT else f"{vb - va:+.2f}")
            lines.append(f"| {label} | {fmt(k, va)} | {fmt(k, vb)} | {delta} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two eval runs")
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    text = compare(args.run_a, args.run_b)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
