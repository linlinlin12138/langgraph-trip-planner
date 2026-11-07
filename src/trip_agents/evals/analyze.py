"""Paired analysis of a finished run, written to a separate file (the run directory is not touched).

    uv run python -m trip_agents.evals.analyze evals/results/gemini_test_v1 --out docs/eval/gemini_test_v1_analysis.md

Same sections that new run reports contain (evals/report.py ``analysis_lines``): pass flips between
systems, paired judge differences with a bootstrap CI, attractions per passing plan, failure attribution.
Use it for runs whose report predates these sections, without regenerating their report.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trip_agents.evals.report import analysis_lines
from trip_agents.evals.systems import RunRecord


def analyze(run: Path) -> str:
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    records = [RunRecord.model_validate_json(x) for x in (run / "records.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    judge = meta.get("judge") or {}
    head = [
        f"# 配对分析 · {meta['name']}",
        "",
        f"由 `python -m trip_agents.evals.analyze {run}` 从 `records.jsonl` 生成（运行目录未修改）。"
        f"模型 {meta['provider']} / `{meta['model']}`（推理强度 {meta.get('thinking_level') or '默认'}），"
        f"Judge {judge.get('provider', '无')} / `{judge.get('model', '-')}`，数据集 `{meta['dataset']}`（{meta['n_samples']} 条），"
        f"每组 {meta['repeats']} 次，记录 {len(records)} 条。",
        "",
    ]
    return "\n".join(head + analysis_lines(records)) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    text = analyze(args.run)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
