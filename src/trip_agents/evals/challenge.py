"""Challenge set: harder phrasings of test samples, to see how much the template wording flatters the results.

    # 1. pick 36 test samples (12 easy / 12 medium / 12 hard, 3 of them infeasible), fixed seed
    uv run python -m trip_agents.evals.challenge select
    # 2. rewrite them with an Anthropic model (no back-check), then review evals/datasets/challenge/review.csv by hand
    uv run python -m trip_agents.evals.challenge rewrite --model claude-sonnet-5-5
    # 3. apply the review (edited texts / dropped rows) -> evals/datasets/challenge/test_challenge.jsonl
    uv run python -m trip_agents.evals.challenge apply-review
    # 4. evaluate full and baseline_catalog on it (template results are reused from gemini_test_v1)
    make eval ARGS="--split test --dataset evals/datasets/challenge/test_challenge.jsonl --systems full,baseline_catalog ..."
    # 5. template vs challenge on the same ids
    uv run python -m trip_agents.evals.challenge compare evals/results/gemini_test_v1 evals/results/<challenge run>

The gold answers are the test set's, unchanged: only ``request_text`` is rewritten. Unlike
``evals.rewrite`` there is no back-extraction gate and no verbatim rule for must-visit names -- the
point is phrasings the template never produces (short names and nicknames, Chinese numerals, relative
dates, indirect head counts), so a human checks every rewrite instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from trip_agents.evals.dataset import DATASET_DIR, EVAL_TODAY, Sample, load, save
from trip_agents.evals.rewrite import describe_gold
from trip_agents.evals.systems import RunRecord

OUT = DATASET_DIR / "challenge"
FIELDS = ["destination", "start_date", "days", "budget_total", "travelers", "dietary_restrictions", "must_visit"]
FIELD_CN = {"destination": "目的地", "start_date": "出发日期", "days": "天数", "budget_total": "预算", "travelers": "人数",
            "dietary_restrictions": "饮食限制", "must_visit": "必去景点"}  # fmt: skip
PER_DIFFICULTY = 12
INFEASIBLE = {"medium": 2, "hard": 1}  # test has 8 infeasible medium and 1 infeasible hard sample
SEED = 20261107

CHALLENGE_SYSTEM = """你在为旅行规划系统构造一批更难的评测请求。给你一份结构化需求，写一段真实用户会在聊天框里打的中文。
信息必须完整、准确，但说法要比模板自然、多变，让系统更难读懂。

【必须准确、能唯一确定】目的地、出发日期、天数、人数、预算、饮食限制、必去景点，一个都不能少，数值和含义不能变。

【鼓励的写法：每条请求选用其中三四种】
- 日期：可以写“12月5号”“12.5”，也可以用相对今天能唯一确定的说法，如“下个月5号”（今天是 {today}）；
  不要用“周末”“过几天”“月底”这类不能确定到某一天的说法。
- 天数：“玩三天”“三天两夜”“待两晚”（两晚 = 3 天）。
- 人数：间接说法，如“带上爸妈”（= 3 人）、“我们一家四口”、“我和一个同事”；不要写“几个朋友”“和朋友”这类数不清的说法。
- 预算：中文数字或口语，如“两千五”“2.5k”“一人一千二”（人均写法只在能整除时用）；不要加“左右/大概/差不多”。
- 饮食限制：换说法，如“对海鲜过敏”“一点辣都吃不了”，但必须是必须遵守的。
- 必去景点：可以用游客常说的简称或俗称（如“故宫”“兵马俑”“熊猫基地”），但必须明确是必须去的（“一定要去”“必须安排”）。
- 信息顺序打乱，夹在闲聊里。

【软偏好】兴趣、节奏、住宿档次可以用模糊的说法，但不能编造或遗漏；节奏为“适中”时可以不提；没给的不要提。

【禁止】添加需求里没有的硬约束；使用列表或 JSON；解释你在做什么。只输出用户说的话。"""


# ----------------------------------------------------------------------------- 1. select
def select(test: list[Sample], seed: int = SEED) -> list[Sample]:
    rng = random.Random(seed)
    picked: list[Sample] = []
    for d in ("easy", "medium", "hard"):
        pool = [s for s in test if s.difficulty == d]
        k_inf = INFEASIBLE.get(d, 0)
        inf = rng.sample([s for s in pool if not s.feasible], k_inf)
        feas = rng.sample([s for s in pool if s.feasible], PER_DIFFICULTY - k_inf)
        picked += inf + feas
    return sorted(picked, key=lambda s: s.id)


# ----------------------------------------------------------------------------- 2. rewrite
def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


def rewrite_all(samples: list[Sample], provider: str, model: str | None) -> list[Sample]:
    from trip_agents.llm.factory import DEFAULT_MODELS, get_chat_model

    llm = get_chat_model(provider, model, temperature=0.9)  # type: ignore[arg-type]
    system = CHALLENGE_SYSTEM.format(today=EVAL_TODAY.isoformat())
    model_name = model or DEFAULT_MODELS[provider]
    out = []
    for s in samples:
        text = _text(llm.invoke([SystemMessage(content=system), HumanMessage(content="需求：\n" + describe_gold(s.gold))]).content)
        info = {"provider": provider, "model": model_name, "template_text": s.request_text, "reviewed": False}
        out.append(s.model_copy(update={"request_text": text.strip().strip('“”"'), "rewrite": "llm_challenge", "rewrite_info": info}))
    return out


REVIEW_HEADER = [
    "id",
    "difficulty",
    "feasible",
    "gold",
    "template_text",
    "challenge_text",
    "decision(ok/edit/drop)",
    "edited_text",
    "notes",
]


def write_review(samples: list[Sample], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel opens CJK correctly
        w = csv.writer(f)
        w.writerow(REVIEW_HEADER)
        for s in samples:
            tmpl = (s.rewrite_info or {}).get("template_text", "")
            w.writerow([s.id, s.difficulty, s.feasible, describe_gold(s.gold).replace("\n", "；"), tmpl, s.request_text, "", "", ""])


# ----------------------------------------------------------------------------- 3. apply review
def apply_review(samples: list[Sample], path: Path) -> tuple[list[Sample], Counter[str]]:
    rows = {r["id"]: r for r in csv.DictReader(path.open(encoding="utf-8-sig"))}
    out, tally = [], Counter[str]()
    for s in samples:
        r = rows.get(s.id, {})
        decision = (r.get("decision(ok/edit/drop)") or "").strip().lower()
        if decision == "drop":
            tally["drop"] += 1
            continue
        info = dict(s.rewrite_info or {})
        info["reviewed"] = decision in ("ok", "edit")
        if decision == "edit" and (r.get("edited_text") or "").strip():
            s = s.model_copy(update={"request_text": r["edited_text"].strip()})
            tally["edit"] += 1
        else:
            tally["ok" if decision == "ok" else "unreviewed"] += 1
        out.append(s.model_copy(update={"rewrite_info": info}))
    return out, tally


# ----------------------------------------------------------------------------- 5. compare
def _pct(xs: list[bool]) -> str:
    return f"{100 * sum(xs) / len(xs):.1f}%" if xs else "n/a"


def _row(label: str, a: list[bool], b: list[bool]) -> str:
    d = f"{100 * (sum(b) / len(b) - sum(a) / len(a)):+.1f} pp" if a and b else "n/a"
    return f"| {label} | {_pct(a)}（{sum(a)}/{len(a)}） | {_pct(b)}（{sum(b)}/{len(b)}） | {d} |"


def _records(run: Path) -> list[RunRecord]:
    return [RunRecord.model_validate_json(x) for x in (run / "records.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def compare(template_run: Path, challenge_run: Path, dataset: Path, systems: tuple[str, ...] = ("full", "baseline_catalog")) -> str:
    samples = {s.id: s for s in load(dataset)}
    ids = set(samples)
    tmpl = {(r.system, r.sample_id): r for r in _records(template_run) if r.sample_id in ids and r.rep == 0}
    chal = {(r.system, r.sample_id): r for r in _records(challenge_run) if r.sample_id in ids and r.rep == 0}
    lines = [
        f"# 挑战集对比：模板版（`{template_run.name}`）vs 挑战版（`{challenge_run.name}`）",
        "",
        f"同一批 {len(ids)} 个样本 id，标准答案相同，只有请求文本不同。模板版结果直接取自模板运行中相同 id 的记录，没有重跑。",
        "",
    ]
    for system in systems:
        pairs = [(tmpl[(system, i)], chal[(system, i)]) for i in sorted(ids) if (system, i) in tmpl and (system, i) in chal]
        if not pairs:
            continue
        feas = [(a, b) for a, b in pairs if a.feasible_gold]
        lines += [f"## {system}", "", f"配对样本 {len(pairs)} 个（可行 {len(feas)} 个）。", ""]
        lines += ["| 指标 | 模板版 | 挑战版 | Δ |", "|---|---|---|---|"]

        def row(label: str, a: list[bool], b: list[bool], out: list[str] = lines) -> None:
            out.append(_row(label, a, b))

        row("硬约束通过率（可行样本）", [a.hard_pass for a, _ in feas], [b.hard_pass for _, b in feas])
        for d in ("easy", "medium", "hard"):
            sub = [(a, b) for a, b in feas if a.difficulty == d]
            row(f"　{d}", [a.hard_pass for a, _ in sub], [b.hard_pass for _, b in sub])
        ext = [(a, b) for a, b in pairs if a.extraction is not None and b.extraction is not None]
        if ext:
            row("抽取全部正确的样本", [all(a.extraction.values()) for a, _ in ext], [all(b.extraction.values()) for _, b in ext])  # type: ignore[union-attr]
            for f in FIELDS:
                row(f"　{FIELD_CN[f]}", [bool(a.extraction.get(f)) for a, _ in ext], [bool(b.extraction.get(f)) for _, b in ext])  # type: ignore[union-attr]
        else:
            lines.append("| 抽取准确率 | n/a | n/a | 单次生成系统没有单独的抽取步骤 |")
        lines.append("")

        if ext:
            errs = [(b, f) for _, b in ext for f in FIELDS if not b.extraction.get(f)]  # type: ignore[union-attr]
            lines += ["### 挑战版的抽取错误", ""]
            if not errs:
                lines.append("无。")
            else:
                lines += ["| 样本 | 难度 | 字段 | 标准答案 | 抽取结果 | 模板版是否抽对 | 请求原文 |", "|---|---|---|---|---|---|---|"]
                for b, f in errs:
                    g = samples[b.sample_id].gold.hard.model_dump(mode="json")[f]
                    got = (b.extracted or {}).get(f, "（未记录）")
                    t_ok = "对" if (tmpl[(system, b.sample_id)].extraction or {}).get(f) else "错"
                    text = samples[b.sample_id].request_text.replace("|", "｜")
                    lines.append(f"| `{b.sample_id}` | {b.difficulty} | {FIELD_CN[f]} | {g} | {got} | {t_ok} | {text} |")
            lines.append("")

        flips = [(a, b) for a, b in feas if a.hard_pass != b.hard_pass]
        lines += ["### 硬约束结论不同的样本", ""]
        if not flips:
            lines.append("无。")
        else:
            lines += ["| 样本 | 难度 | 模板版 | 挑战版 |", "|---|---|---|---|"]
            for a, b in flips:
                fa = "通过" if a.hard_pass else "、".join(sorted(set(a.violations))) or (a.error or "")[:40]
                fb = "通过" if b.hard_pass else "、".join(sorted(set(b.violations))) or (b.error or "")[:40]
                lines.append(f"| `{a.sample_id}` | {a.difficulty} | {fa} | {fb} |")
        lines.append("")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    rw = sub.add_parser("rewrite")
    rw.add_argument("--provider", default="anthropic")
    rw.add_argument("--model", default="claude-sonnet-5-5")
    sub.add_parser("apply-review")
    cp = sub.add_parser("compare")
    cp.add_argument("template_run", type=Path)
    cp.add_argument("challenge_run", type=Path)
    cp.add_argument("--dataset", type=Path, default=OUT / "test_challenge.jsonl")
    cp.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.cmd == "select":
        picked = select(load(DATASET_DIR / "test.jsonl"))
        (OUT / "ids.json").write_text(json.dumps([s.id for s in picked], indent=1), encoding="utf-8")
        strata: defaultdict[str, int] = defaultdict(int)
        for s in picked:
            strata[f"{s.difficulty}/{'feasible' if s.feasible else 'infeasible'}"] += 1
        print(f"selected {len(picked)} test samples (seed {SEED}): {dict(sorted(strata.items()))} -> {OUT / 'ids.json'}")
    elif args.cmd == "rewrite":
        ids = set(json.loads((OUT / "ids.json").read_text(encoding="utf-8")))
        picked = [s for s in load(DATASET_DIR / "test.jsonl") if s.id in ids]
        if args.provider not in ("anthropic", "mock"):
            raise SystemExit("the challenge set is rewritten by an Anthropic model (rewriter and tested system must differ)")
        new = rewrite_all(picked, args.provider, args.model)
        save(new, OUT / "rewritten.jsonl")
        write_review(new, OUT / "review.csv")
        print(f"rewrote {len(new)} samples -> {OUT / 'rewritten.jsonl'}; review by hand: {OUT / 'review.csv'}")
    elif args.cmd == "apply-review":
        final, tally = apply_review(load(OUT / "rewritten.jsonl"), OUT / "review.csv")
        save(final, OUT / "test_challenge.jsonl")
        print(f"{len(final)} samples -> {OUT / 'test_challenge.jsonl'}; review: {dict(tally)}")
        if tally.get("unreviewed"):
            print(f"warning: {tally['unreviewed']} rows have no decision yet (kept as written)")
    else:
        text = compare(args.template_run, args.challenge_run, args.dataset)
        if args.out:
            args.out.write_text(text, encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
