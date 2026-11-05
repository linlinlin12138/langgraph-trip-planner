"""LLM paraphrases of the template dataset, with checks that the answer key still holds.

    uv run python -m trip_agents.evals.rewrite \\
        --rewriter-provider anthropic --rewriter-model <m> --checker-provider openai --checker-model <m>

Risk being controlled: a paraphrase that drops a constraint, changes a number, or turns
"must" into "would be nice" silently invalidates ``gold`` and with it every metric.

1. Model separation. Rewriter, checker and the system under test (``--tested-provider``)
   must be three different providers: a checker from the rewriter's family shares its
   blind spots, and a checker from the tested family would keep only requests that
   family parses well, inflating its extraction scores.
2. Gates on every attempt, cheapest first:
   a. rules: must-visit names verbatim, no hedge word next to them, no vague amount next
      to the budget, no relative dates (hard constraints must be uniquely determinable);
   b. back-extraction: the checker turns the text back into constraints with its own
      strict prompt (not the system's Chat prompt), compared field by field with gold:
      all hard fields exactly, plus interests / pace / hotel tier.
   A failed attempt is retried with the concrete mismatches as feedback; after
   ``--max-attempts`` the sample keeps its template text (``rewrite="template"``), so
   strata and infeasible samples are never lost.
3. ``spotcheck.csv``: a stratified sample for human review (do it in the same sitting
   as the judge calibration labels).
4. Vagueness only on the soft side: the prompt allows vague wording for interests, pace
   and hotel tier, never for hard constraints.

Output goes to ``evals/datasets/llm/`` (the template set is untouched); run evals on it
with ``--dataset evals/datasets/llm/dev.jsonl``. Resumable: finished ids in
``rewrite_log.jsonl`` are skipped, so trial runs (``--limit`` / ``--difficulty``) count
towards the full run. Reuse is refused if the rewriter or checker model changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from trip_agents.evals.dataset import DATASET_DIR, EVAL_TODAY, Sample, load, save
from trip_agents.evals.systems import extraction_accuracy
from trip_agents.schemas import RequestExtraction, TripRequest

OUT_DIR = DATASET_DIR / "llm"

# ----------------------------------------------------------------------------- prompts
_DIET_CN = {"no_spicy": "不吃辣", "no_seafood": "不吃海鲜/海鲜过敏"}
_INTEREST_CN = {
    "history": "历史古迹", "nature": "自然风景", "museum": "博物馆", "food": "美食小吃",
    "shopping": "逛街购物", "nightlife": "夜景/夜生活", "culture": "文化艺术",
}  # fmt: skip
_PACE_CN = {"relaxed": "轻松", "moderate": "适中（可以不提）", "packed": "紧凑"}
_TIER_CN = {"budget": "经济型", "midrange": "舒适型", "luxury": "高档"}

STYLES = [
    "随手打字，很短，口语化，可以省略主语",
    "先交代一两句背景（为什么去、和谁去），再说需求",
    "像在微信里问朋友，语气轻松，可以带语气词",
    "有点啰嗦，想到哪说到哪，分两三句说",
    "直接干脆，像在给客服提需求",
]

REWRITER_SYSTEM = """你在为旅行规划系统的评测集改写用户请求。给你一份结构化需求，写一段真实用户会在聊天框里打的中文。

【硬约束：必须能被唯一确定，一个都不能少、不能改】
- 目的地、出发日期、天数、人数、预算、饮食限制、必去景点。
- 日期写成具体日期（如“12月5号”“2026-12-05”“12月5日”），不要用“下周末”“下个月”这类相对说法。
- 预算给出准确数字（可以写“6000”“六千块”“6k”），不要加“左右/大概/差不多”。可以说总预算，人数大于 1 且能整除时也可以说人均。
- 人数要能数清楚（“我和老婆”=2，“一家三口”=3），不要写“几个朋友”。
- 饮食限制要说成必须遵守的（“不能吃辣”“海鲜过敏”），不要说“尽量少吃辣”。
- 必去景点逐字使用给出的名称，并且明确是必须去的（“一定要去”“必须安排”）。不要写“最好去”“有空的话去”。

【软偏好：可以说得模糊、换说法】
- 兴趣、节奏、住宿档次可以用自然、模糊的说法表达（如“想多拍点照”“别太赶”“住得别太差”），但意思不能变，也不能漏掉。
- 节奏为“适中”时可以不提；住宿档次没给时不要提；没给兴趣时不要编造兴趣。

【禁止】添加需求里没有的约束；使用列表或 JSON；解释你在做什么。只输出用户说的话。"""

CHECKER_SYSTEM = """你是数据核验员。把用户的旅行请求抽取成结构化字段，严格按字面意思，不要猜测。
- 今天是 {today}。只给月日的日期，取今天之后最近的那一天。
- budget_total 是全体人员的总预算（人均预算要乘以人数）。
- 只有用户明确表示必须遵守的才放进 dietary_restrictions 和 must_visit；“最好”“尽量”“有空的话”之类的放进 notes。
- must_visit 使用用户原文里的景点名称。
- 用户没提到的字段留空，不要填默认值。pace 只在用户表达了节奏时填写。
- interests 只能从这些值中选：history, nature, museum, food, shopping, nightlife, culture。"""


def describe_gold(req: TripRequest) -> str:
    h, s = req.hard, req.soft
    lines = [
        f"目的地：{h.destination}",
        f"出发日期：{h.start_date.isoformat()}（今天是 {EVAL_TODAY.isoformat()}）",
        f"天数：{h.days}",
        f"人数：{h.travelers}",
        f"总预算：{h.budget_total:.0f} 元"
        + (f"（人均 {h.budget_total / h.travelers:.0f}）" if h.travelers > 1 and h.budget_total % h.travelers == 0 else ""),
        "饮食限制：" + ("、".join(_DIET_CN[d] for d in h.dietary_restrictions) or "无（不要提）"),
        "必去景点：" + ("、".join(h.must_visit) or "无（不要提）"),
        "兴趣：" + ("、".join(_INTEREST_CN[i] for i in s.interests) or "无（不要提）"),
        f"节奏：{_PACE_CN[s.pace]}",
        "住宿档次：" + (_TIER_CN[s.hotel_tier] if s.hotel_tier else "无（不要提）"),
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- gates
HEDGES = ("最好", "尽量", "如果有时间", "有时间的话", "有空的话", "可以的话", "看情况", "顺便", "也行", "不一定")
VAGUE_AMOUNT = ("左右", "大概", "差不多", "上下", "大约", "估计", "来块")
RELATIVE_DATES = (
    "今天",
    "明天",
    "后天",
    "下周",
    "这周",
    "周末",
    "下个月",
    "月底",
    "月初",
    "过几天",
    "国庆",
    "元旦",
    "春节",
    "五一",
    "圣诞",
)
_MONEY = re.compile(r"预算|块|元|钱|\d+\s*[kK千万]|[千万]")


def _clauses(text: str) -> list[str]:
    return [c for c in re.split(r"[，,。.!！?？；;\n]", text) if c.strip()]


def rule_errors(text: str, gold: TripRequest) -> list[str]:
    errs: list[str] = []
    clauses = _clauses(text)
    for name in gold.hard.must_visit:
        if name not in text:
            errs.append(f"必去景点“{name}”没有逐字出现")
        elif any(name in c and any(h in c for h in HEDGES) for c in clauses):
            errs.append(f"必去景点“{name}”被说成了可选（附近有“最好/尽量/有空的话”等）")
    for c in clauses:
        if _MONEY.search(c) and any(v in c for v in VAGUE_AMOUNT):
            errs.append(f"预算说法不精确：“{c.strip()}”")
    for w in RELATIVE_DATES:
        if w in text:
            errs.append(f"使用了相对日期或节日“{w}”，日期必须是具体日期")
    return errs


def compare(ext: RequestExtraction, gold: TripRequest) -> list[str]:
    """Field-by-field diff of a back-extraction against gold; empty = consistent."""
    if ext.missing_required():
        return [f"反向抽取缺少字段：{', '.join(ext.missing_required())}"]
    pred = ext.to_request()
    errs = [
        f"{k} 不一致：抽取 {getattr(pred.hard, k)!r}，应为 {getattr(gold.hard, k)!r}"
        for k, ok in extraction_accuracy(pred, gold).items()
        if not ok
    ]
    s, g = ext, gold.soft
    if set(s.interests) != set(g.interests):
        errs.append(f"兴趣不一致：抽取 {sorted(s.interests)}，应为 {sorted(g.interests)}")
    if (s.pace or "moderate") != g.pace:
        errs.append(f"节奏不一致：抽取 {s.pace or 'moderate'}，应为 {g.pace}")
    if s.hotel_tier != g.hotel_tier:
        errs.append(f"住宿档次不一致：抽取 {s.hotel_tier}，应为 {g.hotel_tier}")
    return errs


# ----------------------------------------------------------------------------- pipeline
RewriteFn = Callable[[str, str], str]  # (system, user) -> text
CheckFn = Callable[[str], RequestExtraction | None]


@dataclass
class Outcome:
    id: str
    text: str
    accepted: bool
    attempts: list[dict[str, Any]] = field(default_factory=list)


def rewrite_one(sample: Sample, rewrite: RewriteFn, check: CheckFn, *, max_attempts: int = 3) -> Outcome:
    style = STYLES[random.Random(sample.id).randrange(len(STYLES))]
    base = f"需求：\n{describe_gold(sample.gold)}\n\n风格：{style}"
    out = Outcome(sample.id, sample.request_text, False)
    feedback = ""
    for _ in range(max_attempts):
        text = rewrite(REWRITER_SYSTEM, base + feedback).strip().strip('“”"')
        errs = rule_errors(text, sample.gold)
        ext = None
        if not errs:  # back-extraction only for texts that pass the free rule gate
            ext = check(text)
            errs = ["核验模型未返回合法结构"] if ext is None else compare(ext, sample.gold)
        out.attempts.append(
            {"text": text, "errors": errs, "extraction": ext.model_dump(mode="json", exclude_defaults=True) if ext else None}
        )
        if not errs:
            out.text, out.accepted = text, True
            return out
        feedback = "\n\n上一版有问题，请重写并修正：\n- " + "\n- ".join(errs) + f"\n上一版：{text}"
    return out


def apply(samples: list[Sample], outcomes: dict[str, Outcome], info: dict[str, Any]) -> list[Sample]:
    res = []
    for s in samples:
        o = outcomes[s.id]
        upd: dict[str, Any] = {"rewrite_info": {**info, "attempts": len(o.attempts), "accepted": o.accepted}}
        if o.accepted:
            upd |= {"request_text": o.text, "rewrite": "llm"}
        res.append(s.model_copy(update=upd))
    return res


def write_spotcheck(samples: list[Sample], path: Path, k: int = 30, seed: int = 0) -> None:
    """Stratified pick of accepted rewrites for human review; ok/notes columns are left empty."""
    rng = random.Random(seed)
    pool = [s for s in samples if s.rewrite == "llm"]
    groups: dict[tuple[str, bool], list[Sample]] = {}
    for s in pool:
        groups.setdefault((s.difficulty, s.feasible), []).append(s)
    picked: list[Sample] = []
    for key in sorted(groups):  # proportional, at least one per stratum
        g = groups[key]
        picked += rng.sample(g, min(len(g), max(1, round(k * len(g) / max(1, len(pool))))))
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel opens CJK correctly
        w = csv.writer(f)
        w.writerow(["id", "difficulty", "feasible", "gold", "llm_text", "attempts", "ok(y/n)", "notes"])
        for s in sorted(picked, key=lambda x: x.id):
            w.writerow([s.id, s.difficulty, s.feasible, describe_gold(s.gold).replace("\n", "；"), s.request_text,
                        (s.rewrite_info or {}).get("attempts"), "", ""])  # fmt: skip


def run(
    splits: dict[str, list[Sample]],
    rewrite: RewriteFn,
    check: CheckFn,
    info: dict[str, Any],
    out_dir: Path = OUT_DIR,
    *,
    workers: int = 4,
    max_attempts: int = 3,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "rewrite_log.jsonl"
    _check_same_models(out_dir, info)
    outcomes: dict[str, Outcome] = {}
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            outcomes[d["id"]] = Outcome(**d)
    todo = [s for ss in splits.values() for s in ss if s.id not in outcomes]
    lock = threading.Lock()

    with log.open("a", encoding="utf-8") as f:

        def work(s: Sample) -> None:
            o = rewrite_one(s, rewrite, check, max_attempts=max_attempts)
            with lock:
                outcomes[s.id] = o
                f.write(json.dumps(o.__dict__, ensure_ascii=False) + "\n")
                f.flush()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, todo))

    all_new: list[Sample] = []
    for name, ss in splits.items():
        new = apply(ss, outcomes, info)
        save(new, out_dir / f"{name}.jsonl")
        all_new += new
    write_spotcheck(all_new, out_dir / "spotcheck.csv")
    used = [outcomes[s.id] for s in all_new]
    reasons = Counter(e.split("：")[0].split("“")[0] for o in used for a in o.attempts for e in a["errors"])
    meta = {
        **info,
        "n": len(used),
        "accepted": sum(o.accepted for o in used),
        "fallback_to_template": sum(not o.accepted for o in used),
        "first_try": sum(o.accepted and len(o.attempts) == 1 for o in used),
        "attempts_total": sum(len(o.attempts) for o in used),
        "failure_reasons": dict(reasons.most_common()),
        "accepted_by_stratum": _by_stratum(all_new),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def _check_same_models(out_dir: Path, info: dict[str, Any]) -> None:
    """Cached outcomes are only valid for the models that produced them."""
    meta_path = out_dir / "meta.json"
    if not (out_dir / "rewrite_log.jsonl").exists() or not meta_path.exists():
        return
    old = json.loads(meta_path.read_text(encoding="utf-8"))
    for role in ("rewriter", "checker"):
        if old.get(role) != info.get(role):
            raise SystemExit(
                f"{out_dir}/rewrite_log.jsonl was made with {role}={old.get(role)}, now {info.get(role)}; "
                "use another --out or delete the log to start over"
            )


def _by_stratum(samples: list[Sample]) -> dict[str, str]:
    out = {}
    for d, fz in sorted({(s.difficulty, s.feasible) for s in samples}):
        g = [s for s in samples if (s.difficulty, s.feasible) == (d, fz)]
        out[f"{d}/{'feasible' if fz else 'infeasible'}"] = f"{sum(s.rewrite == 'llm' for s in g)}/{len(g)}"
    return out


# ----------------------------------------------------------------------------- CLI
def model_fns(rewriter: tuple[str, str | None], checker: tuple[str, str | None]) -> tuple[RewriteFn, CheckFn]:
    from trip_agents.llm.factory import get_chat_model
    from trip_agents.llm.structured import invoke_structured

    rw = get_chat_model(rewriter[0], rewriter[1], temperature=0.9)  # type: ignore[arg-type]
    ck = get_chat_model(checker[0], checker[1], temperature=0.0)  # type: ignore[arg-type]
    system = CHECKER_SYSTEM.format(today=EVAL_TODAY.isoformat())

    def rewrite(sys: str, user: str) -> str:
        c = rw.invoke([SystemMessage(content=sys), HumanMessage(content=user)]).content
        return c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))

    def check(text: str) -> RequestExtraction | None:
        r = invoke_structured(ck, RequestExtraction, [SystemMessage(content=system), HumanMessage(content=text)], stage="rewrite_check")
        return r.value

    return rewrite, check


def main() -> None:
    from trip_agents.llm.factory import DEFAULT_MODELS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rewriter-provider", required=True)
    ap.add_argument("--rewriter-model", default=None)
    ap.add_argument("--checker-provider", required=True)
    ap.add_argument("--checker-model", default=None)
    ap.add_argument("--tested-provider", default="gemini", help="provider of the system under test")
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--difficulty", default=None, help="only these difficulties, e.g. hard or medium,hard")
    ap.add_argument("--limit", type=int, default=0, help="first N samples per split (0 = all); for a cheap trial")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    roles = {"rewriter": args.rewriter_provider, "checker": args.checker_provider, "tested": args.tested_provider}
    if len(set(roles.values())) < 3 and "mock" not in roles.values():
        raise SystemExit(f"rewriter, checker and tested system must be three different providers, got {roles}")

    splits = {n: load(DATASET_DIR / f"{n}.jsonl") for n in args.splits.split(",")}
    if args.difficulty:
        keep = set(args.difficulty.split(","))
        splits = {n: [s for s in ss if s.difficulty in keep] for n, ss in splits.items()}
    if args.limit:
        splits = {n: ss[: args.limit] for n, ss in splits.items()}
    info = {
        "rewriter": {"provider": args.rewriter_provider, "model": args.rewriter_model or DEFAULT_MODELS[args.rewriter_provider]},
        "checker": {"provider": args.checker_provider, "model": args.checker_model or DEFAULT_MODELS[args.checker_provider]},
        "tested_provider": args.tested_provider,
    }
    rewrite, check = model_fns((args.rewriter_provider, args.rewriter_model), (args.checker_provider, args.checker_model))
    meta = run(splits, rewrite, check, info, args.out, workers=args.workers, max_attempts=args.max_attempts)
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
