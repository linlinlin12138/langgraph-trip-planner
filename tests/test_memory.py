"""Long-term memory: merge rules, defaults precedence, store cap, and the full two-session flow."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from trip_agents.graph.builder import build_graph, sqlite_checkpointer
from trip_agents.graph.deps import AgentDeps, GraphOptions, LocalToolExecutor
from trip_agents.llm.mock import MockChatModel
from trip_agents.memory.ops import apply_memory_defaults, companions_to_travelers
from trip_agents.memory.profile import ExtractedPreference, TripSummary, UserProfile, merge_profile
from trip_agents.memory.store import MemoryManager, sqlite_store
from trip_agents.schemas import RequestExtraction

from .conftest import TODAY

T0 = datetime(2026, 11, 1, 10, 0)


def pref(field: str, value: object, evidence: str) -> ExtractedPreference:
    return ExtractedPreference(field=field, value=value, evidence=evidence)  # type: ignore[arg-type]


def test_merge_only_keeps_explicit_and_valid_preferences() -> None:
    texts = ["我和女朋友去杭州，不吃辣，节奏轻松点"]
    extracted = [
        pref("dietary_restrictions", ["no_spicy"], "不吃辣"),
        pref("pace", "relaxed", "节奏轻松点"),
        pref("hotel_tier", "luxury", "喜欢五星酒店"),  # not said -> rejected
        pref("pace", "super-fast", "轻松"),  # invalid value -> rejected
    ]
    prof, changes, rejected = merge_profile(UserProfile(user_id="u"), extracted, user_texts=texts, thread_id="t1", now=T0)
    assert prof.get("dietary_restrictions") == ["no_spicy"] and prof.get("pace") == "relaxed" and prof.hotel_tier is None
    assert len(changes) == 2 and len(rejected) == 2


def test_conflict_newest_wins_with_history() -> None:
    p, _, _ = merge_profile(UserProfile(user_id="u"), [pref("pace", "relaxed", "轻松")], user_texts=["轻松"], thread_id="t1", now=T0)
    later = T0 + timedelta(days=30)
    p2, changes, _ = merge_profile(p, [pref("pace", "packed", "紧凑")], user_texts=["紧凑"], thread_id="t2", now=later)
    assert p2.get("pace") == "packed" and p2.pace and p2.pace.updated_at == later and p2.pace.source_thread == "t2"
    assert p2.pace.history == [{"value": "relaxed", "updated_at": T0.isoformat()}]
    assert changes == ["节奏：relaxed → packed"]
    # same value again -> timestamp refresh, no new history entry
    p3, changes3, _ = merge_profile(
        p2, [pref("pace", "packed", "紧凑")], user_texts=["紧凑"], thread_id="t3", now=later + timedelta(days=1)
    )
    assert not changes3 and p3.pace and len(p3.pace.history) == 1 and p3.pace.source_thread == "t3"


def _profile() -> UserProfile:
    p, _, _ = merge_profile(
        UserProfile(user_id="u"),
        [pref("dietary_restrictions", ["halal"], "清真"), pref("pace", "relaxed", "轻松"), pref("companions", "家人（3人）", "一家三口"),
         pref("budget_per_person_per_day", 500, "预算")],
        user_texts=["一家三口 清真 轻松 预算"], thread_id="t", now=T0,
    )  # fmt: skip
    return p


def test_defaults_fill_only_unstated_fields() -> None:
    ex = RequestExtraction(destination="成都", start_date=date(2027, 1, 1), days=2, pace="packed")
    merged, applied = apply_memory_defaults(ex, _profile(), None)
    assert merged.pace == "packed"  # explicit wins
    assert merged.dietary_restrictions == ["halal"] and merged.travelers == 3
    assert merged.budget_total == 3000  # 500 x 3 people x 2 days
    assert set(applied) == {"dietary_restrictions", "travelers", "budget_total"}
    none, applied2 = apply_memory_defaults(ex.model_copy(update={"dietary_none": True}), _profile(), None)
    assert none.dietary_restrictions == [] and "dietary_restrictions" not in applied2


def test_past_trip_beats_profile() -> None:
    past = TripSummary(thread_id="old", created_at=T0, destination="杭州", start_date=date(2026, 5, 1), days=4, travelers=2,
                       budget_total=8000, pace="moderate", interests=["nature"])  # fmt: skip
    ex = RequestExtraction(start_date=date(2027, 1, 1), refers_to_past_trip=True)
    merged, applied = apply_memory_defaults(ex, _profile(), past)
    assert (merged.destination, merged.days, merged.travelers, merged.pace) == ("杭州", 4, 2, "moderate")
    assert merged.budget_total == 8000 and merged.dietary_restrictions == ["halal"]  # diet not in past trip -> profile
    assert "上次的杭州行程" in applied["pace"] and "用户画像" in applied["dietary_restrictions"]


def test_companions_parsing() -> None:
    assert [companions_to_travelers(x) for x in ("女朋友", "家人（3人）", "独自", "猫", None)] == [2, 3, 1, None, None]


def test_store_caps_trips_and_finds_by_hint(tmp_path: Path) -> None:
    mm = MemoryManager(sqlite_store(tmp_path / "s.sqlite"), max_trips=3)
    for i, city in enumerate(["北京", "杭州", "成都", "西安"]):
        trip = TripSummary(
            thread_id=f"t{i}",
            created_at=T0 + timedelta(days=i),
            destination=city,
            start_date=date(2026, 1, 1),
            days=2,
            travelers=1,
            budget_total=1000,
            pace="moderate",
        )
        evicted = mm.add_trip("u", trip)
    assert evicted == ["t0"] and [t.destination for t in mm.list_trips("u")] == ["西安", "成都", "杭州"]
    assert mm.find_trip("u", "去杭州").destination == "杭州"  # type: ignore[union-attr]
    assert mm.find_trip("u", None).destination == "西安"  # type: ignore[union-attr]
    assert mm.find_trip("nobody") is None


# ----------------------------------------------------------------------------- graph flow
def _run(g, tid: str, text: str, answers: list[str]):  # type: ignore[no-untyped-def]
    cfg = {"configurable": {"thread_id": tid}}
    out = g.invoke({"messages": [HumanMessage(content=text)], "user_id": "alice"}, cfg)
    asked = []
    while "__interrupt__" in out:
        v = out["__interrupt__"][0].value
        if v["type"] == "clarify":
            asked.append(v)
            ans: object = answers.pop(0)
        elif v["type"] == "select_candidates":
            ans = v["defaults"]
        else:
            ans = {"confirmed": True}
        out = g.invoke(Command(resume=ans), cfg)
    return out, asked


def test_two_sessions_with_confirmation(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path / "store.sqlite")
    deps = AgentDeps(model=MockChatModel(), tools=LocalToolExecutor(), today=TODAY, clock=lambda: T0)
    g = build_graph(deps, checkpointer=sqlite_checkpointer(tmp_path / "cp.sqlite"), store=store)

    out1, asked1 = _run(g, "t1", "我和女朋友11月20日去杭州玩3天，预算6000，不吃辣，节奏轻松点", [])
    assert not asked1 and out1["status"] == "confirmed" and out1["memory_updates"]
    prof = MemoryManager(store).get_profile("alice")
    assert prof and prof.get("dietary_restrictions") == ["no_spicy"] and prof.get("companions") == "女朋友"

    out2, asked2 = _run(g, "t2", "12月10日去成都玩2天", ["可以"])
    assert len(asked2) == 1 and asked2[0]["memory_defaults"]  # chat agent asked to confirm the defaults
    r = out2["request"]
    assert r.hard.dietary_restrictions == ["no_spicy"] and r.hard.travelers == 2 and r.soft.pace == "relaxed"
    assert r.hard.budget_total == 4000  # 1000 per person-day x 2 x 2
    assert len(MemoryManager(store).list_trips("alice")) == 2

    out3, _ = _run(g, "t3", "1月5日去北京玩2天，这次什么都吃，预算5000", ["可以"])
    assert out3["request"].hard.dietary_restrictions == []  # explicit "什么都吃" overrides memory
    prof3 = MemoryManager(store).get_profile("alice")
    assert prof3 and prof3.get("dietary_restrictions") == [] and prof3.dietary_restrictions
    assert prof3.dietary_restrictions.history[-1]["value"] == ["no_spicy"]


def test_no_store_or_disabled_means_no_memory() -> None:
    store = InMemoryStore()
    MemoryManager(store).put_profile(_profile().model_copy(update={"user_id": "alice"}))
    text = "12月10日去成都玩2天，预算3000"
    for g in (
        build_graph(AgentDeps(model=MockChatModel(), tools=LocalToolExecutor(), today=TODAY), GraphOptions(human_in_loop=False)),
        build_graph(
            AgentDeps(model=MockChatModel(), tools=LocalToolExecutor(), today=TODAY),
            GraphOptions(human_in_loop=False, enable_memory=False),
            store=store,
        ),
    ):
        out = g.invoke({"messages": [HumanMessage(content=text)], "user_id": "alice"})
        assert out["request"].hard.dietary_restrictions == [] and not out.get("memory_defaults")
    g = build_graph(
        AgentDeps(model=MockChatModel(), tools=LocalToolExecutor(), today=TODAY), GraphOptions(human_in_loop=False), store=store
    )
    out = g.invoke({"messages": [HumanMessage(content=text)], "user_id": "alice"})
    assert out["request"].hard.dietary_restrictions == ["halal"]  # auto mode applies defaults without asking
