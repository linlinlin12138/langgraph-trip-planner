"""Semantic cache: hard constraints filter exactly, soft preferences by similarity, TTL / eviction, graph reuse."""

from __future__ import annotations

from datetime import date
from typing import Any

from langchain_core.messages import HumanMessage

from trip_agents.cache.semantic import SemanticCache, meta_key, preference_text
from trip_agents.graph.builder import build_graph
from trip_agents.graph.deps import AgentDeps, GraphOptions, LocalToolExecutor
from trip_agents.llm.mock import MockChatModel
from trip_agents.observability.tracer import TraceRecorder
from trip_agents.schemas import TripRequest

from .conftest import TODAY


def req(**kw: Any) -> TripRequest:
    hard = {"destination": "杭州", "start_date": date(2026, 11, 20), "days": 3, "budget_total": 6000, "travelers": 2} | kw.pop("hard", {})
    return TripRequest.model_validate({"hard": hard, "soft": {"interests": ["nature", "food"]} | kw})


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


def test_hard_constraints_are_exact_filters() -> None:
    c = SemanticCache()
    c.put(req(), preference_text(req()), {"v": 1})
    assert c.lookup(req(), preference_text(req())) is not None
    for hard in ({"dietary_restrictions": ["no_spicy"]}, {"travelers": 3}, {"days": 4}, {"start_date": date(2026, 11, 21)},
                 {"must_visit": ["灵隐寺"]}, {"budget_total": 9000}):  # fmt: skip
        other = req(hard=hard)
        assert meta_key(other) != meta_key(req())
        assert c.lookup(other, preference_text(other)) is None, hard
    same_bucket = req(hard={"budget_total": 6200})  # within the same 500-yuan bucket
    assert c.lookup(same_bucket, preference_text(same_bucket)) is not None


def test_soft_preferences_by_similarity_threshold() -> None:
    c = SemanticCache(threshold=0.85)
    c.put(req(), preference_text(req()), {"v": 1})
    reordered = req(interests=["food", "nature"])
    assert (hit := c.lookup(reordered, preference_text(reordered))) and hit.similarity == 1.0
    for different in (req(interests=["shopping", "nightlife"]), req(pace="packed"), req(hotel_tier="luxury")):
        assert c.lookup(different, preference_text(different)) is None


def test_ttl_and_hit_counting() -> None:
    clock = Clock()
    c = SemanticCache(ttl_s=100, clock=clock)
    c.put(req(), preference_text(req()), {"v": 1})
    assert c.lookup(req(), preference_text(req())) and c.stats()["total_hits"] == 1
    clock.t += 101
    assert c.lookup(req(), preference_text(req())) is None and c.stats()["entries"] == 0


def test_eviction_keeps_frequent_and_recent() -> None:
    clock = Clock()
    c = SemanticCache(max_entries=2, clock=clock, age_weight=0.3)
    a, b, d = req(hard={"days": 2}), req(hard={"days": 3}), req(hard={"days": 4})
    c.put(a, preference_text(a), {"k": "a"})
    c.put(b, preference_text(b), {"k": "b"})
    for _ in range(3):
        c.lookup(a, preference_text(a))  # a is popular
    clock.t += 86400
    c.put(d, preference_text(d), {"k": "d"})  # over capacity -> evict lowest score: b (no hits, older)
    assert c.lookup(b, preference_text(b)) is None
    assert c.lookup(a, preference_text(a)) and c.lookup(d, preference_text(d))
    assert c.score(3, clock.t, clock.t) > c.score(0, clock.t - 86400, clock.t)


def test_graph_reuses_and_revalidates_cached_plan() -> None:
    cache = SemanticCache()
    deps = AgentDeps(model=MockChatModel(), tools=LocalToolExecutor(), today=TODAY, cache=cache)
    g = build_graph(deps, GraphOptions(human_in_loop=False))
    first = g.invoke({"messages": [HumanMessage(content="我和女朋友11月20日去杭州玩3天，预算6000，喜欢自然和美食")]})
    assert first["final"].passed_validation and cache.stats()["entries"] == 1

    t = TraceRecorder()
    second = g.invoke(
        {"messages": [HumanMessage(content="11月20日，我和女朋友，杭州3天，总预算6000元，喜欢美食和自然风景")]}, {"callbacks": [t]}
    )
    assert second["cache_hit"] and second["cache_hit"]["similarity"] == 1.0
    assert not second["tool_calls"] and {n.node for n in t.finish().nodes} >= {"cache_lookup", "budget", "validate"}
    assert second["final"].itinerary == first["final"].itinerary and second["final"].passed_validation
    assert cache.stats()["entries"] == 1  # a hit is not stored again

    third = g.invoke({"messages": [HumanMessage(content="我和女朋友11月20日去杭州玩3天，预算6000，喜欢自然和美食，不吃辣")]})
    assert third["cache_hit"] is None and third["tool_calls"]  # dietary restriction differs -> full pipeline
