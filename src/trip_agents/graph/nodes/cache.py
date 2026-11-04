"""Semantic-cache nodes.

chat -> cache_lookup -(hit)-> budget -> validate -> ...   (cached plan is re-checked, never trusted blindly)
                     -(miss)-> research -> ...
confirm -> stores the confirmed plan (unless it came from the cache)
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from trip_agents.cache.semantic import SemanticCache, preference_text
from trip_agents.graph.state import TripState
from trip_agents.schemas import Itinerary, ResearchBundle, Selection


def make_cache_lookup_node(cache: SemanticCache | None):  # type: ignore[no-untyped-def]
    def cache_lookup(state: TripState) -> dict[str, Any]:
        if cache is None:
            return {"cache_hit": None}
        hit = cache.lookup(state["request"], preference_text(state["request"]))
        if hit is None:
            return {"cache_hit": None}
        p = hit.payload
        return {
            "cache_hit": {"entry_id": hit.entry_id, "similarity": hit.similarity},
            "research": ResearchBundle.model_validate(p["research"]),
            "selection": Selection.model_validate(p["selection"]),
            "itinerary": Itinerary.model_validate(p["itinerary"]),
            "revision": 0,
            "status": "planning",
            "messages": [AIMessage(content=f"找到相似的已确认行程（相似度 {hit.similarity:.2f}），复用并重新校验……", name="cache")],
        }

    return cache_lookup


def route_after_cache(state: TripState) -> str:
    return "budget" if state.get("cache_hit") else "research"


def store_in_cache(cache: SemanticCache | None, state: dict[str, Any]) -> bool:
    if cache is None or state.get("cache_hit") or not state.get("final") or not state["final"].passed_validation:
        return False
    cache.put(
        state["request"],
        preference_text(state["request"]),
        {
            "research": state["research"].model_dump(mode="json"),
            "selection": state["selection"].model_dump(mode="json"),
            "itinerary": state["final"].itinerary.model_dump(mode="json"),
        },
    )
    return True
