"""Human-in-the-loop after Research: the user picks hotel / restaurants / attractions.

`interrupt()` pauses the graph and persists state via the checkpointer; the UI
resumes with `Command(resume={"hotel_id": ..., "restaurant_ids": [...], ...})`.
Anything the user sends is sanitised against the research results, so the UI
can never inject an id the planner hasn't seen.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from trip_agents.graph.deps import GraphOptions
from trip_agents.graph.state import TripState
from trip_agents.schemas import ResearchBundle, Selection, TripRequest

_PER_DAY = {"relaxed": 2, "moderate": 3, "packed": 4}


def default_selection(req: TripRequest, rb: ResearchBundle) -> Selection:
    h, s = req.hard, req.soft
    nights = max(h.days - 1, 1)
    rooms = (h.travelers + 1) // 2
    cap = h.budget_total * 0.45 / nights / rooms
    hotels = sorted(rb.hotels, key=lambda x: (s.hotel_tier is not None and x.tier != s.hotel_tier, x.price_per_night > cap, -x.rating))
    diet = set(h.dietary_restrictions)
    rests = sorted(rb.restaurants, key=lambda r: (not diet <= set(r.dietary_tags), -r.rating))
    must = set(h.must_visit)
    interests = set(s.interests)
    attrs = sorted(rb.attractions, key=lambda a: (a.name not in must, a.category not in interests, -a.rating))
    n_attr = min(len(attrs), h.days * _PER_DAY[s.pace] + 2)
    return Selection(
        hotel_id=hotels[0].id if hotels and h.days > 1 else None,
        restaurant_ids=[r.id for r in rests[: max(6, h.days * 2)]],
        attraction_ids=[a.id for a in attrs[:n_attr]],
    )


def sanitize_selection(raw: Any, rb: ResearchBundle, defaults: Selection) -> Selection:
    try:
        sel = raw if isinstance(raw, Selection) else Selection.model_validate(raw or {})
    except Exception:
        return defaults
    hotel = sel.hotel_id if sel.hotel_id and rb.hotel(sel.hotel_id) else defaults.hotel_id
    rests = [r for r in sel.restaurant_ids if rb.restaurant(r)] or defaults.restaurant_ids
    attrs = [a for a in sel.attraction_ids if rb.attraction(a)] or defaults.attraction_ids
    return Selection(hotel_id=hotel, restaurant_ids=rests, attraction_ids=attrs)


def make_human_select_node(options: GraphOptions):  # type: ignore[no-untyped-def]
    def human_select(state: TripState) -> dict[str, Any]:
        rb, req = state["research"], state["request"]
        defaults = default_selection(req, rb)
        if not options.human_in_loop:
            return {"selection": defaults, "status": "planning"}
        answer = interrupt(
            {
                "type": "select_candidates",
                "hotels": [x.model_dump(mode="json") for x in rb.hotels],
                "restaurants": [x.model_dump(mode="json") for x in rb.restaurants],
                "attractions": [x.model_dump(mode="json") for x in rb.attractions],
                "weather": [x.model_dump(mode="json") for x in rb.weather],
                "defaults": defaults.model_dump(mode="json"),
            }
        )
        return {"selection": sanitize_selection(answer, rb, defaults), "status": "planning"}

    return human_select
