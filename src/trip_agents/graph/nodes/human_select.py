"""Human-in-the-loop after Research: the user picks hotel / restaurants / attractions.

`interrupt()` pauses the graph and persists state via the checkpointer; the UI
resumes with `Command(resume={"hotel_id": ..., "restaurant_ids": [...], ...})`.
Anything the user sends is sanitised against the research results, so the UI
can never inject an id the planner hasn't seen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.types import interrupt

from trip_agents.graph.deps import GraphOptions
from trip_agents.graph.state import TripState
from trip_agents.schemas import ResearchBundle, Selection, TripRequest
from trip_agents.validation.names import matches_any

PER_DAY = {"relaxed": 2, "moderate": 3, "packed": 4}
# Below this, money per person per day after lodging (45% of the budget, as in hotel selection) is "tight".
# Dev/test: 48 of 49 feasible hard samples are below 250, 2 of 70 easy, 0 of 69 medium.
TIGHT_PER_PERSON_DAY = 250.0


@dataclass(frozen=True)
class BudgetCaps:
    per_person_day: float  # after lodging
    meal: float  # per person per meal
    ticket: float  # per person per attraction


def budget_caps(req: TripRequest) -> BudgetCaps | None:
    """Price caps for candidates when the budget is tight, else None (rank by rating as usual)."""
    h = req.hard
    per_day = h.budget_total * 0.55 / (h.days * h.travelers)
    if per_day >= TIGHT_PER_PERSON_DAY:
        return None
    return BudgetCaps(per_day, meal=round(per_day * 0.35, 1), ticket=round(per_day * 0.3 / PER_DAY[req.soft.pace], 1))


def default_selection(req: TripRequest, rb: ResearchBundle) -> Selection:
    h, s = req.hard, req.soft
    nights = max(h.days - 1, 1)
    rooms = (h.travelers + 1) // 2
    cap = h.budget_total * 0.45 / nights / rooms
    # within budget first, then preferred tier, then rating; if nothing fits the budget, cheapest first
    hotels = sorted(
        rb.hotels,
        key=lambda x: (
            x.price_per_night > cap,
            s.hotel_tier is not None and x.tier != s.hotel_tier,
            -x.rating if x.price_per_night <= cap else x.price_per_night,
        ),
    )
    diet = set(h.dietary_restrictions)
    caps = budget_caps(req)  # tight budget: affordable candidates first, so the planner starts from them
    meal_cap = caps.meal if caps else float("inf")
    ticket_cap = caps.ticket if caps else float("inf")
    rests = sorted(rb.restaurants, key=lambda r: (not diet <= set(r.dietary_tags), r.avg_price_per_person > meal_cap, -r.rating))
    interests = set(s.interests)
    attrs = sorted(
        rb.attractions,
        key=lambda a: (not matches_any(a.name, h.must_visit), a.ticket_price > ticket_cap, a.category not in interests, -a.rating),
    )
    n_attr = min(len(attrs), h.days * PER_DAY[s.pace] + 2)
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
        if not options.human_in_loop or not (rb.hotels or rb.attractions or rb.restaurants):
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
