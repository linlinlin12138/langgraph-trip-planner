"""The one place where the cost of a plan is decided.

``Activity.est_cost`` is written by the model. Summing it would let a plan pass the budget
check -- in the graph's own validator and in the evaluation -- by under-quoting a meal or a
ticket instead of choosing something cheaper. So for every activity that resolves to the
catalog the price comes from the catalog:

* attraction  ticket_price x travelers
* meal        avg_price_per_person x travelers
* rest        0
* transit     the model's estimate (activities are not linked to a concrete route)
* an attraction / meal whose ref does not resolve (invented entity): the model's estimate;
  such a plan already fails on ``unknown_ref``

The model's own numbers are kept only for auditing how far they were off (``audit``).
"""

from __future__ import annotations

from collections.abc import Iterable

from trip_agents.schemas import Activity, Itinerary, ResearchBundle, TripRequest


def catalog_cost(a: Activity, rb: ResearchBundle, travelers: int) -> float | None:
    """Catalog price of an attraction / meal, or None if it doesn't resolve (or isn't priced by the catalog)."""
    if a.kind == "attraction" and (x := rb.attraction(a.ref_id or "")):
        return round(x.ticket_price * travelers, 2)
    if a.kind == "meal" and (r := rb.restaurant(a.ref_id or "")):
        return round(r.avg_price_per_person * travelers, 2)
    return None


def activity_cost(a: Activity, rb: ResearchBundle, travelers: int) -> float:
    if a.kind == "rest":
        return 0.0
    c = catalog_cost(a, rb, travelers)
    return a.est_cost if c is None else c


def hotel_cost(req: TripRequest, rb: ResearchBundle, hotel_id: str | None) -> float:
    h = rb.hotel(hotel_id) if hotel_id else None
    if h is None or req.hard.days <= 1:
        return 0.0
    return h.price_per_night * (req.hard.days - 1) * ((req.hard.travelers + 1) // 2)


def plan_total(req: TripRequest, rb: ResearchBundle, it: Itinerary) -> float:
    """Lodging + every activity at its authoritative cost. compute_budget and the budget repair both use this."""
    n = req.hard.travelers
    return round(hotel_cost(req, rb, it.hotel_id) + sum(activity_cost(a, rb, n) for d in it.days for a in d.activities), 2)


def apply_catalog_costs(it: Itinerary, rb: ResearchBundle, travelers: int) -> Itinerary:
    """Copy with every est_cost replaced by the authoritative value, so budget, validation, repair,
    display and the judge all see the same number."""
    out = it.model_copy(deep=True)
    for d in out.days:
        for a in d.activities:
            a.est_cost = activity_cost(a, rb, travelers)
    return out


def audit(pairs: Iterable[tuple[Activity, float]], rb: ResearchBundle, travelers: int) -> tuple[float, float, float]:
    """(model sum, catalog sum) over activities that resolve to the catalog, and the model's sum over all activities.

    ``pairs`` is (activity, the model's original est_cost) -- the activity may already carry the rewritten cost.
    """
    model_refs = catalog_refs = model_all = 0.0
    for a, model_est in pairs:
        model_all += model_est
        c = catalog_cost(a, rb, travelers)
        if c is not None:
            model_refs += model_est
            catalog_refs += c
    return round(model_refs, 2), round(catalog_refs, 2), round(model_all, 2)
