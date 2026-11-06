"""Ground-truth world for evaluation.

The graph validates plans against *its own* research results. Evaluation must not
trust that: we re-check every final plan against the full catalog and against
the *gold* constraints (not the system's extraction of them). References may be
ids or names (systems without retrieval only know names); anything that does not
resolve counts as a hallucinated entity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from trip_agents.graph.costs import apply_catalog_costs
from trip_agents.graph.nodes.budget import compute_budget
from trip_agents.schemas import Attraction, BudgetReport, Hotel, Itinerary, ResearchBundle, Restaurant, TripRequest, ValidationIssue
from trip_agents.tools.mock_backend import _load_all
from trip_agents.validation.rules import validate_itinerary


@lru_cache
def world_bundle(city: str) -> ResearchBundle:
    data = _load_all().get(city, {"hotels": [], "restaurants": [], "attractions": []})
    return ResearchBundle(
        hotels=[Hotel(**h) for h in data["hotels"]],
        restaurants=[Restaurant(**r) for r in data["restaurants"]],
        attractions=[Attraction(**a) for a in data["attractions"]],
    )


def _resolve(ref: str | None, by_id: set[str], by_name: dict[str, str]) -> str | None:
    if not ref:
        return ref
    if ref in by_id:
        return ref
    if ref in by_name:
        return by_name[ref]
    hits = [i for n, i in by_name.items() if ref in n or n in ref]
    return hits[0] if len(hits) == 1 else ref


def normalize_refs(it: Itinerary, world: ResearchBundle) -> tuple[Itinerary, int, int]:
    """Map name references to catalog ids. Returns (itinerary, n_refs, n_unresolved)."""
    a_ids, a_names = {a.id for a in world.attractions}, {a.name: a.id for a in world.attractions}
    r_ids, r_names = {r.id for r in world.restaurants}, {r.name: r.id for r in world.restaurants}
    h_ids, h_names = {h.id for h in world.hotels}, {h.name: h.id for h in world.hotels}
    out = it.model_copy(deep=True)
    total = unresolved = 0
    if out.hotel_id:
        total += 1
        out.hotel_id = _resolve(out.hotel_id, h_ids, h_names)
        unresolved += out.hotel_id not in h_ids
    for d in out.days:
        for a in d.activities:
            if a.kind == "attraction":
                a.ref_id = _resolve(a.ref_id, a_ids, a_names)
                total += 1
                unresolved += a.ref_id not in a_ids
            elif a.kind == "meal":
                a.ref_id = _resolve(a.ref_id, r_ids, r_names)
                total += 1
                unresolved += a.ref_id not in r_ids
    return out, total, unresolved


@dataclass
class PlanEvaluation:
    structure_ok: bool
    violations: list[ValidationIssue] = field(default_factory=list)
    budget: BudgetReport | None = None
    n_refs: int = 0
    hallucinated_refs: int = 0
    min_feasible_cost: float = 0.0

    @property
    def errors(self) -> list[str]:
        return [v.code for v in self.violations if v.severity == "error"]

    @property
    def hard_pass(self) -> bool:
        return self.structure_ok and not self.errors


def min_feasible_cost(gold: TripRequest) -> float:
    """Lower bound over the full catalog (same formula as the Budget agent)."""
    empty = Itinerary(hotel_id=None, days=[])
    return compute_budget(gold, world_bundle(gold.hard.destination), empty).min_feasible_cost


def align_days(it: Itinerary, gold: TripRequest) -> Itinerary:
    """Match days by date, not by day_index numbering.

    The graph numbers days from 0, but a single-shot model may number them from 1 with the
    right dates; keyed by index that reads as "day 1 missing, every date wrong, one extra
    day". The evaluation checks the dates, so a day whose date is one of the trip's dates
    is re-indexed to that position; other days keep their index and are flagged as before.
    """
    pos = {dt: i for i, dt in enumerate(gold.hard.dates)}
    if all(d.date in pos for d in it.days) and len({d.date for d in it.days}) == len(it.days):
        out = it.model_copy(deep=True)
        for d in out.days:
            d.day_index = pos[d.date]
        return out
    return it


def evaluate_plan(it: Itinerary | None, gold: TripRequest) -> PlanEvaluation:
    if it is None:
        return PlanEvaluation(structure_ok=False, min_feasible_cost=min_feasible_cost(gold))
    world = world_bundle(gold.hard.destination)
    norm, total, unresolved = normalize_refs(align_days(it, gold), world)
    structure_ok = len(norm.days) == gold.hard.days and all(d.activities for d in norm.days)
    # catalog prices decide the budget (graph/costs.py); the plan's own est_cost only feeds the audit fields
    budget = compute_budget(gold, world, norm)
    priced = apply_catalog_costs(norm, world, gold.hard.travelers)
    issues = validate_itinerary(priced, gold, world, budget)
    return PlanEvaluation(structure_ok, issues, budget, total, unresolved, budget.min_feasible_cost)
