"""Budget agent: arithmetic is deterministic code; the LLM only writes advice.

LLMs are unreliable at summing dozens of numbers, and a budget check that is
itself wrong would poison the validator. So totals are computed here and the
model is consulted only when the plan is over budget.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from trip_agents.graph.costs import activity_cost, audit, hotel_cost
from trip_agents.graph.deps import AgentDeps, GraphOptions
from trip_agents.graph.prompts import BUDGET_SYSTEM
from trip_agents.graph.state import TripState
from trip_agents.llm.context import render_context
from trip_agents.llm.structured import invoke_structured
from trip_agents.schemas import BudgetAdvice, BudgetLine, BudgetReport, Itinerary, ResearchBundle, TripRequest


def compute_budget(req: TripRequest, rb: ResearchBundle, it: Itinerary, model_costs: dict[str, list[float]] | None = None) -> BudgetReport:
    """Totals from catalog prices (graph/costs.py), never from the model's est_cost.

    ``model_costs`` (day_index -> the model's original est_cost per activity, in order) is only used to
    fill the audit fields; without it the activities' current est_cost is taken as the model's.
    """
    n, days = req.hard.travelers, req.hard.days
    rooms = (n + 1) // 2
    nights = max(days - 1, 0)
    sums = {"meals": 0.0, "tickets": 0.0, "transit": 0.0, "other": 0.0}
    for d in it.days:
        for a in d.activities:
            key = {"meal": "meals", "attraction": "tickets", "transit": "transit"}.get(a.kind, "other")
            sums[key] += activity_cost(a, rb, n)
    h = rb.hotel(it.hotel_id) if it.hotel_id else None
    lines = [
        BudgetLine(
            category="hotel",
            amount=hotel_cost(req, rb, it.hotel_id),
            detail=f"{h.name} {h.price_per_night:.0f}元/晚 × {nights}晚 × {rooms}间" if h and nights else "无需住宿",
        )
    ]
    lines += [BudgetLine(category=k, amount=round(v, 2)) for k, v in sums.items() if v > 0]

    # Cheapest plan possible with what research found: lower bound for feasibility.
    diet = set(req.hard.dietary_restrictions)
    ok_rest = [r.avg_price_per_person for r in rb.restaurants if diet <= set(r.dietary_tags)]
    cheapest_hotel = min((x.price_per_night for x in rb.hotels), default=0.0)
    must_tickets = sum(a.ticket_price for a in rb.attractions if a.name in req.hard.must_visit) * n
    min_cost = cheapest_hotel * nights * rooms + min(ok_rest, default=0.0) * 2 * days * n + must_tickets

    total = round(sum(li.amount for li in lines), 2)
    pairs = []
    for d in it.days:
        orig = (model_costs or {}).get(str(d.day_index))
        aligned = orig is not None and len(orig) == len(d.activities)
        pairs += [(a, orig[k] if aligned else a.est_cost) for k, a in enumerate(d.activities)]  # type: ignore[index]
    model_refs, catalog_refs, model_all = audit(pairs, rb, n)
    return BudgetReport(
        lines=lines,
        total=total,
        budget=req.hard.budget_total,
        min_feasible_cost=round(min_cost, 2),
        model_refs_total=model_refs,
        catalog_refs_total=catalog_refs,
        model_plan_total=round(lines[0].amount + model_all, 2),
    )


def make_budget_node(deps: AgentDeps, options: GraphOptions):  # type: ignore[no-untyped-def]
    def budget(state: TripState, config: RunnableConfig) -> dict[str, Any]:
        req, rb, it = state["request"], state["research"], state["itinerary"]
        report = compute_budget(req, rb, it, state.get("model_costs"))
        if report.total > report.budget and options.budget_advice:
            ctx = render_context(
                {
                    "task": "budget_advice",
                    "report": report.model_dump(mode="json") | {"over_by": report.over_by},
                    "request": req.model_dump(mode="json"),
                }
            )
            res = invoke_structured(
                deps.model,
                BudgetAdvice,
                [SystemMessage(content=BUDGET_SYSTEM), HumanMessage(content=ctx)],
                max_retries=deps.structured_max_retries,
                stage="budget",
                item="advice",
                config=config,
            )
            if res.value:
                report.suggestions = res.value.suggestions
        return {"budget": report}

    return budget
