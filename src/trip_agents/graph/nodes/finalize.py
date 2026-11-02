"""Package the result, then ask the user to confirm (second interrupt)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from trip_agents.graph.deps import AgentDeps, GraphOptions
from trip_agents.graph.nodes.chat import memory_manager
from trip_agents.graph.state import TripState
from trip_agents.memory.ops import remember_trip
from trip_agents.schemas import FinalPlan
from trip_agents.validation.rules import has_errors


def finalize(state: TripState) -> dict[str, Any]:
    it, budget, issues = state["itinerary"], state["budget"], state.get("issues", [])
    passed = not has_errors(issues)
    lines = [
        f"行程已生成（修订 {state.get('revision', 0)} 次）：{it.summary}",
        f"预计总花费 {budget.total:.0f} 元 / 预算 {budget.budget:.0f} 元。",
    ]
    if state.get("adjustments"):
        lines.append("调整说明：" + "；".join(state["adjustments"]))
    if not budget.feasible:
        lines.append(f"注意：按现有候选，最低也需约 {budget.min_feasible_cost:.0f} 元，预算不足。")
    if not passed:
        lines.append("仍有未解决的问题：" + "；".join(i.message for i in issues if i.severity == "error"))
    if budget.suggestions:
        lines.append("省钱建议：" + "；".join(budget.suggestions))
    message = "\n".join(lines)
    final = FinalPlan(
        itinerary=it,
        budget=budget,
        passed_validation=passed,
        feasible=budget.feasible,
        remaining_issues=issues,
        revisions=state.get("revision", 0),
        message=message,
    )
    return {"final": final, "status": "done", "messages": [AIMessage(content=message, name="finalize")]}


def make_confirm_node(deps: AgentDeps, options: GraphOptions):  # type: ignore[no-untyped-def]
    def confirm(state: TripState, config: RunnableConfig) -> dict[str, Any]:
        if options.human_in_loop:
            answer = interrupt({"type": "confirm", "message": state["final"].message})
            confirmed = bool(answer.get("confirmed")) if isinstance(answer, dict) else bool(answer)
        else:
            confirmed = True  # automated runs (evals) count as accepted
        if not confirmed:
            return {}
        mem = memory_manager(options)
        if mem is None:
            return {"status": "confirmed"}
        thread_id = str((config.get("configurable") or {}).get("thread_id", "unknown"))
        notes = remember_trip(mem, deps.model, dict(state), thread_id=thread_id, now=deps.clock(), config=config,
                              max_retries=deps.structured_max_retries)  # fmt: skip
        return {"status": "confirmed", "memory_updates": notes}

    return confirm
