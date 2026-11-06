"""Research agent: a ReAct loop expressed as a LangGraph subgraph.

    agent ──(has tool_calls)──▶ tools ──▶ agent
      └──(no tool_calls / step limit)──▶ END

We use a custom tools node instead of the prebuilt `ToolNode` because it needs
three policies the prebuilt one doesn't have:
  * argument validation with the error text returned as the observation;
  * duplicate-call detection (same tool + same args -> cached result, flagged);
  * a per-tool round budget (max N calls of each tool type).
The ResearchBundle is assembled from successful tool results, never from the
model's prose, so candidates shown to the user are guaranteed to be real.
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from trip_agents.graph.deps import AgentDeps, GraphOptions
from trip_agents.graph.nodes.human_select import PER_DAY, budget_caps
from trip_agents.graph.prompts import RESEARCH_SYSTEM
from trip_agents.graph.state import ToolCallRecord, TripState
from trip_agents.llm.context import render_context
from trip_agents.schemas import Attraction, Hotel, ResearchBundle, Restaurant, Route, TripRequest, WeatherDay
from trip_agents.validation.names import matches_any


class ResearchState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    records: Annotated[list[ToolCallRecord], operator.add]
    results: Annotated[list[dict[str, Any]], operator.add]  # successful {name, data}
    step: int


def call_key(name: str, args: dict[str, Any]) -> str:
    return name + ":" + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def build_research_subgraph(deps: AgentDeps):  # type: ignore[no-untyped-def]
    schemas = deps.tools.tool_schemas()
    llm = deps.model.bind_tools(schemas)

    def agent(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
        step = state.get("step", 0)
        if step >= deps.research_max_steps:
            return {"messages": [AIMessage(content=f"已达到最大步数 {deps.research_max_steps}，停止检索。")]}
        return {"messages": [llm.invoke(state["messages"], config=config)], "step": step + 1}

    def tools(state: ResearchState) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        prior = state.get("records", [])
        cache = {call_key(r["name"], r["args"]): r for r in prior if r["ok"]}
        observations = {call_key(r["name"], r["args"]): obs for r, obs in zip(prior, _obs_cache(state), strict=False)}
        rounds: dict[str, int] = {}
        for r in prior:
            if not r["duplicate"]:
                rounds[r["name"]] = rounds.get(r["name"], 0) + 1

        msgs, records, results = [], [], []
        step = state.get("step", 0)
        for tc in last.tool_calls:
            name, args = tc["name"], tc["args"]
            key = call_key(name, args)
            if key in cache:
                obs = "（重复调用：与之前参数完全相同，直接返回之前的结果，请勿重复调用）\n" + observations.get(key, "")
                records.append(ToolCallRecord(name=name, args=args, ok=True, error=None, latency_ms=0.0, step=step, duplicate=True))
            elif rounds.get(name, 0) >= deps.research_max_rounds_per_tool:
                obs = f"ERROR: 工具 {name} 已达到最多 {deps.research_max_rounds_per_tool} 轮的调用上限，请基于已有信息继续。"
                records.append(
                    ToolCallRecord(name=name, args=args, ok=False, error="round_limit", latency_ms=0.0, step=step, duplicate=False)
                )
            else:
                res = deps.tools.call(name, args)
                rounds[name] = rounds.get(name, 0) + 1
                obs = res.to_observation()
                records.append(
                    ToolCallRecord(name=name, args=args, ok=res.ok, error=res.error, latency_ms=res.latency_ms, step=step, duplicate=False)
                )
                if res.ok:
                    results.append({"name": name, "data": res.data})
            msgs.append(
                ToolMessage(
                    content=obs, tool_call_id=tc["id"] or "", name=name, status="success" if not obs.startswith("ERROR") else "error"
                )
            )
        return {"messages": msgs, "records": records, "results": results}

    def route(state: ResearchState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    g = StateGraph(ResearchState)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, ["tools", END])
    g.add_edge("tools", "agent")
    return g.compile()


def _obs_cache(state: ResearchState) -> list[str]:
    """Observation texts aligned with `records` (ToolMessages appear in the same order)."""
    return [m.content if isinstance(m.content, str) else str(m.content) for m in state["messages"] if isinstance(m, ToolMessage)]


def hotel_budget_per_night(req: TripRequest) -> float:
    """Rule of thumb: at most ~45% of the total budget goes to lodging."""
    nights = max(req.hard.days - 1, 1)
    rooms = (req.hard.travelers + 1) // 2
    return req.hard.budget_total * 0.45 / nights / rooms


def assemble_bundle(results: list[dict[str, Any]], summary: str = "") -> ResearchBundle:
    hotels: dict[str, Hotel] = {}
    rests: dict[str, Restaurant] = {}
    attrs: dict[str, Attraction] = {}
    weather: list[WeatherDay] = []
    routes: list[Route] = []
    for r in results:
        name, data = r["name"], r["data"]
        if name == "search_hotels":
            hotels.update({d["id"]: Hotel(**d) for d in data})
        elif name == "search_restaurants":
            rests.update({d["id"]: Restaurant(**d) for d in data})
        elif name == "search_attractions":
            attrs.update({d["id"]: Attraction(**d) for d in data})
        elif name == "get_weather":
            weather = [WeatherDay(**d) for d in data]
        elif name == "get_route":
            routes.append(Route(**data))
    return ResearchBundle(
        hotels=list(hotels.values()),
        restaurants=list(rests.values()),
        attractions=list(attrs.values()),
        weather=weather,
        routes=routes,
        notes=summary,
    )


def top_up(results: list[dict[str, Any]], req: TripRequest, tools: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Deterministic follow-up search when the agent's results can't fill the plan.

    The agent tends to search attractions only by the user's interest categories; with
    1-3 catalog entries per category that leaves the planner a single attraction for a
    multi-day trip, and it pads the days with "rest". Same for restaurants when it adds
    cuisine/price filters. Here, if fewer candidates came back than the planner needs,
    search once more without those filters (hard constraints -- city and dietary
    restrictions -- are kept). These calls are not agent decisions, so they are kept out
    of the tool-call records and the process metrics; the bundle notes say they happened.
    Must-visit places missing from the results are looked up by name.
    """
    bundle = assemble_bundle(results)
    h = req.hard
    need_attr = h.days * PER_DAY[req.soft.pace] + 2
    need_rest = max(6, h.days * 2)
    diet = set(h.dietary_restrictions)
    extra: list[dict[str, Any]] = []
    notes: list[str] = []
    have = [a.name for a in bundle.attractions]
    for name in h.must_visit:  # category / rating search can miss a named landmark in a large catalog
        if not matches_any(name, have):
            res = tools.call("search_attractions", {"city": h.destination, "keyword": name})
            if res.ok and res.data:
                extra.append({"name": "search_attractions", "data": res.data})
                notes.append(f"必去景点「{name}」未在检索结果中，已按名称补充检索")
    if len(bundle.attractions) < need_attr:
        res = tools.call("search_attractions", {"city": h.destination, "limit": 15})
        if res.ok:
            extra.append({"name": "search_attractions", "data": res.data})
            notes.append(f"景点仅 {len(bundle.attractions)} 个（需要 {need_attr}），已不限类别补充检索")
    caps = budget_caps(req)
    if caps is not None:  # tight budget: make sure affordable options exist for the planner and the budget repair
        cheap_rest = sum(diet <= set(r.dietary_tags) and r.avg_price_per_person <= caps.meal for r in bundle.restaurants)
        if cheap_rest < need_rest:
            args = {"city": h.destination, "dietary": sorted(diet), "max_price_per_person": caps.meal, "limit": 15}
            res = tools.call("search_restaurants", args)
            if res.ok and res.data:
                extra.append({"name": "search_restaurants", "data": res.data})
                notes.append(f"预算紧，补充检索人均 {caps.meal:.0f} 元以内的餐厅")
        cheap_attr = sum(a.ticket_price <= caps.ticket for a in bundle.attractions)
        if cheap_attr < need_attr:
            res = tools.call("search_attractions", {"city": h.destination, "max_ticket_price": caps.ticket, "limit": 15})
            if res.ok and res.data:
                extra.append({"name": "search_attractions", "data": res.data})
                notes.append(f"预算紧，补充检索门票 {caps.ticket:.0f} 元以内的景点")
    if sum(diet <= set(r.dietary_tags) for r in bundle.restaurants) < need_rest:
        res = tools.call("search_restaurants", {"city": h.destination, "dietary": sorted(diet), "limit": 15})
        if res.ok:
            extra.append({"name": "search_restaurants", "data": res.data})
            notes.append("符合饮食限制的餐厅不足，已只按饮食限制补充检索")
    return extra, notes


def make_research_node(deps: AgentDeps, options: GraphOptions | None = None):  # type: ignore[no-untyped-def]
    sub = build_research_subgraph(deps)

    def research(state: TripState, config: RunnableConfig) -> dict[str, Any]:
        req = state["request"]
        system = RESEARCH_SYSTEM.format(max_rounds=deps.research_max_rounds_per_tool)
        ctx = render_context(
            {"task": "research", "request": req.model_dump(mode="json"), "hotel_budget_per_night": round(hotel_budget_per_night(req), -1)}
        )
        out = sub.invoke(
            {"messages": [SystemMessage(content=system), HumanMessage(content="请为以下需求检索信息：\n" + ctx)]},
            config={**config, "recursion_limit": 4 * deps.research_max_steps + 5},
        )
        final = out["messages"][-1]
        summary = final.content if isinstance(final, AIMessage) and isinstance(final.content, str) else ""
        results = out.get("results", [])
        extra, notes = top_up(results, req, deps.tools)
        bundle = assemble_bundle(results + extra, "\n".join([summary, *notes]).strip())
        return {"research": bundle, "tool_calls": out.get("records", []), "status": "awaiting_selection"}

    return research
