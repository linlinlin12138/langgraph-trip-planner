"""Smoke test of the real tool backend against the live services (Open-Meteo weather, OSRM routes).

    uv run python scripts/smoke_real_backend.py            # all checks
    uv run python scripts/smoke_real_backend.py --no-mcp   # skip the MCP stdio subprocess

No API key and no LLM cost: the end-to-end part uses the mock model. Needs outbound HTTPS to
geocoding-api.open-meteo.com, api.open-meteo.com and router.project-osrm.org. These are public,
best-effort demo services: rate limits (429) and network trouble are reported as WARN (service
unavailable), not FAIL -- rerun later before concluding anything about the code.

1. Tool calls through both transports -- in-process (LocalToolExecutor) and MCP stdio (a
   `trip_agents.tools.mcp_server` subprocess with TOOL_BACKEND=real) -- printing the parsed results.
2. Error cases must come back as observations (ok=False, "ERROR: ...") and never crash the program:
   forecast beyond 16 days, a place that geocodes to nothing, invalid coordinates sent to OSRM,
   a network timeout, invalid arguments.
3. End to end: one real request per transport through the whole graph with the mock model and the
   real backend; checks that the plan's weather and routes are the live results (they differ from the
   mock backend's).

Exit status 0 if nothing FAILed (WARNs allowed), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import httpx

from trip_agents.graph.deps import AgentDeps, GraphOptions, LocalToolExecutor
from trip_agents.tools.executor import ToolResult, execute_tool
from trip_agents.tools.mock_backend import MockBackend
from trip_agents.tools.real_backend import RealBackend
from trip_agents.tools.specs import WeatherInput

Call = Callable[[str, dict[str, Any]], ToolResult]
RESULTS: list[tuple[str, str, str]] = []  # (check, PASS / FAIL / WARN, detail)
_UNAVAILABLE = ("429", "Too Many Requests", "Timeout", "timed out", "ConnectError", "502", "503", "504")


def unavailable(res: ToolResult | None) -> bool:
    return res is not None and not res.ok and any(m in (res.error or "") for m in _UNAVAILABLE)


def check(name: str, ok: bool, detail: str = "", res: ToolResult | None = None) -> None:
    status = "PASS" if ok else ("WARN" if unavailable(res) else "FAIL")
    if status == "WARN" and res is not None:
        detail = "service unavailable (rate limit / network), not a code failure: " + (res.error or "")[:120]
    RESULTS.append((name, status, detail))
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))


def show(res: ToolResult) -> str:
    body = json.dumps(res.data, ensure_ascii=False)[:300] if res.ok else res.to_observation()[:300]
    return f"{res.name} ok={res.ok} {res.latency_ms:.0f}ms {body}"


# ----------------------------------------------------------------------------- 1. tool calls
def tool_calls(call: Call, label: str) -> None:
    print(f"\n== 1. tool calls via {label}")
    today = date.today()
    w = call("get_weather", {"city": "杭州", "start_date": today.isoformat(), "end_date": (today + timedelta(days=2)).isoformat()})
    print("   ", show(w))
    days = w.data if w.ok else []
    ok = w.ok and len(days) == 3 and all({"condition", "temp_high", "temp_low"} <= d.keys() for d in days)
    check(f"{label}: weather for 3 days", ok, res=w)

    r = call("get_route", {"city": "杭州", "origin": "hz-a00", "destination": "灵隐寺"})  # catalog id / name -> coordinates
    print("   ", show(r))
    check(f"{label}: route between catalog places", r.ok and r.data["distance_km"] > 0 and r.data["duration_min"] > 0, res=r)

    # A place outside the catalog is geocoded with Open-Meteo, which only knows towns and cities: a POI such as a
    # station is expected NOT to resolve. What matters is that this comes back as an observation, not a crash.
    r2 = call("get_route", {"city": "杭州", "origin": "杭州东站", "destination": "西湖", "mode": "taxi"})
    print("   ", show(r2))
    resolved = r2.ok and r2.data["mode"] == "taxi"
    as_obs = not r2.ok and r2.to_observation().startswith("ERROR")
    detail = "resolved" if resolved else "not located (expected for a POI)"
    check(f"{label}: non-catalog place is routed or reported as an observation", resolved or as_obs, detail, res=r2)


# ----------------------------------------------------------------------------- 2. error cases
class BadCoordinates(RealBackend):
    """Geocodes one sentinel name to an impossible coordinate, to see what OSRM does with it."""

    def geocode(self, name: str) -> tuple[float, float]:
        return (999.0, 999.0) if "坐标测试点" in name else super().geocode(name)


def expect_error(label: str, name: str, call: Call, tool: str, args: dict[str, Any], contains: str, backend: Any = None) -> None:
    try:
        res = execute_tool(tool, args, backend) if backend is not None else call(tool, args)
    except Exception as e:  # the whole point: this must not happen
        check(f"{label}: {name}", False, f"raised {type(e).__name__}: {e}")
        return
    obs = res.to_observation()
    print("   ", show(res))
    if contains not in obs and unavailable(res) and name != "network timeout":  # e.g. a 429 before the case was reached
        check(f"{label}: {name}", False, res=res)
        return
    check(
        f"{label}: {name}", not res.ok and obs.startswith("ERROR") and contains in obs, "" if contains in obs else f"missing {contains!r}"
    )


def error_cases(call: Call, label: str, in_process: bool) -> None:
    print(f"\n== 2. error cases via {label}")
    today = date.today()
    far = today + timedelta(days=30)
    one_day = {"city": "杭州", "start_date": today.isoformat(), "end_date": today.isoformat()}
    beyond = {**one_day, "start_date": far.isoformat(), "end_date": (far + timedelta(days=1)).isoformat()}
    expect_error(label, "forecast beyond 16 days", call, "get_weather", beyond, "16")
    expect_error(label, "place that geocodes to nothing", call, "get_weather", {**one_day, "city": "火星基地乌托邦市"}, "无法定位")
    bad_args = {"city": "杭州", "start_date": "2026-13-40", "end_date": "x"}
    expect_error(label, "invalid arguments", call, "get_weather", bad_args, "参数")
    if in_process:  # these need a specially configured backend object, so only in-process
        bad = {"city": "杭州", "origin": "坐标测试点", "destination": "灵隐寺"}
        expect_error(label, "invalid coordinates sent to OSRM", call, "get_route", bad, "400", backend=BadCoordinates())
        slow = RealBackend(client=httpx.Client(timeout=0.001))
        expect_error(label, "network timeout", call, "get_weather", one_day, "异常", backend=slow)


# ----------------------------------------------------------------------------- 3. end to end
def end_to_end(tools: Any, label: str, text: str) -> None:
    from langchain_core.messages import HumanMessage

    from trip_agents.graph.builder import build_graph
    from trip_agents.llm.mock import MockChatModel

    print(f"\n== 3. end to end via {label}: {text}")
    deps = AgentDeps(model=MockChatModel(), tools=tools, today=date.today())
    out = build_graph(deps, GraphOptions(human_in_loop=False)).invoke(
        {"messages": [HumanMessage(content=text)], "user_id": "smoke"}, {"recursion_limit": 80}
    )
    rb, final, calls = out.get("research"), out.get("final"), out.get("tool_calls", [])
    print("    tool calls:", ", ".join(f"{c['name']}{'' if c['ok'] else '(ERR: ' + (c['error'] or '')[:40] + ')'}" for c in calls))
    check(f"{label}: plan produced", final is not None, "" if final else f"status={out.get('status')}")
    if rb is None:
        return
    req = out["request"]
    # if a live service was unavailable during the run (429 / network), a missing result is a WARN, not a FAIL; the
    # mock research policy also keeps retrying a failed weather call up to the round limit, so it may never ask for a route
    outage = next((c for c in calls if not c["ok"] and any(m in (c["error"] or "") for m in _UNAVAILABLE)), None)
    outage_res = ToolResult(name=outage["name"], args={}, ok=False, error=outage["error"]) if outage else None
    if rb.weather:
        print("    weather:", [(w.date.isoformat(), w.condition, w.temp_high, w.temp_low) for w in rb.weather])
        wi = WeatherInput(city=req.hard.destination, start_date=req.hard.start_date, end_date=req.hard.dates[-1])
        live = [(w.temp_high, w.temp_low, w.condition) for w in rb.weather]
        check(f"{label}: weather in the plan is the live forecast (not the mock one)",
              live != [(w.temp_high, w.temp_low, w.condition) for w in MockBackend().get_weather(wi)])  # fmt: skip
    else:
        check(f"{label}: weather in the plan", False, "no weather in research results", res=outage_res)
    if rb.routes:
        print("    routes:", [(x.origin, x.destination, x.mode, x.distance_km, x.duration_min) for x in rb.routes])
        check(f"{label}: route in the plan came from OSRM", all(x.distance_km > 0 for x in rb.routes))
    elif req.hard.days > 1:
        check(f"{label}: route in the plan", False, "no route in research results", res=outage_res)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-mcp", action="store_true", help="skip the MCP stdio subprocess")
    args = ap.parse_args()
    d = date.today() + timedelta(days=3)
    local = LocalToolExecutor(RealBackend())
    tool_calls(local.call, "in-process")
    error_cases(local.call, "in-process", in_process=True)
    end_to_end(local, "in-process", f"我和女朋友{d.month}月{d.day}日去杭州玩2天，预算5000，喜欢自然风景，一定要去灵隐寺")
    if not args.no_mcp:
        from trip_agents.tools.mcp_client import McpToolExecutor

        env = {**os.environ, "TOOL_BACKEND": "real"}  # the subprocess builds its backend from settings
        with McpToolExecutor.stdio(env=env) as mcp:
            tool_calls(mcp.call, "MCP stdio")
            error_cases(mcp.call, "MCP stdio", in_process=False)
            end_to_end(mcp, "MCP stdio", f"我一个人{d.month}月{d.day}日去成都玩2天，预算3000，一定要去武侯祠")
    failed = [n for n, st, _ in RESULTS if st == "FAIL"]
    warned = [n for n, st, _ in RESULTS if st == "WARN"]
    passed = sum(st == "PASS" for _, st, _ in RESULTS)
    print(f"\n{passed}/{len(RESULTS)} passed, {len(warned)} WARN (service unavailable), {len(failed)} FAIL")
    if failed:
        print("failed:", failed)
    if warned:
        print("warned (rerun later; the services are public and rate-limited):", warned)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
