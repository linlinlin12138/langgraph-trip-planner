"""Real backend parsing, against canned HTTP responses (no network in CI).

The payloads mirror the documented response shapes of Open-Meteo and OSRM;
live behaviour is only exercised when TOOL_BACKEND=real is used manually.
"""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from trip_agents.tools.executor import execute_tool
from trip_agents.tools.real_backend import RealBackend, wmo_to_condition


def handler(request: httpx.Request) -> httpx.Response:
    host, path = request.url.host, request.url.path
    if host == "geocoding-api.open-meteo.com":
        name = request.url.params["name"]
        if "火星" in name:
            return httpx.Response(200, json={"generationtime_ms": 0.1})
        return httpx.Response(200, json={"results": [{"name": name, "latitude": 30.29, "longitude": 120.16}]})
    if host == "api.open-meteo.com":
        assert request.url.params["start_date"] == "2026-11-05"
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": ["2026-11-05", "2026-11-06"],
                    "weather_code": [0, 61],
                    "temperature_2m_max": [21.3, 17.0],
                    "temperature_2m_min": [12.1, 11.4],
                    "precipitation_probability_max": [5, 80],
                }
            },
        )
    if host == "router.project-osrm.org":
        assert path.startswith("/route/v1/driving/")
        return httpx.Response(200, json={"code": "Ok", "routes": [{"distance": 5300.0, "duration": 720.0}]})
    return httpx.Response(404)


@pytest.fixture
def backend() -> RealBackend:
    return RealBackend(client=httpx.Client(transport=httpx.MockTransport(handler)), today=date(2026, 11, 2))


def test_weather_parsing(backend: RealBackend) -> None:
    res = execute_tool("get_weather", {"city": "杭州", "start_date": "2026-11-05", "end_date": "2026-11-06"}, backend)
    assert res.ok
    assert [d["condition"] for d in res.data] == ["sunny", "rain"]
    assert res.data[1]["precip_prob"] == 0.8 and res.data[0]["temp_high"] == 21.3


def test_weather_beyond_horizon_is_an_error_not_a_guess(backend: RealBackend) -> None:
    res = execute_tool("get_weather", {"city": "杭州", "start_date": "2026-12-20", "end_date": "2026-12-22"}, backend)
    assert not res.ok and "16 天" in (res.error or "")


def test_route_uses_catalog_coordinates_and_osrm(backend: RealBackend) -> None:
    res = execute_tool("get_route", {"city": "杭州", "origin": "hz-a00", "destination": "灵隐寺", "mode": "taxi"}, backend)
    assert res.ok and res.data["origin"] == "西湖" and res.data["distance_km"] == 5.3
    assert res.data["duration_min"] == 17  # 12 min drive + 5 min pickup


def test_unknown_place_and_fallback_tools(backend: RealBackend) -> None:
    assert "无法定位" in (execute_tool("get_route", {"city": "杭州", "origin": "火星基地", "destination": "西湖"}, backend).error or "")
    assert execute_tool("search_hotels", {"city": "杭州"}, backend).ok  # catalog fallback


@pytest.mark.parametrize(
    "code, cond", [(0, "sunny"), (1, "sunny"), (3, "cloudy"), (45, "cloudy"), (63, "rain"), (95, "rain"), (73, "snow")]
)
def test_wmo_codes(code: int, cond: str) -> None:
    assert wmo_to_condition(code) == cond
