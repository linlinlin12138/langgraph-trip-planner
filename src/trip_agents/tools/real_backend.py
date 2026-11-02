"""Real-API backend for two tools; the other three fall back to the mock catalog.

* get_weather -> Open-Meteo (geocoding + daily forecast), no API key.
  Forecasts only cover ~16 days ahead; beyond that the tool returns an error
  telling the agent so, rather than inventing weather.
* get_route   -> OSRM public demo server (driving), no API key. Coordinates come
  from the catalog when the place is known, else from Open-Meteo geocoding.
  Walking time is derived from the OSRM distance; transit has no free API and
  is estimated (and labelled as such).

Enable with TOOL_BACKEND=real. Both services are public best-effort endpoints:
fine for a demo, not for production traffic.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx

from trip_agents.schemas.research import Attraction, Hotel, Restaurant, Route, WeatherDay
from trip_agents.tools.mock_backend import MockBackend, ToolInputError, _load_all
from trip_agents.tools.specs import AttractionSearchInput, HotelSearchInput, RestaurantSearchInput, RouteInput, WeatherInput

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
FORECAST_HORIZON_DAYS = 16


def wmo_to_condition(code: int) -> str:
    """WMO weather interpretation codes -> our 4 buckets."""
    if code in (71, 73, 75, 77, 85, 86):
        return "snow"
    if code >= 51:
        return "rain"
    if code >= 2:
        return "cloudy"
    return "sunny"


class RealBackend:
    name = "real"

    def __init__(self, client: httpx.Client | None = None, today: date | None = None) -> None:
        self.http = client or httpx.Client(timeout=10.0, headers={"User-Agent": "trip-agents-demo/0.1"})
        self.today = today
        self._fallback = MockBackend()

    # ------------------------------------------------------------------ helpers
    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        r = self.http.get(url, params=params)
        r.raise_for_status()
        return r.json()

    def geocode(self, name: str) -> tuple[float, float]:
        data = self._get(GEOCODE_URL, {"name": name, "count": 1, "language": "zh", "format": "json"})
        results = data.get("results") or []
        if not results:
            raise ToolInputError(f"无法定位地点 '{name}'")
        return float(results[0]["latitude"]), float(results[0]["longitude"])

    def _place(self, city: str, q: str) -> tuple[str, float, float]:
        data = _load_all().get(city.removesuffix("市"))
        if data:
            for kind in ("hotels", "restaurants", "attractions"):
                for p in data[kind]:
                    if q in (p["id"], p["name"]):
                        return p["name"], p["lat"], p["lon"]
        lat, lon = self.geocode(f"{city}{q}")
        return q, lat, lon

    # ------------------------------------------------------------------ real tools
    def get_weather(self, inp: WeatherInput) -> list[WeatherDay]:
        today = self.today or date.today()
        if (inp.end_date - today).days >= FORECAST_HORIZON_DAYS:
            raise ToolInputError(
                f"天气预报只覆盖未来 {FORECAST_HORIZON_DAYS} 天（从 {today.isoformat()} 起算），请基于季节常识规划或缩短日期范围"
            )
        lat, lon = self.geocode(inp.city)
        data = self._get(FORECAST_URL, {
            "latitude": lat, "longitude": lon, "timezone": "auto",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "start_date": inp.start_date.isoformat(), "end_date": inp.end_date.isoformat(),
        })  # fmt: skip
        d = data["daily"]
        rows = zip(
            d["time"], d["weather_code"], d["temperature_2m_max"], d["temperature_2m_min"], d["precipitation_probability_max"], strict=True
        )
        return [
            WeatherDay.model_validate(
                {"date": t, "condition": wmo_to_condition(int(c)), "temp_high": hi, "temp_low": lo, "precip_prob": float(p or 0) / 100}
            )
            for t, c, hi, lo, p in rows
        ]

    def get_route(self, inp: RouteInput) -> Route:
        a_name, lat1, lon1 = self._place(inp.city, inp.origin)
        b_name, lat2, lon2 = self._place(inp.city, inp.destination)
        data = self._get(OSRM_URL.format(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2), {"overview": "false"})
        if data.get("code") != "Ok" or not data.get("routes"):
            raise ToolInputError(f"路线服务未返回结果: {data.get('code')}")
        km = data["routes"][0]["distance"] / 1000
        drive_min = data["routes"][0]["duration"] / 60
        mode = inp.mode or ("walk" if km < 1.5 else "transit" if km < 25 else "taxi")
        if mode == "walk":
            minutes, cost = km / 4.5 * 60, 0.0
        elif mode == "transit":  # estimate: no free transit routing API
            minutes, cost = drive_min * 1.6 + 10, float(min(10, 3 + int(km / 6)))
        else:
            minutes, cost = drive_min + 5, round(13 + 2.6 * max(0.0, km - 3), 1)
        return Route(origin=a_name, destination=b_name, mode=mode, distance_km=round(km, 1), duration_min=round(minutes), est_cost=cost)

    # ------------------------------------------------------------------ catalog fallback
    def search_hotels(self, inp: HotelSearchInput) -> list[Hotel]:
        return self._fallback.search_hotels(inp)

    def search_restaurants(self, inp: RestaurantSearchInput) -> list[Restaurant]:
        return self._fallback.search_restaurants(inp)

    def search_attractions(self, inp: AttractionSearchInput) -> list[Attraction]:
        return self._fallback.search_attractions(inp)
