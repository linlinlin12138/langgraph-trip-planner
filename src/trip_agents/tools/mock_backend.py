"""Offline backend over the fixed JSON files in data/mock/. Deterministic by construction."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from trip_agents.config import PROJECT_ROOT
from trip_agents.schemas.research import Attraction, Hotel, Restaurant, Route, WeatherDay
from trip_agents.tools.specs import (
    AttractionSearchInput,
    HotelSearchInput,
    RestaurantSearchInput,
    RouteInput,
    WeatherInput,
)

MOCK_DIR = PROJECT_ROOT / "data" / "mock"

# Rough monthly climate (high, low, rain probability) per city, for synthetic weather.
_CLIMATE: dict[str, list[tuple[int, int, float]]] = {
    "北京": [
        (2, -8, 0.05),
        (6, -5, 0.05),
        (13, 1, 0.1),
        (21, 8, 0.15),
        (27, 14, 0.2),
        (31, 19, 0.3),
        (31, 22, 0.5),
        (30, 21, 0.45),
        (26, 15, 0.25),
        (19, 8, 0.15),
        (10, 0, 0.1),
        (3, -6, 0.05),
    ],
    "上海": [
        (8, 2, 0.3),
        (10, 3, 0.35),
        (14, 6, 0.4),
        (20, 11, 0.4),
        (25, 17, 0.4),
        (28, 21, 0.55),
        (32, 26, 0.45),
        (32, 26, 0.45),
        (28, 22, 0.35),
        (23, 16, 0.25),
        (17, 10, 0.25),
        (11, 4, 0.25),
    ],
    "杭州": [
        (8, 1, 0.35),
        (10, 3, 0.4),
        (15, 7, 0.45),
        (21, 12, 0.45),
        (26, 17, 0.45),
        (29, 21, 0.55),
        (33, 25, 0.4),
        (33, 25, 0.4),
        (28, 21, 0.35),
        (23, 15, 0.25),
        (17, 9, 0.3),
        (11, 3, 0.3),
    ],
    "成都": [
        (10, 3, 0.15),
        (12, 5, 0.2),
        (16, 9, 0.25),
        (22, 13, 0.35),
        (26, 18, 0.4),
        (28, 21, 0.5),
        (30, 23, 0.55),
        (30, 23, 0.5),
        (26, 19, 0.5),
        (21, 15, 0.4),
        (16, 10, 0.2),
        (11, 4, 0.1),
    ],
    "西安": [
        (5, -4, 0.05),
        (9, -1, 0.1),
        (15, 5, 0.15),
        (22, 11, 0.2),
        (27, 16, 0.25),
        (32, 21, 0.3),
        (33, 23, 0.4),
        (31, 22, 0.4),
        (26, 17, 0.4),
        (20, 10, 0.3),
        (12, 3, 0.15),
        (6, -2, 0.05),
    ],
}


class ToolInputError(ValueError):
    """Semantically invalid input (e.g. unsupported city); message is shown to the model."""


@lru_cache
def _load_all(mock_dir: Path = MOCK_DIR) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(mock_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[data["city"]] = data
    return out


def supported_cities() -> list[str]:
    return list(_load_all())


def _city(city: str) -> dict[str, Any]:
    data = _load_all()
    key = city.strip().removesuffix("市")
    if key not in data:
        raise ToolInputError(f"不支持的城市 '{city}'，mock 数据仅包含: {', '.join(data)}")
    return data[key]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _stable_unit(*parts: str) -> float:
    h = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32


class MockBackend:
    name = "mock"

    def get_weather(self, inp: WeatherInput) -> list[WeatherDay]:
        city = _city(inp.city)["city"]
        out = []
        d = inp.start_date
        while d <= inp.end_date:
            hi, lo, rain = _CLIMATE[city][d.month - 1]
            u = _stable_unit(city, d.isoformat())
            jitter = (u - 0.5) * 6
            cond = "rain" if u < rain else ("cloudy" if u < rain + 0.3 else "sunny")
            if cond == "rain" and hi < 2:
                cond = "snow"
            out.append(
                WeatherDay(
                    date=d,
                    condition=cond,
                    temp_high=round(hi + jitter, 1),
                    temp_low=round(lo + jitter, 1),
                    precip_prob=round(min(1.0, rain + (0.4 if cond in ("rain", "snow") else 0)), 2),
                )
            )
            d += timedelta(days=1)
        return out

    def search_hotels(self, inp: HotelSearchInput) -> list[Hotel]:
        hotels = [Hotel(**h) for h in _city(inp.city)["hotels"]]
        if inp.tier:
            hotels = [h for h in hotels if h.tier == inp.tier]
        if inp.max_price_per_night:
            hotels = [h for h in hotels if h.price_per_night <= inp.max_price_per_night]
        return sorted(hotels, key=lambda h: (-h.rating, h.price_per_night))[: inp.limit]

    def search_restaurants(self, inp: RestaurantSearchInput) -> list[Restaurant]:
        rs = [Restaurant(**r) for r in _city(inp.city)["restaurants"]]
        rs = [r for r in rs if set(inp.dietary) <= set(r.dietary_tags)]
        if inp.cuisine:
            rs = [r for r in rs if inp.cuisine in r.cuisine]
        if inp.max_price_per_person:
            rs = [r for r in rs if r.avg_price_per_person <= inp.max_price_per_person]
        return sorted(rs, key=lambda r: (-r.rating, r.avg_price_per_person))[: inp.limit]

    def search_attractions(self, inp: AttractionSearchInput) -> list[Attraction]:
        attrs = [Attraction(**a) for a in _city(inp.city)["attractions"]]
        if inp.categories:
            attrs = [a for a in attrs if a.category in inp.categories]
        if inp.max_ticket_price is not None:
            attrs = [a for a in attrs if a.ticket_price <= inp.max_ticket_price]
        return sorted(attrs, key=lambda a: -a.rating)[: inp.limit]

    def get_route(self, inp: RouteInput) -> Route:
        data = _city(inp.city)
        places: dict[str, dict[str, Any]] = {p["id"]: p for k in ("hotels", "restaurants", "attractions") for p in data[k]}
        by_name: dict[str, dict[str, Any]] = {p["name"]: p for p in places.values()}

        def find(q: str) -> dict[str, Any]:
            if q in places:
                return places[q]
            if q in by_name:
                return by_name[q]
            hits = [p for n, p in by_name.items() if q in n]
            if len(hits) == 1:
                return hits[0]
            raise ToolInputError(f"找不到地点 '{q}'，请使用搜索结果中的 id 或完整名称")

        a, b = find(inp.origin), find(inp.destination)
        km = _haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]) * 1.3  # road factor
        mode = inp.mode or ("walk" if km < 1.5 else "transit" if km < 25 else "taxi")
        if mode == "walk":
            minutes, cost = km / 4.5 * 60, 0.0
        elif mode == "transit":
            minutes, cost = km / 22 * 60 + 10, float(min(10, 3 + int(km / 6)))
        else:
            minutes, cost = km / 35 * 60 + 5, round(13 + 2.6 * max(0.0, km - 3), 1)
        return Route(
            origin=a["name"], destination=b["name"], mode=mode, distance_km=round(km, 1), duration_min=round(minutes), est_cost=cost
        )
