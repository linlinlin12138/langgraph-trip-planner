"""Domain objects returned by tools and collected by the Research agent."""

from __future__ import annotations

from datetime import date, time
from typing import Literal

from pydantic import BaseModel, Field

from trip_agents.schemas.request import DietaryRestriction, HotelTier, Interest


class Hotel(BaseModel):
    id: str
    name: str
    city: str
    tier: HotelTier
    price_per_night: float = Field(description="Per room, CNY")
    rating: float
    area: str
    lat: float
    lon: float


class Restaurant(BaseModel):
    id: str
    name: str
    city: str
    cuisine: str
    avg_price_per_person: float
    rating: float
    dietary_tags: list[DietaryRestriction] = Field(default_factory=list, description="Restrictions this restaurant can satisfy")
    area: str
    lat: float
    lon: float


class Attraction(BaseModel):
    id: str
    name: str
    city: str
    category: Interest
    ticket_price: float = Field(description="Per person, CNY")
    recommended_hours: float
    open_time: time
    close_time: time
    rating: float
    area: str
    lat: float
    lon: float


class WeatherDay(BaseModel):
    date: date
    condition: Literal["sunny", "cloudy", "rain", "snow"]
    temp_high: float
    temp_low: float
    precip_prob: float = Field(ge=0, le=1)


class Route(BaseModel):
    origin: str
    destination: str
    mode: Literal["walk", "transit", "taxi"]
    distance_km: float
    duration_min: float
    est_cost: float = Field(description="Per person, CNY")


class ResearchBundle(BaseModel):
    """Everything Research found. Built deterministically from tool results,
    not from the model's free-text summary, so nothing can be hallucinated in."""

    hotels: list[Hotel] = Field(default_factory=list)
    restaurants: list[Restaurant] = Field(default_factory=list)
    attractions: list[Attraction] = Field(default_factory=list)
    weather: list[WeatherDay] = Field(default_factory=list)
    routes: list[Route] = Field(default_factory=list)
    notes: str = ""

    def hotel(self, hid: str) -> Hotel | None:
        return next((h for h in self.hotels if h.id == hid), None)

    def restaurant(self, rid: str) -> Restaurant | None:
        return next((r for r in self.restaurants if r.id == rid), None)

    def attraction(self, aid: str) -> Attraction | None:
        return next((a for a in self.attractions if a.id == aid), None)


class Selection(BaseModel):
    """The user's human-in-the-loop choice after Research."""

    hotel_id: str | None = None
    restaurant_ids: list[str] = Field(default_factory=list)
    attraction_ids: list[str] = Field(default_factory=list)
