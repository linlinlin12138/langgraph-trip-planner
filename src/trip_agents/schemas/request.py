"""What the user wants: hard constraints (must hold) vs soft preferences (nice to have)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field

# Only restrictions a restaurant's cuisine determines (the catalog derives them from it). Halal / vegetarian /
# no-pork need per-restaurant certification that open place data does not have, so they are not modelled.
DietaryRestriction = Literal["no_spicy", "no_seafood"]
Pace = Literal["relaxed", "moderate", "packed"]
HotelTier = Literal["budget", "midrange", "luxury"]
Interest = Literal["history", "nature", "museum", "food", "shopping", "nightlife", "culture"]

# Fields the Chat agent must obtain before research can start.
REQUIRED_FIELDS: tuple[str, ...] = ("destination", "start_date", "days", "budget_total")


class HardConstraints(BaseModel):
    """Checked by the rule validator. Violations are errors, not style issues."""

    destination: str
    start_date: date
    days: int = Field(ge=1, le=14)
    budget_total: float = Field(gt=0, description="Total budget for the whole group, CNY")
    travelers: int = Field(default=1, ge=1, le=10)
    dietary_restrictions: list[DietaryRestriction] = Field(default_factory=list)
    must_visit: list[str] = Field(default_factory=list, description="Attraction names that must appear")

    @property
    def end_date(self) -> date:
        return self.start_date + timedelta(days=self.days - 1)

    @property
    def dates(self) -> list[date]:
        return [self.start_date + timedelta(days=i) for i in range(self.days)]


class SoftPreferences(BaseModel):
    """Used for ranking candidates and scored by the LLM judge; never a hard failure."""

    pace: Pace = "moderate"
    hotel_tier: HotelTier | None = None
    interests: list[Interest] = Field(default_factory=list)
    cuisine_likes: list[str] = Field(default_factory=list)
    notes: str = ""


class TripRequest(BaseModel):
    hard: HardConstraints
    soft: SoftPreferences = Field(default_factory=SoftPreferences)


class RequestExtraction(BaseModel):
    """Raw LLM extraction from the conversation. Everything optional: the Chat node
    decides deterministically which required fields are still missing."""

    destination: str | None = None
    start_date: date | None = None
    days: int | None = Field(default=None, ge=1, le=14)
    budget_total: float | None = Field(default=None, gt=0)
    travelers: int | None = Field(default=None, ge=1, le=10)
    dietary_restrictions: list[DietaryRestriction] = Field(default_factory=list)
    must_visit: list[str] = Field(default_factory=list)
    pace: Pace | None = None
    hotel_tier: HotelTier | None = None
    interests: list[Interest] = Field(default_factory=list)
    cuisine_likes: list[str] = Field(default_factory=list)
    notes: str = ""
    # memory hooks
    dietary_none: bool = Field(default=False, description="用户明确表示这次没有饮食限制")
    refers_to_past_trip: bool = Field(default=False, description="用户提到“像上次那样”等，引用过去的行程")
    past_trip_hint: str | None = Field(default=None, description="引用过去行程时提到的线索，如城市名")

    def missing_required(self) -> list[str]:
        return [f for f in REQUIRED_FIELDS if getattr(self, f) is None]

    def to_request(self) -> TripRequest:
        missing = self.missing_required()
        if missing:
            raise ValueError(f"missing required fields: {missing}")
        assert self.destination and self.start_date and self.days and self.budget_total
        return TripRequest(
            hard=HardConstraints(
                destination=self.destination,
                start_date=self.start_date,
                days=self.days,
                budget_total=self.budget_total,
                travelers=self.travelers or 1,
                dietary_restrictions=self.dietary_restrictions,
                must_visit=self.must_visit,
            ),
            soft=SoftPreferences(
                pace=self.pace or "moderate",
                hotel_tier=self.hotel_tier,
                interests=self.interests,
                cuisine_likes=self.cuisine_likes,
                notes=self.notes,
            ),
        )
