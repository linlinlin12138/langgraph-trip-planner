"""Itinerary, budget and validation outputs."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from trip_agents.schemas._types import LocalTime

ActivityKind = Literal["attraction", "meal", "transit", "rest"]


class Activity(BaseModel):
    start_time: LocalTime
    end_time: LocalTime
    kind: ActivityKind
    title: str = Field(min_length=1)
    ref_id: str | None = Field(default=None, description="Id of the attraction / restaurant; required for attraction and meal")
    est_cost: float = Field(ge=0, description="Total for the whole group, CNY")
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> Activity:
        if self.end_time <= self.start_time:
            raise ValueError(f"'{self.title}': end_time {self.end_time} must be after start_time {self.start_time}")
        if self.kind in ("attraction", "meal") and not self.ref_id:
            raise ValueError(f"'{self.title}': ref_id is required for kind={self.kind}")
        return self


class DayPlan(BaseModel):
    day_index: int = Field(ge=0)
    date: date
    theme: str = ""
    activities: list[Activity] = Field(min_length=1)

    @property
    def cost(self) -> float:
        return sum(a.est_cost for a in self.activities)


class Itinerary(BaseModel):
    hotel_id: str | None
    days: list[DayPlan]
    summary: str = ""


class BudgetLine(BaseModel):
    category: Literal["hotel", "meals", "tickets", "transit", "other"]
    amount: float
    detail: str = ""


class BudgetAdvice(BaseModel):
    """LLM output, only requested when over budget."""

    suggestions: list[str] = Field(min_length=1, max_length=5)


class BudgetReport(BaseModel):
    lines: list[BudgetLine]
    total: float
    budget: float
    min_feasible_cost: float = Field(description="Cheapest plan possible with researched candidates")
    suggestions: list[str] = Field(default_factory=list)
    # Audit of the model's own cost estimates (total above always uses catalog prices, see graph/costs.py).
    model_refs_total: float | None = Field(default=None, description="Model est_cost summed over catalog-resolved attractions/meals")
    catalog_refs_total: float | None = Field(default=None, description="Catalog price of the same activities")
    model_plan_total: float | None = Field(default=None, description="Lodging + the model's est_cost of every activity")

    @property
    def over_by(self) -> float:
        return max(0.0, self.total - self.budget)

    @property
    def feasible(self) -> bool:
        return self.min_feasible_cost <= self.budget


class ValidationIssue(BaseModel):
    code: str
    message: str
    severity: Literal["error", "warning"] = "error"
    day_index: int | None = None


class SkippedItem(BaseModel):
    """Something the structured-output pipeline gave up on after max retries."""

    stage: str
    item: str
    errors: list[str]


class FinalPlan(BaseModel):
    itinerary: Itinerary
    budget: BudgetReport
    passed_validation: bool
    feasible: bool
    remaining_issues: list[ValidationIssue] = Field(default_factory=list)
    revisions: int = 0
    message: str = ""
