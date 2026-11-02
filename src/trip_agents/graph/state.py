"""Shared graph state, grouped by the node that writes each field.

Convention: a node only writes the fields in its own group. Fields that
accumulate across nodes / rounds use a reducer (`add_messages`, `operator.add`);
everything else is last-write-wins.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from trip_agents.schemas import (
    BudgetReport,
    FinalPlan,
    Itinerary,
    RequestExtraction,
    ResearchBundle,
    Selection,
    SkippedItem,
    TripRequest,
    ValidationIssue,
)

Status = Literal["clarifying", "researching", "awaiting_selection", "planning", "done", "confirmed"]


class ToolCallRecord(TypedDict):
    name: str
    args: dict[str, Any]
    ok: bool
    error: str | None
    latency_ms: float
    step: int
    duplicate: bool


class TripState(TypedDict, total=False):
    # ---- input / conversation (user + chat node) -------------------------------------
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    status: Status

    # ---- chat ----------------------------------------------------------------
    extraction: RequestExtraction  # running partial extraction, refined every turn
    missing_fields: list[str]
    clarify_rounds: int
    request: TripRequest  # set once all required fields are known
    memory_defaults: dict[str, str]  # fields filled from long-term memory -> note shown to the user
    defaults_pending: bool  # waiting for the user to confirm memory defaults
    defaults_confirmed: bool

    # ---- research --------------------------------------------------------------
    research: ResearchBundle
    tool_calls: Annotated[list[ToolCallRecord], operator.add]

    # ---- human_select ------------------------------------------------------------
    selection: Selection

    # ---- itinerary ------------------------------------------------------------
    itinerary: Itinerary
    revision: int  # 0 = first draft
    skipped_items: Annotated[list[SkippedItem], operator.add]
    adjustments: Annotated[list[str], operator.add]  # changes made on the user's behalf, surfaced in the final message

    # ---- budget ----------------------------------------------------------------
    budget: BudgetReport

    # ---- validate ------------------------------------------------------------------
    issues: list[ValidationIssue]  # latest round only (drives routing)
    issue_history: Annotated[list[list[ValidationIssue]], operator.add]

    # ---- finalize ----------------------------------------------------------------------
    final: FinalPlan
    memory_updates: list[str]  # written by confirm
