from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, time
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

from trip_agents.graph.builder import build_graph, sqlite_checkpointer
from trip_agents.graph.deps import AgentDeps, GraphOptions, LocalToolExecutor
from trip_agents.llm.mock import MockChatModel
from trip_agents.schemas import (
    Activity,
    Attraction,
    DayPlan,
    HardConstraints,
    Hotel,
    Itinerary,
    ResearchBundle,
    Restaurant,
    SoftPreferences,
    TripRequest,
)

TODAY = date(2026, 11, 2)
REQUEST_TEXT = "我和女朋友想11月20日去杭州玩3天，预算6000，不吃辣，喜欢自然和美食，一定要去灵隐寺"


class ScriptedChatModel(BaseChatModel):
    """Returns pre-baked AIMessages in order (repeats the last one). For tool-loop tests."""

    script: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any) -> ChatResult:
        msg = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=msg.model_copy())])

    def bind_tools(self, tools: Sequence[Any], **kw: Any) -> Runnable[Any, AIMessage]:  # type: ignore[override]
        return self  # type: ignore[return-value]


@pytest.fixture
def make_graph(tmp_path: Path) -> Callable[..., Any]:
    def _make(faults: set[str] | None = None, options: GraphOptions | None = None, db: str = "cp.sqlite") -> Any:
        deps = AgentDeps(model=MockChatModel(faults=set(faults or ())), tools=LocalToolExecutor(), today=TODAY)
        return build_graph(deps, options, checkpointer=sqlite_checkpointer(tmp_path / db))

    return _make


# ----------------------------------------------------------------------------- tiny world
@pytest.fixture
def world() -> tuple[TripRequest, ResearchBundle]:
    req = TripRequest(
        hard=HardConstraints(
            destination="杭州",
            start_date=date(2026, 11, 20),
            days=2,
            budget_total=3000,
            travelers=2,
            dietary_restrictions=["no_spicy"],
            must_visit=["灵隐寺"],
        ),
        soft=SoftPreferences(pace="relaxed"),
    )
    common = {"city": "杭州", "area": "西湖", "lat": 30.24, "lon": 120.15}
    rb = ResearchBundle(
        hotels=[Hotel(id="h1", name="H1", tier="midrange", price_per_night=500, rating=4.5, **common)],
        restaurants=[
            Restaurant(id="r1", name="清淡馆", cuisine="杭帮菜", avg_price_per_person=80, rating=4.5, dietary_tags=["no_spicy"], **common),
            Restaurant(id="r2", name="辣味馆", cuisine="川菜", avg_price_per_person=60, rating=4.4, dietary_tags=[], **common),
        ],
        attractions=[
            Attraction(
                id="a1",
                name="灵隐寺",
                category="culture",
                ticket_price=75,
                recommended_hours=2.5,
                open_time=time(7),
                close_time=time(17, 30),
                rating=4.8,
                **common,
            ),
            Attraction(
                id="a2",
                name="西湖",
                category="nature",
                ticket_price=0,
                recommended_hours=3,
                open_time=time(6),
                close_time=time(23),
                rating=4.9,
                **common,
            ),
        ],
    )
    return req, rb


def act(start: str, end: str, kind: str, ref: str | None, cost: float = 0, title: str = "x") -> Activity:
    return Activity(
        start_time=time.fromisoformat(start),
        end_time=time.fromisoformat(end),
        kind=kind,  # type: ignore[arg-type]
        title=title,
        ref_id=ref,
        est_cost=cost,
    )


@pytest.fixture
def good_itinerary() -> Itinerary:
    return Itinerary(
        hotel_id="h1",
        days=[
            DayPlan(
                day_index=0,
                date=date(2026, 11, 20),
                activities=[
                    act("09:00", "11:30", "attraction", "a1", 150),
                    act("12:00", "13:00", "meal", "r1", 160),
                    act("18:00", "19:00", "meal", "r1", 160),
                ],
            ),
            DayPlan(
                day_index=1,
                date=date(2026, 11, 21),
                activities=[act("09:00", "12:00", "attraction", "a2", 0), act("12:30", "13:30", "meal", "r1", 160)],
            ),
        ],
    )
