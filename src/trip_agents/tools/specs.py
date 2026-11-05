"""Tool contracts: name, description and a Pydantic input schema per tool.

The same specs are consumed by the LangGraph Research agent, the MCP server and
`react_from_scratch`, so argument validation behaves identically everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from trip_agents.schemas.request import DietaryRestriction, HotelTier, Interest


class WeatherInput(BaseModel):
    city: str = Field(description="城市名，如 杭州")
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def _range(self) -> WeatherInput:
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be earlier than start_date")
        if (self.end_date - self.start_date).days > 15:
            raise ValueError("date range must be at most 16 days")
        return self


class HotelSearchInput(BaseModel):
    city: str
    tier: HotelTier | None = Field(default=None, description="budget / midrange / luxury")
    max_price_per_night: float | None = Field(default=None, gt=0)
    limit: int = Field(default=5, ge=1, le=10)


class RestaurantSearchInput(BaseModel):
    city: str
    dietary: list[DietaryRestriction] = Field(default_factory=list, description="必须全部满足的饮食限制")
    cuisine: str | None = None
    max_price_per_person: float | None = Field(default=None, gt=0)
    limit: int = Field(default=8, ge=1, le=15)


class AttractionSearchInput(BaseModel):
    city: str
    categories: list[Interest] = Field(default_factory=list, description="为空表示不限类别")
    keyword: str | None = Field(default=None, description="名称关键词，用于查找指定景点（如用户的必去景点）")
    max_ticket_price: float | None = Field(default=None, ge=0)
    limit: int = Field(default=10, ge=1, le=15)


class RouteInput(BaseModel):
    city: str
    origin: str = Field(description="起点：酒店/景点/餐厅的 id 或名称")
    destination: str = Field(description="终点：酒店/景点/餐厅的 id 或名称")
    mode: Literal["walk", "transit", "taxi"] | None = Field(default=None, description="为空则自动选择")


ToolName = Literal["get_weather", "search_hotels", "search_restaurants", "search_attractions", "get_route"]


@dataclass(frozen=True)
class ToolSpec:
    name: ToolName
    description: str
    input_model: type[BaseModel]


TOOL_SPECS: dict[str, ToolSpec] = {
    s.name: s
    for s in [
        ToolSpec("get_weather", "查询城市在日期范围内的逐日天气预报。", WeatherInput),
        ToolSpec("search_hotels", "按档次和价格搜索城市酒店，按评分排序。", HotelSearchInput),
        ToolSpec("search_restaurants", "搜索满足饮食限制的餐厅，可按菜系和人均价格过滤。", RestaurantSearchInput),
        ToolSpec("search_attractions", "按兴趣类别和门票价格搜索景点，返回开放时间和建议游玩时长。", AttractionSearchInput),
        ToolSpec("get_route", "查询两个地点之间的距离、耗时和人均交通费用。", RouteInput),
    ]
}
