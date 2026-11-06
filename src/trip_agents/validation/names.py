"""Lenient place-name matching for must-visit places.

The model often writes a fuller or shorter name than the catalog ("成都大熊猫繁育研究基地" vs
"大熊猫繁育研究基地", "秦始皇帝陵博物院" vs "秦始皇帝陵博物院(兵马俑)"). Exact comparison made the graph's
own validator report a must-visit as missing although it was scheduled, which blocked the budget repair
(Gemini dev v4, s161). Names match when, after dropping a city prefix and parentheses, one contains the
other.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_CITY = re.compile(r"^(北京|上海|杭州|成都|西安)市?")
_PAREN = re.compile(r"[(（][^)）]*[)）]")


def _norm(name: str) -> str:
    return _CITY.sub("", _PAREN.sub("", name)).replace(" ", "").strip()


def same_place(a: str, b: str) -> bool:
    x, y = _norm(a), _norm(b)
    if len(x) < 2 or len(y) < 2:
        return a == b
    return x in y or y in x


def matches_any(name: str, names: Iterable[str]) -> bool:
    return any(same_place(name, n) for n in names)
