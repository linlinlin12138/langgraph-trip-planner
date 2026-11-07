"""Lenient place-name matching for must-visit places.

The model often writes a fuller or shorter name than the catalog ("成都大熊猫繁育研究基地" vs
"大熊猫繁育研究基地", "秦始皇帝陵博物院" vs "秦始皇帝陵博物院(兵马俑)"). Exact comparison made the graph's
own validator report a must-visit as missing although it was scheduled, which blocked the budget repair
(Gemini dev v4, s161).

Two names match when one of their forms contains the other. A name's forms are the name with the city
prefix (北京/上海/杭州/成都/西安, optional 市) and any parenthesised part removed, plus each parenthesised
part on its own -- the catalog writes aliases that way ("秦始皇帝陵博物院(兵马俑)"). Forms shorter than two
characters only match exactly. So a short name that is a contiguous part of the catalog name counts
("故宫" ~ "故宫博物院"), an alias in parentheses counts ("兵马俑"), a nickname that is neither does not
("熊猫基地" !~ "大熊猫繁育研究基地").
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_CITY = re.compile(r"^(北京|上海|杭州|成都|西安)市?")
_PAREN = re.compile(r"[(（][^)）]*[)）]")


def _norm(name: str) -> str:
    return _CITY.sub("", _PAREN.sub("", name)).replace(" ", "").strip()


def _forms(name: str) -> list[str]:
    aliases = [m.strip(" ()（）") for m in _PAREN.findall(name)]
    return [f for f in (_norm(name), *(a.replace(" ", "") for a in aliases)) if f]


def same_place(a: str, b: str) -> bool:
    if a == b:
        return True
    for x in _forms(a):
        for y in _forms(b):
            if len(x) >= 2 and len(y) >= 2 and (x in y or y in x):
                return True
    return False


def matches_any(name: str, names: Iterable[str]) -> bool:
    return any(same_place(name, n) for n in names)
