"""Per-model token prices used to *estimate* cost.

Prices change; this table deliberately ships only the mock model. Add entries
(USD per 1M tokens) for the models you actually run, after checking the
provider's pricing page, either here or via the PRICING_OVERRIDES env var:

    PRICING_OVERRIDES='{"some-model": [0.10, 0.40]}'

Unknown models return ``None`` so reports show "n/a" instead of a made-up number.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

# model name -> (input USD / 1M tokens, output USD / 1M tokens)
PRICES: dict[str, tuple[float, float]] = {
    "mock-planner-v1": (0.0, 0.0),
}


@lru_cache
def _table() -> dict[str, tuple[float, float]]:
    table = dict(PRICES)
    raw = os.environ.get("PRICING_OVERRIDES")
    if raw:
        table.update({k: (float(v[0]), float(v[1])) for k, v in json.loads(raw).items()})
    return table


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    table = _table()
    price = table.get(model)
    if price is None:  # tolerate dated suffixes, e.g. "foo-2026-01-01" -> "foo"
        price = next((p for name, p in table.items() if model.startswith(name)), None)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
