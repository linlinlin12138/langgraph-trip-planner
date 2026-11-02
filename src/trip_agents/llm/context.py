"""Agents pass structured context to the model inside a `<context>` JSON block.

Real models read it as part of the prompt; the mock model parses it to produce
deterministic answers. Keeping it machine-readable is also what makes prompts
diffable and easy to log.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

_CTX_RE = re.compile(r"<context>\s*(\{.*?\})\s*</context>", re.DOTALL)


def _default(o: Any) -> Any:
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    return str(o)


def render_context(data: dict[str, Any]) -> str:
    return "<context>\n" + json.dumps(data, ensure_ascii=False, default=_default, indent=1) + "\n</context>"


def extract_context(messages: Sequence[BaseMessage]) -> dict[str, Any]:
    """Return the most recent context block found in the messages (empty dict if none)."""
    for m in reversed(messages):
        content = m.content if isinstance(m.content, str) else ""
        found = _CTX_RE.findall(content)
        if found:
            parsed: dict[str, Any] = json.loads(found[-1])
            return parsed
    return {}
