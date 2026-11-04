"""Shared field types."""

from __future__ import annotations

from datetime import time
from typing import Annotated

from pydantic import AfterValidator


def _naive(t: time) -> time:
    return t.replace(tzinfo=None)


# Clock times in a trip are local wall-clock times. Real models sometimes emit "09:00:00Z" or
# "09:00+08:00"; aware and naive times cannot be compared, so the offset is dropped.
LocalTime = Annotated[time, AfterValidator(_naive)]
