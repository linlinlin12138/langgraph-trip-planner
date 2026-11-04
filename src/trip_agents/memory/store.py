"""Memory access on top of LangGraph's BaseStore (SqliteStore in practice).

Namespaces
    ("users", user_id)            key "profile"     -> UserProfile
    ("users", user_id, "trips")   key <thread_id>   -> TripSummary   (at most `max_trips`, oldest evicted)

Retrieval for "像上次那样" is recency + destination keyword, not embeddings: with
<= 10 short records per user, exact metadata beats vector search on both cost
and predictability (see docs/DESIGN.md).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore

from trip_agents.memory.profile import TripSummary, UserProfile

MAX_TRIPS = 10


def sqlite_store(path: str | Path) -> SqliteStore:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # autocommit connection: SqliteStore manages its own transactions
    store = SqliteStore(sqlite3.connect(str(path), check_same_thread=False, isolation_level=None))
    store.setup()
    return store


class MemoryManager:
    def __init__(self, store: BaseStore, max_trips: int = MAX_TRIPS) -> None:
        self.store = store
        self.max_trips = max_trips

    # ---------------------------------------------------------------- profile
    def get_profile(self, user_id: str) -> UserProfile | None:
        item = self.store.get(("users", user_id), "profile")
        return UserProfile.model_validate(item.value) if item else None

    def put_profile(self, profile: UserProfile) -> None:
        self.store.put(("users", profile.user_id), "profile", profile.model_dump(mode="json"))

    def clear(self, user_id: str) -> None:
        """Forget everything about a user (profile + trips)."""
        self.store.delete(("users", user_id), "profile")
        for t in self.list_trips(user_id):
            self.store.delete(("users", user_id, "trips"), t.thread_id)

    # ---------------------------------------------------------------- trips
    def list_trips(self, user_id: str) -> list[TripSummary]:
        items = self.store.search(("users", user_id, "trips"), limit=1000)
        trips = [TripSummary.model_validate(i.value) for i in items if i.namespace == ("users", user_id, "trips")]
        return sorted(trips, key=lambda t: t.created_at, reverse=True)

    def add_trip(self, user_id: str, trip: TripSummary) -> list[str]:
        """Insert (or replace) a trip; returns thread ids evicted by the cap."""
        self.store.put(("users", user_id, "trips"), trip.thread_id, trip.model_dump(mode="json"))
        evicted = [t.thread_id for t in self.list_trips(user_id)[self.max_trips :]]
        for tid in evicted:
            self.store.delete(("users", user_id, "trips"), tid)
        return evicted

    def find_trip(self, user_id: str, hint: str | None = None) -> TripSummary | None:
        trips = self.list_trips(user_id)
        if hint:
            for t in trips:
                if t.destination in hint or hint in t.destination:
                    return t
        return trips[0] if trips else None
