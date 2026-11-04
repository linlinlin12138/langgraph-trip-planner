"""Semantic cache for confirmed plans.

Lookup = exact metadata filter, then cosine similarity over the request text.

* Metadata key holds every *hard* constraint (city, start date, days, travelers,
  dietary restrictions, budget bucket). Text similarity is blind to small but
  decisive differences ("不吃辣" vs nothing) -- those must never be decided by a
  threshold. Similarity only covers wording and soft preferences.
* Hits are not trusted blindly: the graph re-runs Budget + rule validation on the
  cached plan (and re-plans on errors), see `graph/nodes/cache.py`.
* Eviction: TTL first; then, over capacity, drop the lowest
  score = log1p(hits) - age_weight * days_since_last_use  (frequency x recency).

Storage is a single SQLite table; vectors are small JSON arrays (fine for
hundreds of entries -- a vector DB would be the next step, not needed here).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from trip_agents.schemas import TripRequest

BUDGET_BUCKET = 500  # CNY: budgets within the same 500-yuan bucket are interchangeable


class Embedder(Protocol):
    name: str

    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Offline, deterministic char 1-3 gram hashing embedder (no model, no key).
    Good enough to rank paraphrases of short Chinese requests; swap for a real
    embedding model in production."""

    name = "hashing-ngram-512"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        t = re.sub(r"[\s，。,.!！?？、；;：:]+", "", text)
        vec = [0.0] * self.dim
        for n, w in ((1, 0.5), (2, 1.0), (3, 1.0)):
            for i in range(len(t) - n + 1):
                h = int.from_bytes(hashlib.md5(t[i : i + n].encode()).digest()[:4], "big")
                vec[h % self.dim] += w
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class GeminiEmbedder:
    """Real embedding model via langchain-google-genai (needs GOOGLE_API_KEY)."""

    def __init__(self, model: str = "models/gemini-embedding-001") -> None:  # text-embedding-004 was retired
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        from trip_agents.config import gemini_api_key

        self.name = model
        self._emb = GoogleGenerativeAIEmbeddings(model=model, api_key=gemini_api_key())

    def embed(self, text: str) -> list[float]:
        v = self._emb.embed_query(text)
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))  # inputs are L2-normalised


_LABEL = {
    "history": "历史", "nature": "自然", "museum": "博物馆", "food": "美食", "shopping": "购物", "nightlife": "夜景", "culture": "文化",
    "relaxed": "节奏轻松", "moderate": "节奏适中", "packed": "节奏紧凑",
    "budget": "经济型住宿", "midrange": "舒适型住宿", "luxury": "高档住宿",
}  # fmt: skip


def preference_text(req: TripRequest) -> str:
    """Canonical text of the SOFT side of a request -- what similarity is computed on.
    Raw user wording was tried first: with a lexical embedder a paraphrase scored 0.68
    while a request with different interests scored 0.86, i.e. surface overlap, not
    meaning. Canonicalising after extraction removes wording from the comparison."""
    s = req.soft
    parts = ["兴趣：" + "、".join(_LABEL[i] for i in sorted(s.interests)) if s.interests else "兴趣：不限", _LABEL[s.pace]]
    parts.append(_LABEL[s.hotel_tier] if s.hotel_tier else "住宿不限")
    if s.cuisine_likes:
        parts.append("菜系：" + "、".join(sorted(s.cuisine_likes)))
    if s.notes:
        parts.append("备注：" + s.notes)
    return "；".join(parts)


def meta_key(req: TripRequest) -> str:
    h = req.hard
    diet = ",".join(sorted(h.dietary_restrictions)) or "-"
    must = ",".join(sorted(h.must_visit)) or "-"
    return f"{h.destination}|{h.start_date.isoformat()}|{h.days}d|{h.travelers}p|{diet}|{must}|b{int(h.budget_total // BUDGET_BUCKET)}"


@dataclass
class CacheHit:
    entry_id: int
    similarity: float
    payload: dict[str, Any]
    request_text: str


class SemanticCache:
    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        embedder: Embedder | None = None,
        threshold: float = 0.85,
        ttl_s: float = 7 * 86400,
        max_entries: int = 500,
        age_weight: float = 0.3,
        clock: Any = time.time,
    ) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.embedder = embedder or HashingEmbedder()
        self.threshold, self.ttl_s, self.max_entries, self.age_weight = threshold, ttl_s, max_entries, age_weight
        self.clock = clock
        self._lock = threading.Lock()
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS semantic_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT, meta TEXT NOT NULL, embedder TEXT NOT NULL,
                vector TEXT NOT NULL, request_text TEXT NOT NULL, payload TEXT NOT NULL,
                created_at REAL NOT NULL, last_used_at REAL NOT NULL, hits INTEGER NOT NULL DEFAULT 0, expires_at REAL NOT NULL)"""
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS ix_cache_meta ON semantic_cache(meta, embedder)")
        self.conn.commit()

    # ---------------------------------------------------------------- public API
    def lookup(self, req: TripRequest, text: str) -> CacheHit | None:
        now = self.clock()
        with self._lock:
            self._purge_expired(now)
            rows = self.conn.execute(
                "SELECT id, vector, payload, request_text FROM semantic_cache WHERE meta = ? AND embedder = ?",
                (meta_key(req), self.embedder.name),
            ).fetchall()
            if not rows:
                return None
            q = self.embedder.embed(text)
            best = max(((cosine(q, json.loads(v)), i, p, t) for i, v, p, t in rows), key=lambda x: x[0])
            sim, eid, payload, req_text = best
            if sim < self.threshold:
                return None
            self.conn.execute("UPDATE semantic_cache SET hits = hits + 1, last_used_at = ? WHERE id = ?", (now, eid))
            self.conn.commit()
            return CacheHit(eid, round(sim, 4), json.loads(payload), req_text)

    def put(self, req: TripRequest, text: str, payload: dict[str, Any]) -> int:
        now = self.clock()
        vec = self.embedder.embed(text)
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO semantic_cache (meta, embedder, vector, request_text, payload, created_at, last_used_at, hits, expires_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
                (
                    meta_key(req),
                    self.embedder.name,
                    json.dumps(vec),
                    text,
                    json.dumps(payload, ensure_ascii=False),
                    now,
                    now,
                    now + self.ttl_s,
                ),
            )
            self.conn.commit()
            self._evict(now)
            return int(cur.lastrowid or 0)

    def stats(self) -> dict[str, Any]:
        n, hits = self.conn.execute("SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM semantic_cache").fetchone()
        return {"entries": n, "total_hits": hits, "threshold": self.threshold, "embedder": self.embedder.name}

    # ---------------------------------------------------------------- maintenance
    def _purge_expired(self, now: float) -> None:
        self.conn.execute("DELETE FROM semantic_cache WHERE expires_at <= ?", (now,))

    def score(self, hits: int, last_used_at: float, now: float) -> float:
        return math.log1p(hits) - self.age_weight * (now - last_used_at) / 86400

    def _evict(self, now: float) -> None:
        self._purge_expired(now)
        rows = self.conn.execute("SELECT id, hits, last_used_at FROM semantic_cache").fetchall()
        excess = len(rows) - self.max_entries
        if excess > 0:
            victims = sorted(rows, key=lambda r: self.score(r[1], r[2], now))[:excess]
            self.conn.executemany("DELETE FROM semantic_cache WHERE id = ?", [(r[0],) for r in victims])
        self.conn.commit()
