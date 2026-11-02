"""Trajectory recording through LangChain callbacks -- zero changes to node code.

Attach with ``graph.invoke(inp, {"callbacks": [tracer]})``. What we capture and from where:

* LLM calls      on_chat_model_start / on_llm_end   -> tokens (usage_metadata), latency, model, node
* node spans     on_chain_start / on_chain_end where run name == langgraph_node
* tool calls     on_chain_end of the Research `tools` node (its output carries ToolCallRecords)
* retries        on_custom_event("structured_output") emitted by llm/structured.py
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from trip_agents.llm.structured import STRUCTURED_EVENT
from trip_agents.observability.pricing import estimate_cost


@dataclass
class LLMCall:
    node: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    has_tool_calls: bool
    error: str | None = None


@dataclass
class NodeSpan:
    node: str
    parent: str | None
    latency_ms: float
    error: str | None = None


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    ok: bool
    error: str | None
    latency_ms: float
    duplicate: bool
    step: int


@dataclass
class StructuredEvent:
    stage: str
    item: str
    attempt: int
    ok: bool
    error: str | None = None


@dataclass
class Trace:
    llm_calls: list[LLMCall] = field(default_factory=list)
    nodes: list[NodeSpan] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    structured: list[StructuredEvent] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    wall_ms: float = 0.0

    # ------------------------------------------------------------- aggregates
    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.llm_calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)

    @property
    def cost_usd(self) -> float | None:
        costs = [estimate_cost(c.model, c.input_tokens, c.output_tokens) for c in self.llm_calls]
        return None if any(c is None for c in costs) else round(sum(c for c in costs if c is not None), 6)

    @property
    def parse_failures(self) -> int:
        return sum(not e.ok for e in self.structured)

    @property
    def skipped(self) -> int:
        """Items that exhausted all retries (last attempt failed)."""
        last: dict[tuple[str, str], StructuredEvent] = {}
        for e in self.structured:
            last[(e.stage, e.item)] = e
        return sum(not e.ok for e in last.values())

    def node_latency(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for n in self.nodes:
            if n.parent is None:
                out.setdefault(n.node, []).append(n.latency_ms)
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "llm_calls": len(self.llm_calls),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "tool_calls": len(self.tool_calls),
            "tool_errors": sum(not t.ok for t in self.tool_calls),
            "tool_duplicates": sum(t.duplicate for t in self.tool_calls),
            "parse_failures": self.parse_failures,
            "skipped_items": self.skipped,
            "wall_ms": round(self.wall_ms, 1),
            "node_ms": {k: round(sum(v), 1) for k, v in self.node_latency().items()},
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "llm_calls": [asdict(c) for c in self.llm_calls],
            "nodes": [asdict(n) for n in self.nodes],
            "tool_calls": [asdict(t) for t in self.tool_calls],
            "structured": [asdict(e) for e in self.structured],
        }


def _node_of(metadata: dict[str, Any] | None) -> tuple[str, str | None]:
    """(node, parent) from LangGraph run metadata. Subgraph nodes get their parent node."""
    md = metadata or {}
    node = str(md.get("langgraph_node", ""))
    ns = str(md.get("langgraph_checkpoint_ns") or md.get("checkpoint_ns") or "")
    first = ns.split("|")[0].split(":")[0] if ns else ""
    parent = first if first and first != node else None
    return node, parent


class TraceRecorder(BaseCallbackHandler):
    """One recorder per run. Thread-safe (LangGraph may run sibling nodes in threads)."""

    raise_error = False

    def __init__(self) -> None:
        self.trace = Trace()
        self._lock = threading.Lock()
        self._llm_start: dict[UUID, tuple[float, str, str]] = {}
        self._chain_start: dict[UUID, tuple[float, str, str | None]] = {}
        self._t0 = time.perf_counter()

    def finish(self) -> Trace:
        self.trace.wall_ms = (time.perf_counter() - self._t0) * 1000
        return self.trace

    # ------------------------------------------------------------------ LLM
    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: Any, *, run_id: UUID, metadata: dict[str, Any] | None = None, **kw: Any
    ) -> None:
        node, parent = _node_of(metadata)
        model = str((metadata or {}).get("ls_model_name") or (serialized or {}).get("name") or "unknown")
        with self._lock:
            self._llm_start[run_id] = (time.perf_counter(), f"{parent}.{node}" if parent else node, model)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kw: Any) -> None:
        with self._lock:
            t0, node, model = self._llm_start.pop(run_id, (time.perf_counter(), "?", "unknown"))
        gen = response.generations[0][0] if response.generations and response.generations[0] else None
        msg = getattr(gen, "message", None)
        usage = getattr(msg, "usage_metadata", None) or {}
        meta_model = (getattr(msg, "response_metadata", None) or {}).get("model_name")
        call = LLMCall(
            node=node,
            model=str(meta_model or model),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            has_tool_calls=bool(getattr(msg, "tool_calls", None)),
        )
        with self._lock:
            self.trace.llm_calls.append(call)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kw: Any) -> None:
        with self._lock:
            t0, node, model = self._llm_start.pop(run_id, (time.perf_counter(), "?", "unknown"))
            self.trace.llm_calls.append(LLMCall(node, model, 0, 0, round((time.perf_counter() - t0) * 1000, 2), False, repr(error)))

    # ------------------------------------------------------------------ nodes
    def on_chain_start(
        self, serialized: dict[str, Any], inputs: Any, *, run_id: UUID, metadata: dict[str, Any] | None = None, **kw: Any
    ) -> None:
        node, parent = _node_of(metadata)
        if node and kw.get("name") == node:
            with self._lock:
                self._chain_start[run_id] = (time.perf_counter(), node, parent)

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kw: Any) -> None:
        with self._lock:
            start = self._chain_start.pop(run_id, None)
        if start is None:
            return
        t0, node, parent = start
        with self._lock:
            self.trace.nodes.append(NodeSpan(node, parent, round((time.perf_counter() - t0) * 1000, 2)))
            if node == "tools" and isinstance(outputs, dict):
                for r in outputs.get("records", []):
                    self.trace.tool_calls.append(
                        ToolCall(r["name"], r["args"], r["ok"], r["error"], r["latency_ms"], r["duplicate"], r["step"])
                    )

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kw: Any) -> None:
        with self._lock:
            start = self._chain_start.pop(run_id, None)
            if start is not None:
                t0, node, parent = start
                # GraphInterrupt surfaces as an error on the interrupting node; record it as a span, not a failure.
                err = None if type(error).__name__ == "GraphInterrupt" else repr(error)
                self.trace.nodes.append(NodeSpan(node, parent, round((time.perf_counter() - t0) * 1000, 2), err))

    # ------------------------------------------------------------------ custom
    def on_custom_event(self, name: str, data: Any, *, run_id: UUID, **kw: Any) -> None:
        if name == STRUCTURED_EVENT and isinstance(data, dict):
            with self._lock:
                self.trace.structured.append(
                    StructuredEvent(
                        data.get("stage", ""), data.get("item", ""), int(data.get("attempt", 0)), bool(data.get("ok")), data.get("error")
                    )
                )
