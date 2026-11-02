"""Synchronous `ToolExecutor` backed by an MCP client session.

The graph is synchronous; the MCP client is async. We run one event loop in a
daemon thread that owns the session for its whole life, and bridge calls with
`run_coroutine_threadsafe`. Tools are *discovered* via `list_tools` -- the agent
never imports tool code, it only sees names, descriptions and JSON Schemas.

    with McpToolExecutor.stdio() as tools:          # spawns `python -m trip_agents.tools.mcp_server`
        deps = AgentDeps(model=..., tools=tools)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import threading
import time
from collections.abc import Coroutine
from concurrent.futures import Future
from contextlib import AsyncExitStack
from typing import Any, TypeVar

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.server.lowlevel import Server

from trip_agents.tools.executor import ToolResult

T = TypeVar("T")


class McpToolExecutor:
    def __init__(self, server: Server[Any] | StdioServerParameters, *, timeout_s: float = 60.0) -> None:
        self._target = server
        self._timeout = timeout_s
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-client", daemon=True)
        self._thread.start()
        self._stack: AsyncExitStack | None = None
        self._client: Any = None
        self._schemas: list[dict[str, Any]] = []
        self._run(self._connect())

    # ---------------------------------------------------------------- construction
    @classmethod
    def stdio(cls, env: dict[str, str] | None = None) -> McpToolExecutor:
        params = StdioServerParameters(command=sys.executable, args=["-m", "trip_agents.tools.mcp_server"], env=env)
        return cls(params)

    @classmethod
    def in_process(cls, server: Server[Any]) -> McpToolExecutor:
        """Same protocol path without a subprocess (tests)."""
        return cls(server)

    # ---------------------------------------------------------------- plumbing
    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        fut: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=self._timeout)

    async def _connect(self) -> None:
        self._stack = AsyncExitStack()
        self._client = await self._stack.enter_async_context(Client(self._target))
        listed = await self._client.list_tools()
        self._schemas = [
            {"type": "function", "function": {"name": t.name, "description": t.description or "", "parameters": t.input_schema}}
            for t in listed.tools
        ]

    def close(self) -> None:
        if self._stack is not None:
            with contextlib.suppress(Exception):  # best effort on shutdown
                self._run(self._stack.aclose())
            self._stack = None
        self._loop.call_soon_threadsafe(self._loop.stop)

    def __enter__(self) -> McpToolExecutor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- ToolExecutor
    def tool_schemas(self) -> list[dict[str, Any]]:
        return self._schemas

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        t0 = time.perf_counter()
        try:
            res = self._run(self._client.call_tool(name, args))
        except Exception as e:  # transport / protocol failure -> observation, not crash
            return ToolResult(name=name, args=args, ok=False, error=f"MCP 调用失败: {type(e).__name__}: {e}",
                              latency_ms=round((time.perf_counter() - t0) * 1000, 2))  # fmt: skip
        latency = round((time.perf_counter() - t0) * 1000, 2)
        text = "".join(getattr(c, "text", "") for c in res.content)
        if res.is_error:
            return ToolResult(name=name, args=args, ok=False, error=text, latency_ms=latency)
        data = res.structured_content["result"] if res.structured_content and "result" in res.structured_content else json.loads(text)
        return ToolResult(name=name, args=args, ok=True, data=data, latency_ms=latency)
