"""The five travel tools exposed as an MCP server (stdio).

    uv run python -m trip_agents.tools.mcp_server            # backend from TOOL_BACKEND (mock | real)

Uses the low-level `Server` rather than the decorator API on purpose:
* `list_tools` publishes the JSON Schema of *our* Pydantic input models, so the
  contract the model sees is exactly the one we validate against;
* `call_tool` routes through `execute_tool`, so argument errors come back as a
  readable `is_error` result the agent can correct, instead of a protocol error.
The structured JSON is returned both as text (for any client) and as
`structured_content`.
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from trip_agents.config import get_settings
from trip_agents.tools.executor import ToolBackend, execute_tool, get_backend
from trip_agents.tools.specs import TOOL_SPECS

SERVER_NAME = "trip-tools"


def build_server(backend: ToolBackend | None = None) -> Server[Any]:
    backend = backend or get_backend(get_settings().tool_backend)

    async def list_tools(ctx: Any, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(name=s.name, description=s.description, input_schema=s.input_model.model_json_schema())
                for s in TOOL_SPECS.values()
            ]
        )

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        # Backends are sync (mock is CPU-only, real uses blocking httpx); keep the event loop free.
        res = await anyio.to_thread.run_sync(execute_tool, params.name, dict(params.arguments or {}), backend)
        if not res.ok:
            return types.CallToolResult(content=[types.TextContent(type="text", text=res.error or "error")], is_error=True)
        payload = {"result": res.data}
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(res.data, ensure_ascii=False))],
            structured_content=payload,
        )

    return Server(SERVER_NAME, version="0.1.0", on_list_tools=list_tools, on_call_tool=call_tool)


async def _serve() -> None:
    server = build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    anyio.run(_serve)


if __name__ == "__main__":
    main()
