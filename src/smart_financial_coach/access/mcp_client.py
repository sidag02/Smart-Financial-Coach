"""The coach's tools: the MCP server, reached in-process with a bearer token for one user.

Calls go through the full HTTP stack (bearer auth, schema validation) of the app's own `/mcp`
endpoint, over an in-memory ASGI transport rather than the network, so the coach is exactly as
privileged as an outside assistant holding that user's token. The tool list comes from the
server, so the coach and outside assistants see one contract.

    tools = McpTools(app, token, as_of=date(2026, 9, 30))
    tools.call("get_spending_summary", {"start_date": "2026-09-01", "end_date": "2026-09-30"})
"""

import json
from collections.abc import Awaitable, Callable, Coroutine
from datetime import date
from typing import Any, TypeVar

import anyio
import anyio.from_thread
import httpx2
from anyio.lowlevel import EventLoopToken
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from starlette.types import ASGIApp

from smart_financial_coach.access.mcp_server import INTERNAL_HOST, PATH
from smart_financial_coach.access.tools import (
    Source,
    ToolError,
    ToolResult,
    ToolSpec,
    ToolsUnavailableError,
)

T = TypeVar("T")
TIMEOUT_SECONDS = 20.0


class McpTools:
    def __init__(
        self, app: ASGIApp, token: str, *, as_of: date, loop: EventLoopToken | None = None
    ) -> None:
        """`loop` is the app's event loop, which serves the ASGI app; None runs a private loop
        (a script). The caller decides, so a failed call is never retried on another loop."""
        self._app = app
        self._token = token
        self.as_of = as_of
        self._loop = loop
        self._specs: list[ToolSpec] | None = None

    def _blocking(self, fn: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """Run an async MCP call from the coach's synchronous code, exactly once."""
        if self._loop is not None:
            return anyio.from_thread.run(fn, token=self._loop)
        return anyio.run(fn)

    async def _session(self, use: Callable[[Client], Awaitable[T]]) -> T:
        http = httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=self._app),
            base_url=f"http://{INTERNAL_HOST}",
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=TIMEOUT_SECONDS,
        )
        try:
            async with (
                http,
                Client(
                    streamable_http_client(f"http://{INTERNAL_HOST}{PATH}", http_client=http)
                ) as client,
            ):
                return await use(client)
        except Exception as error:  # 401, the server down, a protocol error: no tools at all
            raise ToolsUnavailableError(f"MCP server: {error}") from error

    @property
    def specs(self) -> list[ToolSpec]:
        if self._specs is None:

            async def listed(client: Client) -> list[ToolSpec]:
                tools = (await client.list_tools()).tools
                return [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "input_schema": t.input_schema,
                    }
                    for t in tools
                ]

            self._specs = self._blocking(lambda: self._session(listed))
        return self._specs

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        async def called(client: Client) -> Any:
            return await client.call_tool(name, arguments)

        result = self._blocking(lambda: self._session(called))
        if result.is_error:
            text = " ".join(getattr(part, "text", "") for part in result.content)
            raise ToolError(text.removeprefix(f"Error executing tool {name}: ") or f"{name} failed")
        data = dict(result.structured_content or json.loads(result.content[0].text))
        source = data.pop("source", {"title": name, "detail": ""})
        return ToolResult(data, Source(source["title"], source["detail"]))
