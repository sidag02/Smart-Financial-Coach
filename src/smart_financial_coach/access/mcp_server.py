"""The MCP server: the tools, over Streamable HTTP, for the coach and outside assistants (FR-19).

One contract for every client (Technical Design, "Tool interface"): the coach and an assistant
such as Claude Desktop call the same tools here and get the same numbers (key scenario 6). Each
tool wraps `access.tools.Tools`, built for the user named by the bearer token, never by a tool
argument. Results are the tools' JSON plus a `source` (title, detail), which the coach shows as a
source chip (FR-16).

    server, asgi = build_mcp_server(sources, tokens, public_url="https://…")
    app.mount("/", asgi)  # serves /mcp; run `server.session_manager.run()` in the lifespan
"""

from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as McpToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from starlette.applications import Starlette

from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.tokens import AccessTokens
from smart_financial_coach.access.tools import MAX_TRANSACTIONS, TOOL_SPECS, ToolError, Tools

PATH = "/mcp"
# The host the coach uses to reach this server in-process (access.mcp_client)
INTERNAL_HOST = "sfc.internal"
INSTRUCTIONS = (
    "Smart Financial Coach: the signed-in person's own categorized spending, income and savings "
    "goals, from synthetic data that ends on the as-of date in each result. Quote numbers only "
    "from tool results. Tools marked not_available have no model yet; don't estimate them."
)

_DESCRIPTIONS = {spec["name"]: spec["description"] for spec in TOOL_SPECS}
Day = Annotated[str, Field(description="YYYY-MM-DD, inclusive")]


def build_mcp_server(
    sources: DataSources, tokens: AccessTokens, *, public_url: str
) -> tuple[MCPServer, Starlette]:
    server = MCPServer(
        "smart-financial-coach",
        title="Smart Financial Coach",
        instructions=INSTRUCTIONS,
        token_verifier=tokens,
        auth=AuthSettings(
            issuer_url=public_url,  # pydantic parses the URL
            resource_server_url=f"{public_url}{PATH}",
            validate_token_resource=False,  # AccessTokens checks its own signature and expiry
        ),
    )

    def run(name: str, **arguments: Any) -> dict[str, Any]:
        token = get_access_token()
        if token is None or token.subject is None:  # the transport requires one; belt and braces
            raise McpToolError("not signed in")
        tools = Tools(Ledger.load(sources, token.subject))
        try:
            result = tools.call(name, {k: v for k, v in arguments.items() if v is not None})
        except ToolError as error:
            raise McpToolError(str(error)) from error
        return {
            **result.data,
            "source": {"title": result.source.title, "detail": result.source.detail},
        }

    @server.tool(description=_DESCRIPTIONS["get_spending_summary"])
    def get_spending_summary(start_date: Day, end_date: Day) -> dict[str, Any]:
        return run("get_spending_summary", start_date=start_date, end_date=end_date)

    @server.tool(description=_DESCRIPTIONS["get_transactions"])
    def get_transactions(
        start_date: Day,
        end_date: Day,
        category: Annotated[str | None, Field(description="one of the 13 categories")] = None,
        search: Annotated[str | None, Field(description="merchant text to match")] = None,
        sort: Literal["newest", "largest"] = "newest",
        limit: Annotated[int, Field(ge=1, le=MAX_TRANSACTIONS)] = 25,
    ) -> dict[str, Any]:
        return run(
            "get_transactions",
            start_date=start_date,
            end_date=end_date,
            category=category,
            search=search,
            sort=sort,
            limit=limit,
        )

    @server.tool(description=_DESCRIPTIONS["list_goals"])
    def list_goals() -> dict[str, Any]:
        return run("list_goals")

    @server.tool(description=_DESCRIPTIONS["detect_anomalies"])
    def detect_anomalies(start_date: Day, end_date: Day) -> dict[str, Any]:
        return run("detect_anomalies", start_date=start_date, end_date=end_date)

    @server.tool(description=_DESCRIPTIONS["forecast_goal"])
    def forecast_goal(goal_name: str) -> dict[str, Any]:
        return run("forecast_goal", goal_name=goal_name)

    public = urlsplit(public_url)
    hosts = [public.netloc, INTERNAL_HOST, "127.0.0.1:*", "localhost:*", "testserver"]
    origins = [f"{public.scheme}://{public.netloc}", "http://127.0.0.1:*", "http://localhost:*"]
    asgi = server.streamable_http_app(
        streamable_http_path=PATH,
        stateless_http=True,  # every request carries its token; nothing to keep between them
        json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=origins
        ),
    )
    return server, asgi
