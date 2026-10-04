"""The MCP server: the tools, over Streamable HTTP, for the coach and outside assistants (FR-19).

One contract for every client (Technical Design, "Tool interface"): the coach and an assistant
such as Claude Desktop call the same tools here and get the same numbers (key scenario 6). Each
tool wraps `access.tools.Tools`, built for the user named by the bearer token, never by a tool
argument. Results are the tools' JSON plus a `source` (title, detail), which the coach shows as a
source chip (FR-16). Category feedback (FR-5, FR-6) is read and written for the token's feedback
subject, as an assistant: changes to more than one transaction need `confirm` (#15 §3). Savings
goals (FR-10) are read and written for the same subject; the token's `client` claim says whether
a change comes from Wren (`coach`) or another assistant (`assistant`), and every goal change needs
`confirm`. A goal change that breaks a rule comes back as status `invalid` with its problems. A
token without a feedback subject gets read-only tools.

    server, asgi = build_mcp_server(sources, tokens, public_url="https://…")
    app.mount("/", asgi)  # serves /mcp; run `server.session_manager.run()` in the lifespan
"""

from collections.abc import Sequence
from typing import Annotated, Any, Literal, cast
from urllib.parse import urlsplit

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as McpToolError
from mcp.server.mcpserver.utilities.func_metadata import ArgModelBase
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import ConfigDict, Field
from starlette.applications import Starlette

from smart_financial_coach.access.feedback import FeedbackStore
from smart_financial_coach.access.goals import GoalStore
from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.tokens import AccessTokens
from smart_financial_coach.access.tools import (
    MAX_ITEMS,
    MAX_TRANSACTIONS,
    TOOL_SPECS,
    Feedback,
    GoalAccess,
    GoalProblemsError,
    ToolError,
    Tools,
)

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
GoalId = Annotated[str, Field(description="a goal_id from list_goals")]
GoalName = Annotated[
    str | None, Field(description="what the user is saving for, up to 40 characters")
]
Dollars = Annotated[float | None, Field(description="US dollars, to the cent")]
GoalDate = Annotated[
    str | None, Field(description="YYYY-MM-DD; the goal is due at the end of that month")
]
Saved = Annotated[float | None, Field(description="US dollars saved toward it so far")]
Confirm = Annotated[bool, Field(description="the user agreed to the preview")]
READ = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
WRITE = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False
)


def build_mcp_server(
    sources: DataSources,
    tokens: AccessTokens,
    *,
    public_url: str,
    feedback: FeedbackStore | None = None,
    goals: GoalStore | None = None,
    extra_hosts: Sequence[str] = (),
) -> tuple[MCPServer, Starlette]:
    """`extra_hosts` adds Host headers to accept beyond the public and in-process ones (tests)."""
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

    def run(tool: str, /, **arguments: Any) -> dict[str, Any]:
        token = get_access_token()
        if token is None or token.subject is None:  # the transport requires one; belt and braces
            raise McpToolError("not signed in")
        subject = tokens.feedback_subject(token.token)
        context = Feedback(feedback, subject, "coach") if feedback and subject else None
        # Wren's in-process tokens say client "coach"; any other client is an outside assistant
        source = "coach" if token.client_id == "coach" else "assistant"
        goal_access = GoalAccess(goals, subject, source) if goals and subject else None
        tools = Tools(Ledger.load(sources, token.subject), context, goal_access)
        try:
            result = tools.call(tool, {k: v for k, v in arguments.items() if v is not None})
        except GoalProblemsError as error:
            # A goal change that breaks a rule is an answer, not a failure: the problems go back
            # with their fields and codes, as check_goal returns them (review on #43)
            return {
                "status": "invalid",
                "message": str(error),
                "problems": [
                    {"field": p.field, "code": p.code, "message": p.message} for p in error.problems
                ],
                "source": {"title": "Goal not changed", "detail": f"{tool} refused"},
            }
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

    @server.tool(description=_DESCRIPTIONS["list_review_items"])
    def list_review_items(limit: Annotated[int, Field(ge=1, le=MAX_ITEMS)] = 10) -> dict[str, Any]:
        return run("list_review_items", limit=limit)

    @server.tool(description=_DESCRIPTIONS["resolve_review_item"])
    def resolve_review_item(
        item_id: str,
        action: Literal["confirm", "correct"],
        category: Annotated[str | None, Field(description="the new category, for correct")] = None,
        confirm: Annotated[bool, Field(description="the user agreed to the preview")] = False,
    ) -> dict[str, Any]:
        return run(
            "resolve_review_item",
            item_id=item_id,
            action=action,
            category=category,
            confirm=confirm,
        )

    @server.tool(description=_DESCRIPTIONS["correct_category"])
    def correct_category(
        transaction_id: str,
        category: Annotated[str, Field(description="one of the 13 categories")],
        scope: Literal["merchant", "transaction"] = "merchant",
        confirm: Annotated[bool, Field(description="the user agreed to the preview")] = False,
    ) -> dict[str, Any]:
        return run(
            "correct_category",
            transaction_id=transaction_id,
            category=category,
            scope=scope,
            confirm=confirm,
        )

    @server.tool(description=_DESCRIPTIONS["undo_correction"])
    def undo_correction(correction_id: str) -> dict[str, Any]:
        return run("undo_correction", correction_id=correction_id)

    @server.tool(description=_DESCRIPTIONS["list_corrections"])
    def list_corrections(limit: Annotated[int, Field(ge=1, le=MAX_ITEMS)] = 10) -> dict[str, Any]:
        return run("list_corrections", limit=limit)

    @server.tool(description=_DESCRIPTIONS["list_goals"], annotations=READ)
    def list_goals(include_ended: bool = True, include_archived: bool = False) -> dict[str, Any]:
        return run("list_goals", include_ended=include_ended, include_archived=include_archived)

    @server.tool(description=_DESCRIPTIONS["check_goal"], annotations=READ)
    def check_goal(
        name: GoalName = None,
        target_amount: Dollars = None,
        target_date: GoalDate = None,
        saved: Saved = None,
        goal_id: Annotated[str | None, Field(description="the goal being edited")] = None,
    ) -> dict[str, Any]:
        return run(
            "check_goal",
            name=name,
            target_amount=target_amount,
            target_date=target_date,
            saved=saved,
            goal_id=goal_id,
        )

    @server.tool(description=_DESCRIPTIONS["create_goal"], annotations=WRITE)
    def create_goal(
        name: Annotated[str, Field(description="what the user is saving for, up to 40 characters")],
        target_amount: Annotated[float, Field(description="US dollars, to the cent")],
        target_date: Annotated[
            str, Field(description="YYYY-MM-DD; the goal is due at the end of that month")
        ],
        saved: Saved = None,
        confirm: Confirm = False,
    ) -> dict[str, Any]:
        return run(
            "create_goal",
            name=name,
            target_amount=target_amount,
            target_date=target_date,
            saved=saved,
            confirm=confirm,
        )

    @server.tool(description=_DESCRIPTIONS["update_goal"], annotations=WRITE)
    def update_goal(
        goal_id: GoalId,
        name: GoalName = None,
        target_amount: Dollars = None,
        target_date: GoalDate = None,
        saved: Saved = None,
        confirm: Confirm = False,
    ) -> dict[str, Any]:
        return run(
            "update_goal",
            goal_id=goal_id,
            name=name,
            target_amount=target_amount,
            target_date=target_date,
            saved=saved,
            confirm=confirm,
        )

    @server.tool(description=_DESCRIPTIONS["archive_goal"], annotations=WRITE)
    def archive_goal(goal_id: GoalId, confirm: Confirm = False) -> dict[str, Any]:
        return run("archive_goal", goal_id=goal_id, confirm=confirm)

    @server.tool(description=_DESCRIPTIONS["undo_goal_change"], annotations=WRITE)
    def undo_goal_change(revision_id: str) -> dict[str, Any]:
        return run("undo_goal_change", revision_id=revision_id)

    @server.tool(description=_DESCRIPTIONS["detect_anomalies"])
    def detect_anomalies(start_date: Day, end_date: Day) -> dict[str, Any]:
        return run("detect_anomalies", start_date=start_date, end_date=end_date)

    @server.tool(description=_DESCRIPTIONS["forecast_goal"], annotations=READ)
    def forecast_goal(goal_id: GoalId) -> dict[str, Any]:
        return run("forecast_goal", goal_id=goal_id)

    _reject_unknown_arguments(server)
    public = urlsplit(public_url)
    hosts = [public.netloc, INTERNAL_HOST, "127.0.0.1:*", "localhost:*", *extra_hosts]
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


def _reject_unknown_arguments(server: MCPServer) -> None:
    """Hold every tool's arguments to its schema, as `Tools.call` does in-process: an argument the
    tool doesn't take (a `user_id`, say) is refused rather than dropped, and values aren't
    coerced (`confirm: "yes"` isn't true). The SDK's argument models ignore extras and validate
    leniently, so each is replaced by a strict subclass, and the published schema says
    `additionalProperties: false`. Scope never came from arguments, so nothing leaked; this makes
    a wrong call fail loudly (review on #43)."""
    for tool in server._tool_manager.list_tools():
        lenient = tool.fn_metadata.arg_model
        config = ConfigDict(**{**lenient.model_config, "extra": "forbid", "strict": True})
        namespace = {"model_config": config, "__module__": lenient.__module__}
        strict = cast(type[ArgModelBase], type(lenient.__name__, (lenient,), namespace))
        tool.fn_metadata.arg_model = strict
        tool.parameters = strict.model_json_schema(by_alias=True)
