"""The MCP server: bearer tokens decide whose data a tool reads, and every client gets the same
numbers as the in-process tools (FR-19, key scenarios 5 and 6)."""

import re
from collections.abc import Callable, Iterator
from datetime import date, timedelta
from functools import partial
from typing import Any, TypeVar

import anyio.to_thread
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.mcp_client import McpTools
from smart_financial_coach.access.tokens import AccessTokens
from smart_financial_coach.access.tools import ToolError, Tools, ToolsUnavailableError
from smart_financial_coach.config import Settings
from smart_financial_coach.experience.accounts import Account
from smart_financial_coach.experience.web.app import create_app

T = TypeVar("T")
SECRET = "s" * 32
SEPTEMBER = {"start_date": "2026-09-01", "end_date": "2026-09-30"}
AS_OF = date(2026, 9, 30)


@pytest.fixture
def users(two_users: tuple[str, str]) -> list[str]:
    return list(two_users)


@pytest.fixture
def client(sources: DataSources, users: list[str]) -> Iterator[TestClient]:
    accounts = [
        Account(users[0], "Maya Chen", "maya@x.com"),
        Account(users[1], "Ada Okafor", "ada@x.com"),
    ]
    settings = Settings(
        _env_file=None,
        demo_password=SecretStr("correct horse"),
        session_secret=SecretStr(SECRET),
        secure_cookies=False,
    )
    with TestClient(create_app(settings, sources=sources, accounts=accounts)) as c:
        yield c


def in_app(client: TestClient, fn: Callable[[], T]) -> T:
    """Run blocking MCP client code in a worker thread of the app's event loop, as a request
    handler would."""
    assert client.portal is not None, "use the client inside its with-block"
    return client.portal.call(anyio.to_thread.run_sync, fn)


def mcp_tools(
    client: TestClient, user: str, lifetime: timedelta = timedelta(minutes=5)
) -> McpTools:
    token = AccessTokens(SECRET, [user]).issue(user, lifetime)
    return McpTools(client.app, token, as_of=AS_OF)


def test_tokens_name_a_known_user_until_they_expire(users: list[str]) -> None:
    tokens = AccessTokens(SECRET, users)
    token = tokens.issue(users[0], timedelta(minutes=1))

    assert tokens.user(token) == users[0]
    assert tokens.user(token + "x") is None
    assert tokens.user(token.removeprefix("sfc_")) is None
    assert AccessTokens("another secret " * 3, users).user(token) is None
    assert AccessTokens(SECRET, [users[1]]).user(token) is None  # no longer an account
    assert tokens.user(tokens.issue(users[0], timedelta(seconds=-1))) is None
    with pytest.raises(ValueError, match="unknown user"):
        tokens.issue("u_nobody", timedelta(minutes=1))


def test_the_mcp_endpoint_needs_a_valid_token(client: TestClient, users: list[str]) -> None:
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    headers = {"Accept": "application/json, text/event-stream"}

    assert client.post("/mcp", json=request, headers=headers).status_code == 401
    forged = {**headers, "Authorization": "Bearer sfc_forged"}
    assert client.post("/mcp", json=request, headers=forged).status_code == 401
    expired = mcp_tools(client, users[0], lifetime=timedelta(seconds=-1))
    with pytest.raises(ToolsUnavailableError):
        in_app(client, lambda: expired.specs)


def test_the_server_lists_the_tools_with_no_user_argument(
    client: TestClient, users: list[str]
) -> None:
    specs = in_app(client, lambda: mcp_tools(client, users[0]).specs)

    assert {s["name"] for s in specs} == {
        "get_spending_summary",
        "get_transactions",
        "list_goals",
        "detect_anomalies",
        "forecast_goal",
    }
    for spec in specs:
        assert "user_id" not in spec["input_schema"].get("properties", {})
        assert spec["description"]


def test_mcp_returns_the_same_numbers_as_the_in_process_tools(
    client: TestClient, sources: DataSources, users: list[str]
) -> None:
    for user in users:
        direct = Tools(Ledger.load(sources, user)).call("get_spending_summary", SEPTEMBER)
        call = partial(mcp_tools(client, user).call, "get_spending_summary", SEPTEMBER)
        over_mcp = in_app(client, call)
        assert over_mcp.data == direct.data
        assert over_mcp.source == direct.source


def test_the_token_not_an_argument_decides_whose_data(
    client: TestClient, sources: DataSources, users: list[str]
) -> None:
    mine, theirs = (Ledger.load(sources, u).transactions for u in users)
    span = {"start_date": "2023-01-01", "end_date": "2026-09-30", "limit": 50, "user_id": users[1]}

    result = in_app(client, lambda: mcp_tools(client, users[0]).call("get_transactions", span))

    ids = {t["transaction_id"] for t in result.data["transactions"]}
    assert ids
    assert ids <= set(mine["transaction_id"])
    assert ids.isdisjoint(theirs["transaction_id"])


def test_tool_errors_come_back_as_tool_errors(client: TestClient, users: list[str]) -> None:
    tools = mcp_tools(client, users[0])

    with pytest.raises(ToolError, match="unknown category 'Crypto'"):
        in_app(client, lambda: tools.call("get_transactions", {**SEPTEMBER, "category": "Crypto"}))
    with pytest.raises(ToolError, match="limit"):
        in_app(client, lambda: tools.call("get_transactions", {**SEPTEMBER, "limit": "ten"}))
    not_available = in_app(client, lambda: tools.call("forecast_goal", {"goal_name": "x"}))
    assert not_available.data["status"] == "not_available"


def test_the_connect_page_hands_out_a_working_token_for_the_signed_in_user(
    client: TestClient, users: list[str]
) -> None:
    client.post("/signin", data={"email": "maya@x.com", "password": "correct horse"})
    page = client.get("/connect").text

    match = re.search(r'<code id="token">(sfc_[^<]+)</code>', page)
    assert match
    token = match.group(1)
    assert AccessTokens(SECRET, users).user(token) == users[0]
    assert "http://127.0.0.1:8000/mcp" in page
    assert "claude mcp add --transport http" in page
    tools = McpTools(client.app, token, as_of=AS_OF)
    summary: Any = in_app(client, lambda: tools.call("list_goals", {}))
    assert summary.source.title == "Savings goals"


def test_a_signed_out_visitor_gets_no_token(client: TestClient) -> None:
    reply = client.get("/connect", follow_redirects=False)
    assert reply.status_code == 303
    assert "sfc_" not in reply.text
