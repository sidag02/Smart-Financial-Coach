"""The MCP server: bearer tokens decide whose data a tool reads, and every client gets the same
numbers as the in-process tools (FR-19, key scenarios 5 and 6)."""

import re
import threading
import time
from collections.abc import Callable, Iterator
from datetime import date, timedelta
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import anyio
import anyio.to_thread
import pytest
from anyio.lowlevel import EventLoopToken, current_token
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.types import ASGIApp

from smart_financial_coach.access.goals import GoalStore
from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.mcp_client import McpTools
from smart_financial_coach.access.mcp_server import build_mcp_server
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


def in_worker_thread(client: TestClient, fn: Callable[[], T]) -> T:
    """Run blocking code in a worker thread of the app's event loop, as a request handler does."""
    assert client.portal is not None, "use the client inside its with-block"
    return client.portal.call(anyio.to_thread.run_sync, fn)


def app_loop(client: TestClient) -> EventLoopToken:
    assert client.portal is not None, "use the client inside its with-block"
    return client.portal.call(current_token)


def mcp_tools(
    client: TestClient,
    user: str,
    lifetime: timedelta = timedelta(minutes=5),
    app: ASGIApp | None = None,
    feedback_subject: str | None = None,
) -> McpTools:
    token = AccessTokens(SECRET, [user]).issue(user, lifetime, feedback_subject=feedback_subject)
    return McpTools(app or client.app, token, as_of=AS_OF, loop=app_loop(client))


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
        in_worker_thread(client, lambda: expired.specs)


def test_a_failed_call_from_a_request_thread_runs_once(
    client: TestClient, users: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure is raised, not retried on another loop: a retried write would apply twice."""
    sessions: list[int] = []
    session = McpTools._session

    async def counted(self: McpTools, use: Any) -> Any:
        sessions.append(threading.get_ident())
        return await session(self, use)

    monkeypatch.setattr(McpTools, "_session", counted)
    expired = mcp_tools(client, users[0], lifetime=timedelta(seconds=-1))

    with pytest.raises(ToolsUnavailableError):
        in_worker_thread(client, lambda: expired.call("list_goals", {}))
    assert len(sessions) == 1
    assert sessions[0] == client.portal.call(threading.get_ident)  # type: ignore[union-attr]


def test_the_server_lists_the_tools_with_no_user_argument(
    client: TestClient, users: list[str]
) -> None:
    specs = in_worker_thread(client, lambda: mcp_tools(client, users[0]).specs)

    assert {s["name"] for s in specs} == {
        "get_spending_summary",
        "get_transactions",
        "list_review_items",
        "resolve_review_item",
        "correct_category",
        "undo_correction",
        "list_corrections",
        "list_goals",
        "check_goal",
        "create_goal",
        "update_goal",
        "archive_goal",
        "undo_goal_change",
        "detect_anomalies",
        "get_alert_settings",
        "set_alert_sensitivity",
        "act_on_flag",
        "undo_flag_action",
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
        over_mcp = in_worker_thread(client, call)
        assert over_mcp.data == direct.data
        assert over_mcp.source == direct.source


def test_the_token_not_an_argument_decides_whose_data(
    client: TestClient, sources: DataSources, users: list[str]
) -> None:
    mine, theirs = (Ledger.load(sources, u).transactions for u in users)
    span = {"start_date": "2023-01-01", "end_date": "2026-09-30", "limit": 50}
    tools = mcp_tools(client, users[0])

    result = in_worker_thread(client, lambda: tools.call("get_transactions", span))

    ids = {t["transaction_id"] for t in result.data["transactions"]}
    assert ids
    assert ids <= set(mine["transaction_id"])
    assert ids.isdisjoint(theirs["transaction_id"])
    # Naming someone else is refused outright, not quietly ignored (review on #43)
    with pytest.raises(ToolError, match="user_id"):
        in_worker_thread(
            client, lambda: tools.call("get_transactions", {**span, "user_id": users[1]})
        )


def test_every_tool_refuses_unknown_arguments_and_doesnt_coerce(
    client: TestClient, users: list[str]
) -> None:
    tools = mcp_tools(client, users[0], feedback_subject="session-a")
    specs = in_worker_thread(client, lambda: tools.specs)
    for spec in specs:
        assert spec["input_schema"]["additionalProperties"] is False, spec["name"]

    for name, args in (
        ("list_review_items", {"subject": "session-b"}),
        ("detect_anomalies", {**SEPTEMBER, "user_id": users[1]}),
    ):
        with pytest.raises(ToolError, match="Extra inputs are not permitted"):
            in_worker_thread(client, partial(tools.call, name, args))
    with pytest.raises(ToolError, match="limit"):  # "10" isn't 10
        in_worker_thread(client, lambda: tools.call("list_review_items", {"limit": "10"}))


def test_tool_errors_come_back_as_tool_errors(client: TestClient, users: list[str]) -> None:
    tools = mcp_tools(client, users[0])

    with pytest.raises(ToolError, match="unknown category 'Crypto'"):
        in_worker_thread(
            client, lambda: tools.call("get_transactions", {**SEPTEMBER, "category": "Crypto"})
        )
    with pytest.raises(ToolError, match="limit"):
        in_worker_thread(
            client, lambda: tools.call("get_transactions", {**SEPTEMBER, "limit": "ten"})
        )
    trip = {"name": "Trip", "target_amount": 3000, "target_date": "2027-06-01", "confirm": True}
    created = in_worker_thread(client, lambda: tools.call("create_goal", trip))
    goal_id = created.data["goal"]["goal_id"]
    not_available = in_worker_thread(
        client, lambda: tools.call("forecast_goal", {"goal_id": goal_id})
    )
    assert not_available.data["status"] == "not_available"  # no goal-forecasting model here


def test_the_connect_page_hands_out_a_working_token_for_the_signed_in_user(
    client: TestClient, users: list[str]
) -> None:
    client.post("/signin", data={"email": "maya@x.com", "password": "correct horse"})
    reply = client.get("/connect")
    assert reply.headers["Cache-Control"] == "no-store"  # it shows a credential
    page = reply.text

    match = re.search(r'<code id="token">(sfc_[^<]+)</code>', page)
    assert match
    token = match.group(1)
    assert AccessTokens(SECRET, users).user(token) == users[0]
    assert "http://127.0.0.1:8000/mcp" in page
    assert "claude mcp add --transport http" in page
    tools = McpTools(client.app, token, as_of=AS_OF, loop=app_loop(client))
    summary: Any = in_worker_thread(client, lambda: tools.call("list_goals", {}))
    assert summary.source.title == "Savings goals"


def test_a_signed_out_visitor_gets_no_token(client: TestClient) -> None:
    reply = client.get("/connect", follow_redirects=False)
    assert reply.status_code == 303
    assert "sfc_" not in reply.text


def test_feedback_follows_the_tokens_feedback_subject(client: TestClient, users: list[str]) -> None:
    """Two visitors on one demo account (two sessions, two tokens): a correction made through one
    assistant is invisible to the other, and changes the first one's numbers at once."""
    mine = mcp_tools(client, users[0], feedback_subject="session-a")
    theirs = mcp_tools(client, users[0], feedback_subject="session-b")

    def scenario() -> tuple[dict[str, Any], ...]:
        item = mine.call("list_review_items", {"limit": 1}).data["items"][0]
        new = "Travel" if item["suggested_category"] != "Travel" else "Entertainment"
        args = {"item_id": item["item_id"], "action": "correct", "category": new}
        preview = mine.call("resolve_review_item", args).data
        applied = mine.call("resolve_review_item", {**args, "confirm": True}).data
        their_items = theirs.call("list_review_items", {"limit": 25}).data
        my_items = mine.call("list_review_items", {"limit": 25}).data
        return item, preview, applied, my_items, their_items

    item, preview, applied, my_items, their_items = in_worker_thread(client, scenario)

    if item["transaction_count"] > 1:
        assert preview["status"] == "needs_confirmation"  # the coach asks before a bulk change
    assert applied["status"] == "applied"
    assert applied["correction"]["source"] == "coach"
    assert item["item_id"] not in {i["item_id"] for i in my_items["items"]}
    assert my_items["open_items"] == their_items["open_items"] - 1
    assert item["item_id"] in {i["item_id"] for i in their_items["items"]}


def test_a_token_from_before_feedback_gets_read_only_tools(
    client: TestClient, users: list[str]
) -> None:
    """No `fb` claim: not the shared account's feedback, which every visitor would share."""
    tokens = AccessTokens(SECRET, users)
    expires = int(time.time()) + 300
    old = "sfc_" + tokens._signer.dumps({"sub": users[0], "exp": expires, "client": "assistant"})
    assert tokens.user(old) == users[0]
    assert tokens.feedback_subject(old) is None
    tools = McpTools(client.app, old, as_of=AS_OF, loop=app_loop(client))

    listed = in_worker_thread(client, lambda: tools.call("list_review_items", {}).data)
    assert listed["open_items"] > 0  # reading works
    with pytest.raises(ToolError, match="isn't available"):
        in_worker_thread(client, lambda: tools.call("list_corrections", {}))


TRIP = {"name": "Trip", "target_amount": 3000, "target_date": "2027-06-01"}


def test_goal_changes_follow_the_tokens_subject_and_client(
    client: TestClient, users: list[str], goals_db: Path
) -> None:
    """Wren (client "coach") and an outside assistant on one session change that session's goals,
    each recorded as itself; another session on the same account sees neither change (FR-10)."""
    tokens = AccessTokens(SECRET, users)
    lifetime = timedelta(minutes=5)
    loop = app_loop(client)
    wren_token = tokens.issue(users[0], lifetime, client="coach", feedback_subject="session-a")
    wren = McpTools(client.app, wren_token, as_of=AS_OF, loop=loop)
    outside = mcp_tools(client, users[0], feedback_subject="session-a")
    other = mcp_tools(client, users[0], feedback_subject="session-b")
    bike = {"name": "Bike", "target_amount": 800.5, "target_date": "2027-03-31", "confirm": True}

    def scenario() -> tuple[dict[str, Any], ...]:
        preview = wren.call("create_goal", TRIP).data
        wren.call("create_goal", {**TRIP, "confirm": True})
        outside.call("create_goal", bike)
        mine = outside.call("list_goals", {}).data
        theirs = other.call("list_goals", {}).data
        return preview, mine, theirs

    preview, mine, theirs = in_worker_thread(client, scenario)

    assert preview["status"] == "needs_confirmation"
    assert preview["target_date"] == "2027-06-30"
    assert {"Trip", "Bike"} <= {g["name"] for g in mine["goals"]}
    assert {"Trip", "Bike"}.isdisjoint(g["name"] for g in theirs["goals"])
    revisions = GoalStore(goals_db).revisions("session-a", users[0])
    assert [(r.name, r.source) for r in revisions] == [("Bike", "assistant"), ("Trip", "coach")]
    assert revisions[0].target_amount_cents == 800_50


def test_a_token_from_before_feedback_reads_goals_but_cant_change_them(
    client: TestClient, users: list[str]
) -> None:
    tokens = AccessTokens(SECRET, users)
    expires = int(time.time()) + 300
    old = "sfc_" + tokens._signer.dumps({"sub": users[0], "exp": expires, "client": "assistant"})
    tools = McpTools(client.app, old, as_of=AS_OF, loop=app_loop(client))

    listed = in_worker_thread(client, lambda: tools.call("list_goals", {}).data)
    assert listed["goals"]
    assert in_worker_thread(client, lambda: tools.call("check_goal", TRIP).data)["valid"]
    with pytest.raises(ToolError, match="isn't available for this sign-in"):
        in_worker_thread(client, lambda: tools.call("create_goal", {**TRIP, "confirm": True}))


def test_goal_problems_come_back_with_their_fields_and_codes(
    client: TestClient, users: list[str]
) -> None:
    tools = mcp_tools(client, users[0], feedback_subject="session-a")
    args = {**TRIP, "name": "", "target_amount": 5, "confirm": True}
    result = in_worker_thread(client, lambda: tools.call("create_goal", args))

    assert result.data["status"] == "invalid"
    assert {(p["field"], p["code"]) for p in result.data["problems"]} == {
        ("name", "name_missing"),
        ("target_amount", "amount_range"),
    }
    assert "Goals start at $50." in result.data["message"]
    assert result.source.title == "Goal not changed"


def test_reads_are_marked_read_only_and_writes_not(sources: DataSources, users: list[str]) -> None:
    server, _ = build_mcp_server(
        sources, AccessTokens(SECRET, users), public_url="http://127.0.0.1:8000"
    )
    hints = {t.name: t.annotations for t in anyio.run(server.list_tools)}

    for name in ("list_goals", "check_goal", "forecast_goal"):
        hint = hints[name]
        assert hint is not None
        assert hint.read_only_hint is True
    for name in ("create_goal", "update_goal", "archive_goal", "undo_goal_change"):
        hint = hints[name]
        assert hint is not None
        assert hint.read_only_hint is False
        assert hint.destructive_hint is False


def test_alert_settings_follow_the_tokens_subject(
    preset_flagged_sources: DataSources, preset_spike_sources: DataSources, users: list[str]
) -> None:
    """An outside assistant previews a sensitivity change until the person agrees, on its token's
    session only; a token without a feedback subject reads the setting but can't change it
    (FR-9 §5)."""
    sources = DataSources(
        preset_flagged_sources.dataset,
        preset_flagged_sources.predictions,
        preset_flagged_sources.flags,
        spikes=preset_spike_sources.spikes,
    )
    accounts = [Account(users[0], "Maya Chen", "maya@x.com")]
    settings = Settings(
        _env_file=None,
        demo_password=SecretStr("correct horse"),
        session_secret=SecretStr(SECRET),
        secure_cookies=False,
    )
    with TestClient(create_app(settings, sources=sources, accounts=accounts)) as client:
        mine = mcp_tools(client, users[0], feedback_subject="session-a")
        theirs = mcp_tools(client, users[0], feedback_subject="session-b")
        tokens = AccessTokens(SECRET, users)
        claims = {"sub": users[0], "exp": int(time.time()) + 300, "client": "assistant"}
        old = "sfc_" + tokens._signer.dumps(claims)  # no `fb` claim: read-only tools
        read_only = McpTools(client.app, old, as_of=AS_OF, loop=app_loop(client))

        def scenario() -> tuple[Any, ...]:
            preview = mine.call("set_alert_sensitivity", {"level": "more"}).data
            before = mine.call("get_alert_settings", {}).data["sensitivity"]
            mine.call("set_alert_sensitivity", {"level": "more", "confirm": True})
            after = mine.call("get_alert_settings", {}).data["sensitivity"]
            other = theirs.call("get_alert_settings", {}).data["sensitivity"]
            try:
                read_only.call("set_alert_sensitivity", {"level": "less", "confirm": True})
                refused = False
            except ToolError:
                refused = True
            return preview, before, after, other, refused

        preview, before, after, other, refused = in_worker_thread(client, scenario)

    assert preview["status"] == "needs_confirmation"
    assert (before, after, other) == ("balanced", "more", "balanced")
    assert refused


def test_alert_reads_and_writes_are_marked(sources: DataSources, users: list[str]) -> None:
    server, _ = build_mcp_server(
        sources, AccessTokens(SECRET, users), public_url="http://127.0.0.1:8000"
    )
    hints = {t.name: t.annotations for t in anyio.run(server.list_tools)}
    get = hints["get_alert_settings"]
    assert get is not None
    assert get.read_only_hint is True
    for name in ("set_alert_sensitivity", "act_on_flag", "undo_flag_action"):
        hint = hints[name]
        assert hint is not None
        assert hint.read_only_hint is False
        assert hint.destructive_hint is False
