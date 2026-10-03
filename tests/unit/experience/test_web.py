"""The web app: sign-in, pages, data isolation between users, and chat (NFR-2, NFR-5, NFR-6)."""

import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.config import Settings
from smart_financial_coach.experience.accounts import Account, SharedPassword
from smart_financial_coach.experience.coach import Coach
from smart_financial_coach.experience.web.app import MINUS, create_app, money, render_answer
from smart_financial_coach.experience.web.charts import money_flow, trend
from tests.unit.experience.fakes import FakeClient, api_error, response, text, tool_use

PASSWORD = "correct horse"


@pytest.fixture
def accounts(two_users: tuple[str, str]) -> list[Account]:
    return [
        Account(two_users[0], "Maya Chen", "maya@example.com"),
        Account(two_users[1], "Ada Okafor", "ada@example.com"),
    ]


def settings(**overrides: object) -> Settings:
    values = {
        "demo_password": SecretStr(PASSWORD),
        "session_secret": SecretStr("s" * 32),
        "secure_cookies": False,
        **overrides,
    }
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def make_client(
    sources: DataSources, accounts: list[Account], coach: Coach | None = None, **overrides: object
) -> TestClient:
    app = create_app(settings(**overrides), sources=sources, accounts=accounts, coach=coach)
    return TestClient(app)


@pytest.fixture
def client(sources: DataSources, accounts: list[Account]) -> Iterator[TestClient]:
    with make_client(sources, accounts) as c:
        yield c


def sign_in(client: TestClient, email: str = "maya@example.com") -> None:
    reply = client.post(
        "/signin", data={"email": email, "password": PASSWORD}, follow_redirects=False
    )
    assert reply.status_code == 303, reply.text


def test_serving_needs_a_password_and_a_session_secret(
    sources: DataSources, accounts: list[Account]
) -> None:
    with pytest.raises(RuntimeError, match="SFC_DEMO_PASSWORD"):
        create_app(Settings(_env_file=None), sources=sources, accounts=accounts)


def test_essentials_are_a_setting_checked_against_the_taxonomy(
    sources: DataSources, accounts: list[Account]
) -> None:
    for bad in (("Housing", "Crypto"), ("Housing", "Income")):
        with pytest.raises(ValueError, match="can't be essentials"):
            create_app(settings(essentials=bad), sources=sources, accounts=accounts)

    with make_client(sources, accounts, essentials=("Dining",)) as c:
        sign_in(c)
        page = c.get("/").text
    dining = Ledger.load(sources, accounts[0].user_id)
    month = dining.between(dining.as_of.replace(day=1), dining.as_of)
    spent = -month[month["category"] == "Dining"]["amount"].sum()
    assert f"{money(spent)} on essentials" in page


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads files whatever their mode")
def test_health_fails_when_the_demo_data_does_not_load(
    sources: DataSources, accounts: list[Account], tmp_path: Path
) -> None:
    """The deploy's smoke test relies on this: a bundle the app can't read isn't healthy."""
    predictions = tmp_path / "predictions.sqlite"
    shutil.copyfile(sources.predictions, predictions)
    predictions.chmod(0)
    unreadable = DataSources(sources.dataset, predictions)
    try:
        with make_client(unreadable, accounts) as c:
            reply = c.get("/healthz")
    finally:
        predictions.chmod(0o644)
    assert reply.status_code == 503
    assert reply.json()["status"] == "error"


def test_pages_need_a_signed_in_user(client: TestClient) -> None:
    for path in ("/", "/transactions", "/chat", "/goals", "/worth-a-look", "/flow"):
        reply = client.get(path, follow_redirects=False)
        assert reply.status_code == 303
        assert reply.headers["location"] == "/signin"
    htmx = client.get("/flow", headers={"HX-Request": "true"})
    assert htmx.status_code == 204
    assert htmx.headers["HX-Redirect"] == "/signin"
    assert client.get("/healthz").json() == {"status": "ok", "as_of": "2026-09-30", "users": 2}


def test_sign_in_checks_email_and_password(client: TestClient) -> None:
    for email, password in (("maya@example.com", "wrong"), ("nobody@example.com", PASSWORD)):
        reply = client.post("/signin", data={"email": email, "password": password})
        assert reply.status_code == 200
        assert "don&#39;t match" in reply.text

    sign_in(client, "  MAYA@example.com ")
    assert "Maya Chen" in client.get("/").text

    client.post("/signout")
    assert client.get("/", follow_redirects=False).status_code == 303


def test_quick_sign_in_is_off_unless_enabled(
    sources: DataSources, accounts: list[Account], client: TestClient
) -> None:
    assert "Continue as" not in client.get("/signin").text
    reply = client.post(
        "/signin/quick", data={"user_id": accounts[0].user_id}, follow_redirects=False
    )
    assert reply.headers["location"] == "/signin"

    with make_client(sources, accounts, quick_signin=True) as local:
        assert "Continue as Maya" in local.get("/signin").text
        local.post("/signin/quick", data={"user_id": accounts[0].user_id})
        assert "Maya Chen" in local.get("/").text


def test_sign_in_attempts_are_rate_limited(sources: DataSources, accounts: list[Account]) -> None:
    with make_client(sources, accounts, signin_attempts_per_minute=2) as c:
        for _ in range(2):
            c.post("/signin", data={"email": "maya@example.com", "password": "wrong"})
        reply = c.post("/signin", data={"email": "maya@example.com", "password": PASSWORD})
        assert "Too many attempts" in reply.text


def test_a_forged_forwarded_address_does_not_escape_the_sign_in_limit(
    sources: DataSources, accounts: list[Account]
) -> None:
    """Behind the ingress, only the X-Forwarded-For entry it appended is the visitor's."""
    with make_client(sources, accounts, signin_attempts_per_minute=2, trusted_proxy_hops=1) as c:
        for forged in ("6.6.6.1", "6.6.6.2", "6.6.6.3"):
            reply = c.post(
                "/signin",
                data={"email": "maya@example.com", "password": "wrong"},
                headers={"X-Forwarded-For": f"{forged}, 203.0.113.7"},
            )
        assert "Too many attempts" in reply.text
        other = c.post(
            "/signin",
            data={"email": "maya@example.com", "password": "wrong"},
            headers={"X-Forwarded-For": "203.0.113.8"},
        )
        assert "Too many attempts" not in other.text


def test_sign_in_has_a_cap_across_all_addresses(
    sources: DataSources, accounts: list[Account]
) -> None:
    with make_client(
        sources, accounts, signin_attempts_per_minute_total=2, trusted_proxy_hops=1
    ) as c:
        for n in range(3):
            reply = c.post(
                "/signin",
                data={"email": "maya@example.com", "password": "wrong"},
                headers={"X-Forwarded-For": f"203.0.113.{n}"},
            )
        assert "Too many attempts" in reply.text


def test_new_chat_does_not_reset_the_chat_limit(
    sources: DataSources, accounts: list[Account]
) -> None:
    fake = FakeClient(response(text("One.")), response(text("Two.")))
    wren = Coach(fake, coach_name="Wren", model="m")
    with make_client(sources, accounts, coach=wren, chat_messages_per_hour=1) as c:
        sign_in(c)
        assert "One." in c.post("/chat", data={"question": "1"}).text
        c.post("/chat/new")
        assert "a lot of questions" in c.post("/chat", data={"question": "2"}).text


def test_a_question_while_one_is_answered_is_turned_away(
    sources: DataSources, accounts: list[Account]
) -> None:
    import threading

    started, finish = threading.Event(), threading.Event()

    def slow() -> Any:
        started.set()
        finish.wait(5)
        return response(text("Done."))

    class SlowClient(FakeClient):
        def _create(self, **request: Any) -> Any:
            reply = super()._create(**request)
            return reply() if callable(reply) else reply

    wren = Coach(SlowClient(slow), coach_name="Wren", model="m")
    with make_client(sources, accounts, coach=wren) as c:
        sign_in(c)
        results: dict[str, str] = {}
        first = threading.Thread(
            target=lambda: results.update(a=c.post("/chat", data={"question": "1"}).text)
        )
        first.start()
        assert started.wait(5)
        busy = c.post("/chat", data={"question": "2"}).text
        finish.set()
        first.join(5)
    assert "Still answering your last question" in busy
    assert "Done." in results["a"]


def test_pages_render_quickly(client: TestClient) -> None:
    sign_in(client)
    pages = (
        "/",
        "/?month=2026-08&horizon=year",
        "/flow?horizon=week",
        "/transactions",
        "/transactions?review=1&q=a",
        "/drill?category=Dining",
        "/goals",
        "/worth-a-look",
        "/chat",
    )
    for path in pages:
        started = time.perf_counter()
        reply = client.get(path)
        assert reply.status_code == 200, path
        assert time.perf_counter() - started < 2.0, path  # NFR-5, dashboard < 2 s
    headers = client.get("/").headers
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_not_sure_follows_the_review_flags_from_the_batch(
    sources: DataSources, accounts: list[Account], client: TestClient
) -> None:
    """FR-5: the page shows the flags the model's review policy set, not its own threshold."""
    sign_in(client)
    ledger = Ledger.load(sources, accounts[0].user_id)
    month = ledger.between(ledger.as_of.replace(day=1), ledger.as_of)
    flagged = int(month["needs_review"].sum())
    assert 0 < flagged < len(month)

    page = client.get("/transactions").text
    only_flagged = client.get("/transactions?review=1").text

    assert f"Not sure · {flagged}" in page
    assert "below 95% confidence for a merchant it hasn't seen before, below 50%" in page
    assert only_flagged.count('class="raw"') == flagged


def test_a_user_never_sees_another_users_transactions(
    sources: DataSources, accounts: list[Account], client: TestClient
) -> None:
    mine, theirs = (Ledger.load(sources, a.user_id) for a in accounts)
    september = theirs.between(theirs.as_of.replace(day=1), theirs.as_of)
    only_theirs = set(september["merchant_raw"]) - set(mine.transactions["merchant_raw"])
    assert only_theirs, "the fixture users should have some merchants the other doesn't"
    sign_in(client)  # as Maya

    # Identity comes from the session; a user_id in the URL is ignored
    for path in (
        "/transactions",
        f"/transactions?user_id={accounts[1].user_id}",
        f"/?user_id={accounts[1].user_id}",
        "/drill?category=Dining",
    ):
        page = client.get(path).text
        assert "Maya Chen" in page or path.startswith("/drill")
        leaked = [raw for raw in only_theirs if raw in page]
        assert not leaked, (path, leaked[:3])


def test_unknown_categories_and_bad_months_are_handled(client: TestClient) -> None:
    sign_in(client)
    assert client.get("/drill?category=Crypto").status_code == 404
    for month in ("2031-01", "garbage", "2026-13"):
        assert "September 2026 at a glance" in client.get(f"/?month={month}").text


def test_chat_without_a_coach_says_it_is_unavailable(client: TestClient) -> None:
    sign_in(client)
    assert "available right now" in client.get("/chat").text
    reply = client.post("/chat", data={"question": "How much did I spend?"})
    assert "isn&#39;t available right now" in reply.text
    assert client.get("/").status_code == 200  # NFR-6: the dashboard is unaffected


def test_chat_answers_with_source_chips(sources: DataSources, accounts: list[Account]) -> None:
    fake = FakeClient(
        response(
            tool_use("get_spending_summary", {"start_date": "2026-09-01", "end_date": "2026-09-30"})
        ),
        response(text("You spent **$1,000** [S1] in September.\n- One <b>thing</b>")),
        api_error(),
    )
    wren = Coach(fake, coach_name="Wren", model="m")
    with make_client(sources, accounts, coach=wren) as c:
        sign_in(c)
        reply = c.post("/chat", data={"question": "How much did I spend?"}).text
        assert (
            '<span class="src-chip" title="Spending summary · Sep 2026" data-src="S1">1</span>'
            in reply
        )
        assert "<b>$1,000</b>" in reply
        assert "&lt;b&gt;thing&lt;/b&gt;" in reply  # model text is escaped
        assert 'hx-swap-oob="true"' in reply  # the sources panel updates too
        assert "How much did I spend?" in c.get("/chat").text  # kept for a reload

        failed = c.post("/chat", data={"question": "And August?"}).text
        assert "couldn&#39;t be reached" in failed


def test_two_sessions_as_the_same_user_keep_separate_chats(
    sources: DataSources, accounts: list[Account]
) -> None:
    """Visitors share demo accounts, so chat is per session, never per user (NFR-2)."""
    fake = FakeClient(response(text("Answer for the first visitor.")))
    wren = Coach(fake, coach_name="Wren", model="m")
    app = create_app(settings(), sources=sources, accounts=accounts, coach=wren)
    # One client runs the app's startup; the second is another browser on the same server
    with TestClient(app) as first:
        second = TestClient(app)
        sign_in(first)
        sign_in(second)  # the same demo user, another browser
        first.post("/chat", data={"question": "My private question"})

        assert "My private question" in first.get("/chat").text
        other = second.get("/chat").text
        assert "My private question" not in other
        assert "Answer for the first visitor." not in other


def test_chat_is_rate_limited(sources: DataSources, accounts: list[Account]) -> None:
    fake = FakeClient(response(text("One.")), response(text("Two.")))
    wren = Coach(fake, coach_name="Wren", model="m")
    with make_client(sources, accounts, coach=wren, chat_messages_per_hour=1) as c:
        sign_in(c)
        assert "One." in c.post("/chat", data={"question": "1"}).text
        assert "a lot of questions" in c.post("/chat", data={"question": "2"}).text


def test_render_answer_drops_unknown_sources() -> None:
    html = str(render_answer("Spent $5 [S9].", {}))
    assert html == "<p>Spent $5.</p>"


def test_money_formatting() -> None:
    assert money(3840.4) == "$3,840"
    assert money(-6.75, cents=True, sign=True) == MINUS + "$6.75"
    assert money(2600, cents=True, sign=True) == "+$2,600.00"


def test_money_flow_balances() -> None:
    cats = [("Housing", 1650.0), ("Dining", 610.0), ("Shopping", 260.0), ("Travel", -20.0)]
    essentials = frozenset({"Housing", "Groceries"})
    flow = money_flow(5200.0, cats, essentials)
    blocks = {b["label"]: b["amount"] for b in flow["blocks"]}

    assert blocks == {
        "Came in": 5200.0,
        "Essentials": 1650.0,
        "Everything else": 870.0,
        "Left over": 2680.0,
    }
    assert [c["name"] for c in flow["categories"]] == ["Housing", "Dining", "Shopping"]

    overspent = money_flow(1000.0, cats, essentials)
    labels = [b["label"] for b in overspent["blocks"]]
    assert "From savings" in labels
    assert "Left over" not in labels
    assert money_flow(0.0, [], essentials)["empty"] is True


def test_trend_marks_the_selected_month_and_the_peak() -> None:
    bars = trend([("2026-07", 100.0), ("2026-08", 300.0), ("2026-09", 200.0)], "2026-09")
    assert [b["kind"] for b in bars] == ["", "peak", "selected"]


def test_the_shared_password_is_checked_not_stored() -> None:
    check = SharedPassword(PASSWORD)
    assert check.matches(PASSWORD)
    assert not check.matches(PASSWORD + " ")
    assert PASSWORD not in repr(vars(check))
    with pytest.raises(ValueError, match="8 characters"):
        SharedPassword("short")
