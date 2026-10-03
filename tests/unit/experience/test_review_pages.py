"""FR-5/FR-6 on the transactions page: review, correct, undo, per-session isolation (mockup 1e)."""

import re
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.review_items import open_review_items
from smart_financial_coach.experience.accounts import Account
from tests.unit.experience.test_web import make_client, sign_in


@pytest.fixture
def accounts(two_users: tuple[str, str]) -> list[Account]:
    return [
        Account(two_users[0], "Maya Chen", "maya@example.com"),
        Account(two_users[1], "Ada Okafor", "ada@example.com"),
    ]


@pytest.fixture
def client(sources: DataSources, accounts: list[Account]) -> Iterator[TestClient]:
    with make_client(sources, accounts) as c:
        sign_in(c)
        yield c


def top_item(sources: DataSources, user: str) -> dict[str, object]:
    ledger = Ledger.load(sources, user)
    item = open_review_items(ledger.transactions, user).iloc[0]
    return dict(item)


def other(category: object) -> str:
    return "Travel" if category != "Travel" else "Entertainment"


def open_count(page: str) -> int:
    found = re.search(r"We weren't confident about (\d+) merchant", page)
    return int(found.group(1)) if found else 0


def test_the_page_shows_the_review_panel(
    client: TestClient, sources: DataSources, accounts: list[Account]
) -> None:
    item = top_item(sources, accounts[0].user_id)
    page = client.get("/transactions").text

    assert "Help us get these right" in page
    assert open_count(page) == len(
        open_review_items(Ledger.load(sources, accounts[0].user_id).transactions, "x")
    )
    assert f'action="/review/{item["item_id"]}"' in page
    assert f"✓ {item['suggested_category']}" in page


def test_correcting_an_item_then_undoing_it(
    client: TestClient, sources: DataSources, accounts: list[Account]
) -> None:
    item = top_item(sources, accounts[0].user_id)
    before = client.get("/transactions").text
    target = other(item["suggested_category"])

    reply = client.post(
        f"/review/{item['item_id']}",
        data={"action": "correct", "category": target, "back": "/transactions?review=1"},
        follow_redirects=False,
    )
    assert reply.status_code == 303
    assert reply.headers["location"] == "/transactions?review=1"
    after = client.get("/transactions").text

    assert f"at {item['merchant']} to {target}." in after  # the flash
    assert open_count(after) == open_count(before) - 1
    assert f'action="/review/{item["item_id"]}"' not in after
    undo = re.search(r'action="(/corrections/[0-9a-f]+/undo)"', after)
    assert undo is not None
    restored = client.post(undo.group(1), data={"back": "/transactions"}).text
    assert open_count(restored) == open_count(before)
    assert "is back to how it was" in restored
    assert "· undone" in restored


def test_two_visitors_on_one_account_keep_their_own_corrections(
    client: TestClient, sources: DataSources, accounts: list[Account]
) -> None:
    """The demo's shared accounts: feedback is per browser session (owner, Oct 3, 2026)."""
    item = top_item(sources, accounts[0].user_id)
    visitor = TestClient(client.app)  # another browser; the app is already running
    sign_in(visitor)  # the same demo account
    client.post(f"/review/{item['item_id']}", data={"action": "confirm"})

    mine, theirs = client.get("/transactions").text, visitor.get("/transactions").text

    assert open_count(mine) == open_count(theirs) - 1
    assert f'action="/review/{item["item_id"]}"' in theirs
    assert "Your recent changes" in mine
    assert "Your recent changes" not in theirs


def test_one_transaction_can_be_moved_alone(
    client: TestClient, sources: DataSources, accounts: list[Account]
) -> None:
    ledger = Ledger.load(sources, accounts[0].user_id)
    t = ledger.transactions
    row = t[t["merchant_key"] == t["merchant_key"].value_counts().index[0]].iloc[0]
    target = other(row["category"])

    page = client.post(
        f"/transactions/{row['transaction_id']}/category",
        data={"category": target, "scope": "transaction"},
    ).text

    assert f"Moved 1 transaction at {row['merchant']} to {target}." in page
    assert "Corrected, just one transaction" in page


def test_bad_requests_change_nothing(client: TestClient, sources: DataSources) -> None:
    reply = client.post(
        "/review/nope",
        data={"action": "confirm", "back": "https://evil.example/"},
        follow_redirects=False,
    )
    assert reply.headers["location"] == "/transactions"  # never off-site
    assert "Couldn&#39;t change that: no open review item" in client.get("/transactions").text
    reply = client.post(
        "/review/x", data={"action": "confirm", "back": "//evil.example/"}, follow_redirects=False
    )
    assert reply.headers["location"] == "/transactions"
    stranger = TestClient(client.app)
    reply = stranger.post("/review/x", data={"action": "confirm"}, follow_redirects=False)
    assert reply.headers["location"] == "/signin"
