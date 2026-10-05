"""'Worth a look' with alert sensitivity and flag actions (FR-9 §4): the switch, the actions,
undo, hidden alerts, and one visitor's choices staying their own."""

from collections.abc import Iterator
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.tools import Tools
from smart_financial_coach.data import store
from smart_financial_coach.experience.accounts import Account
from tests.unit.experience.test_web import make_client, sign_in

WINDOW_DAYS = 60


@pytest.fixture(scope="module")
def alert_sources(
    preset_flagged_sources: DataSources, preset_spike_sources: DataSources
) -> DataSources:
    return DataSources(
        preset_flagged_sources.dataset,
        preset_flagged_sources.predictions,
        preset_flagged_sources.flags,
        spikes=preset_spike_sources.spikes,
    )


@pytest.fixture(scope="module")
def recent_user(alert_sources: DataSources) -> str:
    """A user with unusual charges in the page's 60 days at Balanced."""
    for user in store.load_users(alert_sources.dataset)["user_id"]:
        ledger = Ledger.load(alert_sources, user)
        start = ledger.as_of - timedelta(days=WINDOW_DAYS - 1)
        found = Tools(ledger).detect_anomalies(start.isoformat(), ledger.as_of.isoformat()).data
        if found["count"]:
            return str(user)
    pytest.skip("no user with a recent flag in the small data")


@pytest.fixture
def client(alert_sources: DataSources, recent_user: str) -> Iterator[TestClient]:
    accounts = [Account(recent_user, "Maya Chen", "maya@example.com")]
    with make_client(alert_sources, accounts) as c:
        sign_in(c)
        yield c


def _first_flag_id(page: str) -> str:
    marker = 'name="flag_id" value="'
    start = page.index(marker) + len(marker)
    return page[start : page.index('"', start)]


def test_the_page_has_the_switch_and_the_actions(client: TestClient) -> None:
    page = client.get("/worth-a-look").text
    assert "How often should we point things out?" in page
    assert 'value="balanced" aria-pressed="true"' in page
    assert "I recognize this" in page
    assert "Not me — what now?" in page


def test_the_switch_changes_the_level(client: TestClient) -> None:
    reply = client.post("/worth-a-look/sensitivity", data={"level": "more"}, follow_redirects=False)
    assert reply.status_code == 303
    assert reply.headers["location"] == "/worth-a-look"
    page = client.get("/worth-a-look").text
    assert "Alerts: More often." in page  # the toast
    assert 'value="more" aria-pressed="true"' in page
    assert "more of them will turn out to be ordinary" in page


def test_recognize_hides_with_undo_and_shows_on_request(client: TestClient) -> None:
    flag_id = _first_flag_id(client.get("/worth-a-look").text)
    reply = client.post(
        "/alerts/act", data={"flag_id": flag_id, "action": "recognize"}, follow_redirects=False
    )
    assert reply.status_code == 303

    page = client.get("/worth-a-look").text
    assert f'name="flag_id" value="{flag_id}"' not in page  # no longer listed
    assert "you recognized" in page
    assert "/alerts/undo" in page  # the toast's Undo

    shown = client.get("/worth-a-look?hidden=1").text
    assert "Show it again" in shown
    action_id = shown.split('name="action_id" value="')[1].split('"')[0]
    reply = client.post("/alerts/undo", data={"action_id": action_id})
    assert "Undone: the alert shows again." in reply.text


def test_not_me_shows_the_guidance(client: TestClient) -> None:
    flag_id = _first_flag_id(client.get("/worth-a-look").text)
    page = client.post("/alerts/act", data={"flag_id": flag_id, "action": "not_me"}).text
    assert "You said this wasn" in page
    assert "Contact your card issuer or bank" in page
    assert "block cards or dispute charges" in page


def test_one_visitors_choices_stay_their_own(
    alert_sources: DataSources, recent_user: str, client: TestClient
) -> None:
    client.post("/worth-a-look/sensitivity", data={"level": "less"})
    accounts = [Account(recent_user, "Maya Chen", "maya@example.com")]
    with make_client(alert_sources, accounts) as other:
        sign_in(other)
        assert 'value="balanced" aria-pressed="true"' in other.get("/worth-a-look").text


def test_bad_input_is_reported_not_raised(client: TestClient) -> None:
    # Each post redirects back to the page, which shows the toast once
    reply = client.post("/worth-a-look/sensitivity", data={"level": "loud"})
    assert "Couldn&#39;t change that" in reply.text
    reply = client.post("/alerts/act", data={"flag_id": "nope", "action": "recognize"})
    assert "Couldn&#39;t do that" in reply.text
    reply = client.post(
        "/alerts/act",
        data={"flag_id": "nope", "action": "recognize", "back": "https://example.com"},
        follow_redirects=False,
    )
    assert reply.headers["location"] == "/worth-a-look"  # only this app's pages


def test_actions_need_a_signed_in_user(alert_sources: DataSources, recent_user: str) -> None:
    accounts = [Account(recent_user, "Maya Chen", "maya@example.com")]
    with make_client(alert_sources, accounts) as anonymous:
        for path in ("/worth-a-look/sensitivity", "/alerts/act", "/alerts/undo"):
            reply = anonymous.post(path, follow_redirects=False)
            assert reply.headers["location"] == "/signin"
