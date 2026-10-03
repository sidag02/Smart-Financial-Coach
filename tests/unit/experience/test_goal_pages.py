"""FR-10 pages: the Goals page, setup and edit with the live check, remove and undo, overview."""

import re
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.tools import Tools
from smart_financial_coach.experience.accounts import Account
from tests.unit.experience.test_web import make_client, sign_in

TRIP = {"name": "Trip", "target_amount": "$3,000", "target_month": "2027-06", "saved": ""}
HX = {"HX-Request": "true"}


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


@pytest.fixture
def generated(sources: DataSources, accounts: list[Account]) -> list[dict[str, object]]:
    """Maya's generated goals, as the tools list them with no changes."""
    goals: list[dict[str, object]] = (
        Tools(Ledger.load(sources, accounts[0].user_id)).list_goals().data["goals"]
    )
    return goals


def create(client: TestClient, **fields: str) -> str:
    """Create a goal through the form; returns the Goals page it lands on."""
    reply = client.post("/goals", data={**TRIP, **fields}, follow_redirects=False)
    assert reply.status_code == 303, reply.text
    assert reply.headers["location"] == "/goals"
    return client.get("/goals").text


def goal_id(page: str, name: str) -> str:
    match = re.search(rf"<b>{re.escape(name)}</b>.*?/goals/(gu?_[\w]+)/edit", page, re.S)
    assert match, f"no {name} card"
    return match.group(1)


def undo_id(page: str) -> str:
    match = re.search(r'class="toast".*?name="revision_id" value="(\w+)"', page, re.S)
    assert match, "no undo in the toast"
    return match.group(1)


def test_the_goals_page_lists_running_and_ended_goals(
    client: TestClient, generated: list[dict[str, object]]
) -> None:
    page = client.get("/goals").text

    assert "New goal" in page
    assert "Coming next" not in page
    for g in generated:
        assert str(g["name"]) in page
    if any(g["status"] == "ended" for g in generated):
        assert "Ended" in page
    assert "a median of" in page
    nav = page[page.index('aria-label="Main"') :]
    assert '/goals" aria-current="page">Goals</a>' in nav  # no "Soon" badge any more


def test_a_goal_set_up_through_the_form_shows_with_an_undo(client: TestClient) -> None:
    page = create(client)

    assert "Created Trip." in page
    assert "$3,000 by Jun 30, 2027" in page
    assert "$334 a month" in page  # $3,000 over 9 months, rounded up
    undone = client.post("/goals/undo", data={"revision_id": undo_id(page)})
    assert "Undone." in undone.text
    assert "<b>Trip</b>" not in undone.text


def test_a_refused_goal_shows_each_problem_by_its_field(client: TestClient) -> None:
    reply = client.post("/goals", data={**TRIP, "target_amount": "20", "target_month": ""})

    assert reply.status_code == 200
    assert "Goals start at $50." in reply.text
    assert "Pick a month." in reply.text
    assert 'value="20"' in reply.text  # what was typed stays
    assert "<b>Trip</b>" not in client.get("/goals").text


def test_with_htmx_the_form_opens_in_the_drawer_and_redirects_on_success(
    client: TestClient,
) -> None:
    form = client.get("/goals/new", headers=HX).text
    assert form.lstrip().startswith('<div class="drawer"')
    assert 'hx-post="/goals"' in form

    refused = client.post("/goals", data={**TRIP, "name": ""}, headers=HX)
    assert refused.status_code == 200
    assert refused.text.lstrip().startswith('<div class="drawer"')
    assert "Give the goal a name." in refused.text

    done = client.post("/goals", data=TRIP, headers=HX)
    assert (done.status_code, done.headers["HX-Redirect"]) == (204, "/goals")
    assert "Created Trip." in client.get("/goals").text


def test_without_htmx_the_form_is_a_page_of_its_own(client: TestClient) -> None:
    page = client.get("/goals/new").text
    assert '<h1 class="page-title">New savings goal</h1>' in page
    assert 'hx-post="/goals"' not in page  # a plain post
    assert 'action="/goals"' in page


def test_the_live_check_states_the_facts(client: TestClient) -> None:
    fit = client.post("/goals/check", data=TRIP).text

    assert "$334" in fit
    assert "a month to get there by Jun 30, 2027 (9 months)" in fit
    assert "a median of" in fit


def test_the_live_check_only_complains_about_fields_filled_in(client: TestClient) -> None:
    empty = client.post("/goals/check", data={"name": "Trip"}).text
    assert "field-error" not in empty
    assert "Fill in an amount and a month" in empty

    bad = client.post("/goals/check", data={"name": "Trip", "target_amount": "10"}).text
    assert "Goals start at $50." in bad
    assert "Pick a month." not in bad


def test_editing_a_goal(client: TestClient) -> None:
    gid = goal_id(create(client), "Trip")

    form = client.get(f"/goals/{gid}/edit").text
    assert 'value="Trip"' in form
    assert 'value="2027-06"' in form
    assert 'value="3000"' in form

    reply = client.post(f"/goals/{gid}", data={**TRIP, "name": "Lisbon", "saved": "300"})
    assert "Saved Lisbon." in reply.text
    assert '<span class="mid">$300</span> <span class="muted">of $3,000' in reply.text
    assert "<b>Trip</b>" not in reply.text

    blank = client.post(f"/goals/{gid}", data={**TRIP, "name": ""})
    assert "Give the goal a name." in blank.text


def test_removing_and_restoring_a_goal(client: TestClient) -> None:
    gid = goal_id(create(client), "Trip")

    removed = client.post(f"/goals/{gid}/archive").text
    assert "Removed Trip." in removed
    assert "Removed goals (1)" in removed
    assert "<b>Trip</b> <span" in removed  # listed under Removed goals, not as a card

    # After the toast is gone (a reload), the removal can still be undone from the list
    page = client.get("/goals").text
    assert 'class="toast"' not in page
    restore = re.search(r'Removed goals.*?name="revision_id" value="(\w+)"', page, re.S)
    assert restore
    restored = client.post("/goals/undo", data={"revision_id": restore.group(1)}).text
    assert "Undone: Trip." in restored
    assert "Removed goals" not in restored


def test_an_undo_that_breaks_a_rule_says_so(client: TestClient) -> None:
    gid = goal_id(create(client), "Trip")
    client.post(f"/goals/{gid}/archive")
    page = create(client)  # a new "Trip"
    old = re.search(r'Removed goals.*?name="revision_id" value="(\w+)"', page, re.S)
    assert old
    reply = client.post("/goals/undo", data={"revision_id": old.group(1)}).text
    assert "Couldn&#39;t undo that: You already have a goal called Trip." in reply


def test_another_session_on_the_same_account_sees_none_of_it(
    client: TestClient, sources: DataSources, accounts: list[Account]
) -> None:
    gid = goal_id(create(client), "Trip")

    with make_client(sources, accounts) as other:
        sign_in(other)
        assert "<b>Trip</b>" not in other.get("/goals").text
        assert other.get(f"/goals/{gid}/edit").status_code == 404
        assert other.post(f"/goals/{gid}", data=TRIP).status_code == 404
        removed = other.post(f"/goals/{gid}/archive")  # a neutral toast, nothing removed
        assert "That goal isn&#39;t there any more." in removed.text
    assert "<b>Trip</b>" in client.get("/goals").text


def test_ended_goals_cant_be_edited(client: TestClient, generated: list[dict[str, object]]) -> None:
    ended = [g for g in generated if g["status"] == "ended"]
    if not ended:
        pytest.skip("this user has no ended goal")
    assert client.get(f"/goals/{ended[0]['goal_id']}/edit").status_code == 404


def test_the_overview_card_shows_the_goal_due_soonest(
    client: TestClient, generated: list[dict[str, object]]
) -> None:
    for g in generated:  # start from no goals
        if g["status"] != "ended":
            client.post(f"/goals/{g['goal_id']}/archive")
    assert "Set up a savings goal" in client.get("/").text

    bike = goal_id(create(client, name="Bike", target_amount="900", target_month="2027-03"), "Bike")
    client.post(
        f"/goals/{bike}", data={**TRIP, "name": "Bike", "target_amount": "900", "saved": "900"}
    )
    reached = client.get("/").text
    assert '<div class="card-label">Bike</div><span class="badge good">Reached</span>' in reached

    create(client)
    card = client.get("/").text
    assert '<div class="card-label">Trip</div>' in card  # active beats reached
    assert "$334 a month" in card


def test_saving_an_edit_unchanged_keeps_the_amounts_exactly(client: TestClient) -> None:
    """The edit form is filled with exact amounts, not 6 significant digits (review on #45)."""
    page = create(client, name="House", target_amount="123,456.78", saved="12,345.67")
    gid = goal_id(page, "House")
    form = client.get(f"/goals/{gid}/edit").text
    assert 'value="123456.78"' in form
    assert 'value="12345.67"' in form

    values = dict(re.findall(r'name="(\w+)"[^>]*?value="([^"]*)"', form))
    keep = {k: values[k] for k in ("name", "target_amount", "target_month", "saved")}
    after = client.post(f"/goals/{gid}", data=keep).text
    assert '<span class="mid">$12,346</span> <span class="muted">of $123,457' in after
    reopened = client.get(f"/goals/{gid}/edit").text
    assert 'value="123456.78"' in reopened
    assert 'value="12345.67"' in reopened
    assert "Saved as of Sep 30, 2026" in after


def test_amounts_must_be_plain_decimals(client: TestClient) -> None:
    started = time.perf_counter()
    for amount in ("1e100000000", "1e999990", "3e3", "0x10", "12.5.1"):
        fit = client.post("/goals/check", data={**TRIP, "target_amount": amount}).text
        assert "Enter an amount in dollars." in fit, amount
        refused = client.post("/goals", data={**TRIP, "target_amount": amount})
        assert refused.status_code == 200
        assert "Enter an amount in dollars." in refused.text
    assert time.perf_counter() - started < 5  # none of them is costly to read


def test_the_overview_card_picks_the_active_goal_due_soonest(
    client: TestClient, generated: list[dict[str, object]]
) -> None:
    for g in generated:
        if g["status"] != "ended":
            client.post(f"/goals/{g['goal_id']}/archive")
    create(client, name="Later", target_month="2028-01")
    create(client, name="Sooner", target_month="2027-02")
    card = client.get("/").text
    assert '<div class="card-label">Sooner</div>' in card
    assert "Later" not in card


def test_goal_names_are_escaped(client: TestClient) -> None:
    page = create(client, name="<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page  # the card and the toast
    assert "<script>alert(1)</script>" not in client.get("/").text


def test_removing_twice_says_its_gone(client: TestClient) -> None:
    gid = goal_id(create(client), "Trip")
    client.post(f"/goals/{gid}/archive")
    again = client.post(f"/goals/{gid}/archive", follow_redirects=False)
    assert again.status_code == 303
    assert "That goal isn&#39;t there any more." in client.get("/goals").text


def test_the_live_check_on_an_edit_leaves_the_goal_out_of_other_goals(client: TestClient) -> None:
    gid = goal_id(create(client), "Trip")
    fit = client.post("/goals/check", data={**TRIP, "target_amount": "3600", "goal_id": gid}).text
    assert "$400" in fit  # $3,600 over 9 months
    assert "You already have a goal called Trip." not in fit  # not clashing with itself
    unknown = client.post("/goals/check", data={**TRIP, "goal_id": "gu_nobody"})
    assert unknown.status_code == 404
