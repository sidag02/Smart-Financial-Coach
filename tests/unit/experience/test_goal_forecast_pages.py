"""FR-11 pages: goal statuses on the cards and the overview, the goal detail (1g, 1l), reached
goals, the fit badge, the short-history notice and the assumption line."""

import re
from collections.abc import Iterator
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from smart_financial_coach.access import ledger as ledger_module
from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.tools import Tools
from smart_financial_coach.data.store import load_goals
from smart_financial_coach.experience.accounts import Account
from smart_financial_coach.experience.web.app import chance
from smart_financial_coach.experience.web.charts import goal_projection
from smart_financial_coach.intelligence.forecasting.batch import GoalForecaster
from tests.unit.experience.test_web import make_client, sign_in

ASSUMPTION = "draws on what you've set aside"
HX = {"HX-Request": "true"}
TRIP = {"name": "Trip", "target_amount": "9000", "target_month": "2028-06", "saved": "400"}


@pytest.fixture
def saver(forecast_sources: DataSources) -> str:
    """A user with a goal still running at the dataset's end."""
    goals = load_goals(forecast_sources.dataset)
    return str(goals.loc[goals["target_date"] > "2026-09-30", "user_id"].iloc[0])


@pytest.fixture
def client(forecast_sources: DataSources, saver: str) -> Iterator[TestClient]:
    with make_client(forecast_sources, [Account(saver, "Maya Chen", "maya@example.com")]) as c:
        sign_in(c)
        yield c


@pytest.fixture
def tools(forecast_sources: DataSources, saver: str) -> Tools:
    return Tools(Ledger.load(forecast_sources, saver))


def running(tools: Tools) -> list[dict[str, object]]:
    goals = tools.list_goals().data["goals"]
    return [g for g in goals if g["status"] in ("active", "reached")]


LABELS = {
    "on_track": "On track",
    "either_way": "Could go either way",
    "off_track": "Off track",
    "reached": "Reached",
}


def test_cards_show_each_goals_status(client: TestClient, tools: Tools) -> None:
    page = client.get("/goals").text
    for g in running(tools):
        assert f"<b>{g['name']}</b>" in page
        assert LABELS[str(g["forecast_status"])] in page
    assert "Forecast soon" not in page
    assert ASSUMPTION in page
    overview = client.get("/").text
    assert "Forecast soon" not in overview
    assert "See the forecast" in overview


def test_the_detail_page_states_the_forecast(client: TestClient, tools: Tools) -> None:
    g = next(g for g in running(tools) if g["status"] == "active")
    f = tools.forecast_goal(str(g["goal_id"])).data
    page = client.get(f"/goals/{g['goal_id']}").text

    assert LABELS[f["status"]] in page
    assert "We're fairly confident you'll land between" in page
    for amount in (f["range"]["low"], f["range"]["high"], f["projected_balance"]):
        assert f"${amount:,.0f}" in page
    assert page.count('class="band"') == len(f["monthly"])
    assert ASSUMPTION in page
    if f["extra_per_month"]:
        assert f"Set aside ${f['extra_per_month']:,.0f} more a month" in page
    else:
        assert "You're on track" in page


def test_a_reached_goal_shows_no_chance_or_range(client: TestClient, tools: Tools) -> None:
    g = next(g for g in running(tools) if g["status"] == "active")
    client.post(
        f"/goals/{g['goal_id']}",
        data={
            "name": g["name"],
            "target_amount": str(g["target_amount"]),
            "target_month": str(g["target_date"])[:7],
            "saved": str(g["target_amount"]),
        },
    )
    page = client.get(f"/goals/{g['goal_id']}").text
    assert "You've reached it." in page
    assert "fairly confident" not in page
    assert "Set aside" not in page
    assert 'class="band"' not in page


def test_the_setup_check_shows_the_fit_badge(client: TestClient) -> None:
    fit = client.post("/goals/check", data={**TRIP, "goal_id": ""}, headers=HX).text
    assert re.search(r"Within reach|Could go either way|A stretch", fit)
    assert "you'd likely have" in fit


def test_a_short_history_is_called_a_rough_guide(
    forecast_sources: DataSources, saver: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = Ledger.load(forecast_sources, saver)
    assert real.forecaster is not None
    states = {u: replace(s, months=4) for u, s in real.forecaster.states.items()}
    short = GoalForecaster(real.forecaster.model, states, real.forecaster.as_of)
    monkeypatch.setattr(DataSources, "forecaster", lambda self: short)
    ledger_module._load.cache_clear()  # ledgers are cached with the real forecaster
    try:
        with make_client(forecast_sources, [Account(saver, "Maya", "maya@example.com")]) as c:
            sign_in(c)
            active = running(Tools(Ledger.load(forecast_sources, saver)))
            gid = next(g["goal_id"] for g in active if g["status"] == "active")
            page = c.get(f"/goals/{gid}").text
    finally:
        ledger_module._load.cache_clear()
    assert "Based on only 4 months of your history, so this is a rough guide." in page


def test_without_a_model_the_detail_page_says_so(
    sources: DataSources, two_users: tuple[str, str]
) -> None:
    with make_client(sources, [Account(two_users[0], "Maya", "maya@example.com")]) as c:
        sign_in(c)
        c.post("/goals", data={**TRIP})
        page = c.get("/goals").text
        gid = re.search(r"<b>Trip</b>.*?/goals/(gu_\w+)/edit", page, re.S)
        assert gid
        detail = c.get(f"/goals/{gid.group(1)}").text
        assert "arrives with goal forecasting" in detail
        assert c.get("/goals/g_nobody_1").status_code == 404


def test_chances_are_said_the_way_people_say_them() -> None:
    assert chance(0.72) == "about a 7 in 10 chance"
    assert chance(0.97) == "better than a 9 in 10 chance"
    assert chance(0.01) == "less than a 1 in 10 chance"
    assert chance(0.06) == "about a 1 in 10 chance"


def test_the_projection_chart_scales_to_the_target_and_ranges() -> None:
    monthly = [{"month": "2026-10", "median": 500.0, "low": 400.0, "high": 1000.0}]
    chart = goal_projection(200.0, 800.0, monthly)
    assert chart["columns"][0]["now"]
    assert chart["target"] == "72.73%"  # 800 of 1,000 x 1.1
    assert chart["columns"][1]["band_height"] == "54.55%"
