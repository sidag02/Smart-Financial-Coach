"""FR-11/FR-12 tools: forecast_goal, check_goal's forecast and fit badge, list_goals' statuses;
a draft gets the saved goal's numbers; reached goals hide the probability."""

import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from smart_financial_coach.access.goal_forecasts import (
    HISTORY_POINTS,
    MONTHLY_POINTS,
    first_entries,
    forecast_fields,
    saved_history,
    thin,
)
from smart_financial_coach.access.goals import Goal, GoalStore, Revision
from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.tools import GoalAccess, Tools
from smart_financial_coach.config import Settings
from smart_financial_coach.data.store import load_goals
from smart_financial_coach.experience.accounts import Account
from smart_financial_coach.experience.web.app import create_app
from smart_financial_coach.intelligence.forecasting.batch import forecast_dataset
from smart_financial_coach.intelligence.forecasting.contract import (
    ON_TRACK,
    history_json,
    parse_history,
)
from smart_financial_coach.intelligence.forecasting.paths import PathsModel, run_with_deposit
from tests.unit.access.test_mcp import SECRET, in_worker_thread, mcp_tools

TODAY = date(2026, 9, 30)
GOALS = pd.DataFrame(
    [
        # (generated) an active goal, a reached one and one whose date has passed
        ("g_x_1", "u", "Vacation fund", 3000.0, "2026-01-15", "2026-12-31", "2026-09-30", 2140.0),
        ("g_x_2", "u", "New laptop", 2100.0, "2026-01-15", "2027-05-31", "2026-09-30", 2100.0),
        ("g_x_3", "u", "College fund", 5900.0, "2024-05-15", "2025-01-31", "2024-09-30", 5606.21),
    ],
    columns=[
        "goal_id",
        "user_id",
        "name",
        "target_amount",
        "created_date",
        "target_date",
        "as_of_date",
        "current_balance",
    ],
)
TRIP = {"name": "Trip", "target_amount": 9000, "target_date": "2028-06-01", "saved": 400}


@pytest.fixture
def ledger(forecast_sources: DataSources, two_users: tuple[str, str]) -> Ledger:
    ledger = replace(Ledger.load(forecast_sources, two_users[0]), goals=GOALS)
    assert ledger.as_of == TODAY
    assert ledger.forecaster is not None
    return ledger


@pytest.fixture
def store(tmp_path: Path) -> GoalStore:
    return GoalStore(tmp_path / "goals.sqlite")


@pytest.fixture
def page(ledger: Ledger, store: GoalStore) -> Tools:
    return Tools(ledger, goals=GoalAccess(store, "s1"))


def by_name(tools: Tools) -> dict[str, dict[str, Any]]:
    return {g["name"]: g for g in tools.call("list_goals", {}).data["goals"]}


def test_list_goals_gives_running_goals_a_status(page: Tools) -> None:
    data = page.call("list_goals", {}).data
    goals = {g["name"]: g for g in data["goals"]}

    assert data["forecast"] == "available"
    vacation = goals["Vacation fund"]
    assert vacation["forecast_status"] in ("on_track", "either_way", "off_track")
    assert 0 <= vacation["p_goal_met"] <= 1
    assert goals["New laptop"]["forecast_status"] == "reached"
    assert goals["New laptop"]["p_goal_met"] is None
    assert "forecast_status" not in goals["College fund"]  # ended: nothing to forecast


def test_forecast_goal_states_the_band_range_and_top_up(page: Tools, ledger: Ledger) -> None:
    result = page.call("forecast_goal", {"goal_id": "g_x_1"})
    f = result.data

    assert f["status"] in ("on_track", "either_way", "off_track")
    assert f["range"]["low"] <= f["projected_balance"] <= f["range"]["high"]
    assert f["range"]["chance"] == 0.8
    assert f["gap"] == pytest.approx(max(0.0, 3000 - f["projected_balance"]), abs=0.01)
    assert f["share_source"] == "track_record"
    assert f["months_of_history"] >= 6
    assert f["short_history"] is False
    assert f["may_draw_down"] is None
    assert f["model_version"] == "fr11-test"
    # The chart's months run to the target month and end at the forecast's numbers
    assert [m["month"] for m in f["monthly"]] == ["2026-10", "2026-11", "2026-12"]
    assert f["monthly"][-1]["median"] == f["projected_balance"]
    # The history: each month since it was created, before this one, under the same share
    assert [h["month"] for h in f["history"]] == [f"2026-{m:02d}" for m in range(1, 9)]
    assert all(0 <= h["saved"] <= 3000 for h in f["history"])
    assert result.source.title == "Goal forecast · Vacation fund"
    # The same answer every time (NFR-8)
    assert page.call("forecast_goal", {"goal_id": "g_x_1"}).data == f


def test_the_top_up_puts_a_goal_on_track(page: Tools, ledger: Ledger) -> None:
    page.call("create_goal", TRIP)
    trip = by_name(page)["Trip"]
    f = page.call("forecast_goal", {"goal_id": trip["goal_id"]}).data
    if f["status"] == "on_track":
        assert f["extra_per_month"] is None
        return
    forecaster = ledger.forecaster
    assert forecaster is not None
    paths = forecaster.paths(ledger.user_id)[:, : trip["months_left"]]
    running = [g for g in page._goals() if g.running(TODAY)]
    out = page._forecasts(running)
    assert out is not None
    share = out[trip["goal_id"]]["share"]

    def chance(deposit: float) -> float:
        return float((run_with_deposit(400.0, share, paths, deposit) >= 9000).mean())

    assert chance(f["extra_per_month"]) >= ON_TRACK
    assert chance(f["extra_per_month"] - 1) < ON_TRACK  # the smallest whole-dollar amount


def test_a_reached_goal_hides_its_chance_range_and_top_up(page: Tools) -> None:
    f = page.call("forecast_goal", {"goal_id": "g_x_2"}).data

    assert f["status"] == "reached"
    for hidden in ("p_goal_met", "range", "gap", "extra_per_month"):
        assert f[hidden] is None
    assert isinstance(f["may_draw_down"], bool)
    assert f["monthly"] == []


def test_an_ended_goal_has_no_forecast(page: Tools) -> None:
    f = page.call("forecast_goal", {"goal_id": "g_x_3"}).data
    assert f["status"] == "ended"
    assert not any(isinstance(v, int | float) for v in f.values())


def test_a_draft_gets_the_numbers_the_saved_goal_gets(page: Tools) -> None:
    """One future per user, and the draft counted among the running goals (§3, §7)."""
    check = page.call("check_goal", TRIP).data
    assert check["fit"] in ("within_reach", "either_way", "stretch")
    assert check["forecast"]["share_source"] == "typical"

    page.call("create_goal", TRIP)
    trip = by_name(page)["Trip"]
    saved = page.call("forecast_goal", {"goal_id": trip["goal_id"]}).data
    assert {k: v for k, v in saved.items() if k in check["forecast"]} == check["forecast"]
    assert trip["p_goal_met"] == check["forecast"]["p_goal_met"]


def test_an_edit_is_checked_as_it_would_be_saved(page: Tools) -> None:
    check = page.call("check_goal", {"goal_id": "g_x_1", "target_amount": 2500}).data
    page.call("update_goal", {"goal_id": "g_x_1", "target_amount": 2500})
    saved = page.call("forecast_goal", {"goal_id": "g_x_1"}).data
    assert {k: v for k, v in saved.items() if k in check["forecast"]} == check["forecast"]


def test_a_new_goal_shares_the_typical_allocation(page: Tools) -> None:
    """The typical total divided by every running goal, the draft included: with two running
    goals and the draft, a third of it (§3)."""
    forecaster = page.ledger.forecaster
    assert forecaster is not None
    rows = page._forecasts([g for g in page._goals() if g.running(TODAY)])
    assert rows is not None
    assert len(rows) == 2
    page.call("create_goal", TRIP)
    trip = by_name(page)["Trip"]
    out = page._forecasts([g for g in page._goals() if g.running(TODAY)])
    assert out is not None
    assert isinstance(forecaster.model, PathsModel)
    assert out[trip["goal_id"]]["share"] == pytest.approx(forecaster.model.typical_total / 3)


def test_an_invalid_draft_has_no_forecast(page: Tools) -> None:
    data = page.call("check_goal", {"name": "Trip", "target_amount": 10}).data
    assert data["valid"] is False
    assert data["forecast"] is None
    assert "fit" not in data


def test_a_short_history_says_so(page: Tools) -> None:
    """Fewer than 6 full months of the user's own history: "a rough guide"."""
    history = parse_history(page._net_history())
    page._history = history_json(history[-4:])
    f = page.call("forecast_goal", {"goal_id": "g_x_1"}).data
    assert f["months_of_history"] == 4
    assert f["short_history"] is True


def test_without_a_model_forecasts_stay_not_available(
    sources: DataSources, two_users: tuple[str, str], store: GoalStore
) -> None:
    tools = Tools(
        replace(Ledger.load(sources, two_users[0]), goals=GOALS), goals=GoalAccess(store, "s1")
    )
    assert tools.call("forecast_goal", {"goal_id": "g_x_1"}).data["status"] == "not_available"
    assert tools.call("list_goals", {}).data["forecast"] == "not_available"
    assert "forecast_status" not in by_name(tools)["Vacation fund"]
    assert tools.call("check_goal", TRIP).data["forecast"] == "not_available"


def test_first_entries_skip_undone_and_archived_changes() -> None:
    def rev(seq: int, goal: str, op: str, cents: int, day: str, undone: bool = False) -> Revision:
        return Revision(
            revision_id=f"r{seq}",
            seq=seq,
            subject="s",
            user_id="u",
            goal_id=goal,
            op=op,
            name="Trip",
            target_amount_cents=900_000,
            target_date="2028-06-30",
            saved_cents=cents,
            saved_as_of=day,
            created_date="2026-01-31",
            source="edit",
            created_at="",
            undone_at="x" if undone else None,
        )

    revisions = [
        rev(3, "a", "update", 90_000, "2026-09-30"),
        rev(1, "a", "create", 40_000, "2026-01-31", undone=True),
        rev(2, "a", "update", 50_000, "2026-03-31"),
        rev(4, "b", "archive", 10_000, "2026-02-28"),
        rev(5, "b", "update", 20_000, "2026-04-30"),
    ]
    assert first_entries(revisions) == {
        "a": (50_000, date(2026, 3, 31)),
        "b": (20_000, date(2026, 4, 30)),
    }


def test_long_goals_thin_the_monthly_series_and_keep_the_target_month() -> None:
    months = [{"month": str(i)} for i in range(119)]
    thinned = thin(months)
    assert len(thinned) <= MONTHLY_POINTS
    assert thinned[-1] == months[-1]
    assert thin(months[:10]) == months[:10]


def test_forecast_fields_round_and_hide() -> None:
    out = {
        "status": "reached",
        "p_goal_met": 0.93,
        "projected_balance": 2300.456,
        "range_lo": 2100.0,
        "range_hi": 2500.0,
        "gap": 0.0,
        "extra_per_month": np.nan,
        "share_source": "track_record",
        "may_draw_down": np.True_,
        "model_version": "v",
    }
    f = forecast_fields(out, 30, lambda v: round(float(v), 2), target=2000.0)
    assert f["projected_balance"] == 2300.46
    assert f["may_draw_down"] is True
    assert f["p_goal_met"] is None


@pytest.fixture
def saver(forecast_sources: DataSources) -> str:
    """A user with a goal still running at the dataset's end."""
    goals = load_goals(forecast_sources.dataset)
    return str(goals.loc[goals["target_date"] > TODAY.isoformat(), "user_id"].iloc[0])


@pytest.fixture
def forecast_client(forecast_sources: DataSources, saver: str) -> Iterator[TestClient]:
    settings = Settings(
        _env_file=None,
        demo_password=SecretStr("correct horse"),
        session_secret=SecretStr(SECRET),
        secure_cookies=False,
    )
    accounts = [Account(saver, "Maya Chen", "maya@x.com")]
    with TestClient(create_app(settings, sources=forecast_sources, accounts=accounts)) as c:
        yield c


def test_mcp_gives_the_same_forecast_as_the_tools(
    forecast_client: TestClient, forecast_sources: DataSources, saver: str
) -> None:
    remote = mcp_tools(forecast_client, saver)
    listed = in_worker_thread(forecast_client, lambda: remote.call("list_goals", {})).data
    running = [g for g in listed["goals"] if g["status"] in ("active", "reached")]
    assert running
    assert listed["forecast"] == "available"
    local = Tools(Ledger.load(forecast_sources, saver))
    for goal in running:
        args = {"goal_id": goal["goal_id"]}
        over_mcp = in_worker_thread(forecast_client, partial(remote.call, "forecast_goal", args))
        assert over_mcp.data == local.call("forecast_goal", args).data


@pytest.fixture
def baseline_page(
    sources: DataSources, two_users: tuple[str, str], store: GoalStore, tmp_path: Path
) -> Tools:
    """No model promoted: naive pace behind the same pipeline (owner decision 10 on #54)."""
    forecast_dataset(sources.dataset, tmp_path / "forecasts.json", baseline=True)
    baseline = DataSources(
        sources.dataset, sources.predictions, sources.flags, tmp_path / "forecasts.json"
    )
    ledger = replace(Ledger.load(baseline, two_users[0]), goals=GOALS)
    return Tools(ledger, goals=GoalAccess(store, "s1"))


def test_the_baseline_is_a_simple_projection_with_no_chance(baseline_page: Tools) -> None:
    f = baseline_page.call("forecast_goal", {"goal_id": "g_x_1"}).data
    assert f["method"] == "simple_projection"
    assert f["status"] in ("on_track", "off_track")
    assert f["p_goal_met"] is None
    assert f["range"] is None
    assert f["monthly"] == []
    assert f["share_source"] is None
    # The pace so far, extended: $2,140 over Jan-Sep (9 months), 3 more months
    assert f["projected_balance"] == pytest.approx(2140 + 2140 / 9 * 3, abs=0.01)
    # Its history is the same pace: a straight line from creation, Jan to Aug
    assert [h["saved"] for h in f["history"]] == [round(2140 * k / 9, 2) for k in range(1, 9)]
    vacation = by_name(baseline_page)["Vacation fund"]
    assert vacation["forecast_method"] == "simple_projection"
    assert vacation["p_goal_met"] is None


def test_the_baseline_gives_no_fit_and_no_projection_for_a_new_goal(baseline_page: Tools) -> None:
    assert baseline_page.call("check_goal", TRIP).data["forecast"] == "not_available"
    edit = baseline_page.call("check_goal", {"goal_id": "g_x_1", "target_amount": 2500}).data
    assert edit["forecast"]["method"] == "simple_projection"
    assert edit["fit"] is None


def test_a_goal_you_keep_up_gets_a_share_from_your_entries(page: Tools, store: GoalStore) -> None:
    """An FR-10 goal with two saved entries 3+ months apart: the share comes from them (§3)."""
    page.call("create_goal", {**TRIP, "saved": 400})
    trip = by_name(page)["Trip"]
    with sqlite3.connect(store.path) as conn:  # the first entry, six months ago
        conn.execute(
            "UPDATE goal_revisions SET saved_as_of = '2026-03-31', created_date = '2026-03-31' "
            "WHERE goal_id = ?",
            (trip["goal_id"],),
        )
    page.call("update_goal", {"goal_id": trip["goal_id"], "saved": 2400})
    f = page.call("forecast_goal", {"goal_id": trip["goal_id"]}).data
    assert f["share_source"] == "your_entries"
    # Its history starts at the first entry and runs to last month
    assert f["history"][0] == {"month": "2026-03", "saved": 400.0}
    assert [h["month"] for h in f["history"]][-1] == "2026-08"


def test_a_new_goal_has_only_its_first_entry_as_history(page: Tools) -> None:
    page.call("create_goal", TRIP)
    f = page.call("forecast_goal", {"goal_id": by_name(page)["Trip"]["goal_id"]}).data
    assert f["share_source"] == "typical"
    assert f["history"] == []  # entered this month: today's amount is the chart's "Now"


def test_long_histories_are_thinned_and_keep_last_month() -> None:
    goal = Goal(
        "g", "Fund", 900_000, date(2027, 12, 31), 500_000, TODAY, date(2022, 1, 15), "existing"
    )
    net = pd.Series(100.0, index=pd.period_range("2022-01", "2026-09", freq="M"))
    history = saved_history(goal, 0.5, "track_record", net, None, TODAY)
    assert len(history) <= HISTORY_POINTS
    assert history[-1]["month"] == "2026-08"


def test_forecasting_one_goal_matches_the_whole_set(page: Tools) -> None:
    running = [g for g in page._goals() if g.running(TODAY)]
    whole = page._forecasts(running)
    one = page._forecasts(running, only="g_x_1")
    assert whole is not None
    assert one is not None
    assert list(one) == ["g_x_1"]
    assert one["g_x_1"] == whole["g_x_1"]
