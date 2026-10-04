"""FR-10 tools: list, check, create, update, archive and undo goals; confirmation; read-only."""

from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from smart_financial_coach.access.goals import GoalStore, median_monthly_savings
from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.tools import (
    GoalAccess,
    GoalProblemsError,
    ToolError,
    Tools,
)

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
TRIP = {"name": "Trip", "target_amount": 3000, "target_date": "2027-06-01"}


@pytest.fixture
def ledger(sources: DataSources, two_users: tuple[str, str]) -> Ledger:
    ledger = replace(Ledger.load(sources, two_users[0]), goals=GOALS)
    assert ledger.as_of == TODAY
    return ledger


@pytest.fixture
def store(tmp_path: Path) -> GoalStore:
    return GoalStore(tmp_path / "goals.sqlite")


def tools_as(ledger: Ledger, store: GoalStore, source: str = "edit", subject: str = "s1") -> Tools:
    return Tools(ledger, goals=GoalAccess(store, subject, source))


@pytest.fixture
def page(ledger: Ledger, store: GoalStore) -> Tools:
    """The Goals page: writes apply on submit."""
    return tools_as(ledger, store)


@pytest.fixture
def coach(ledger: Ledger, store: GoalStore) -> Tools:
    return tools_as(ledger, store, "coach")


def goals_by_name(tools: Tools, **args: Any) -> dict[str, dict[str, Any]]:
    return {g["name"]: g for g in tools.call("list_goals", args).data["goals"]}


def full_months(ledger: Ledger) -> pd.Series:
    """An independent count: net per month, from the first month (if it starts on the 1st) or
    the one after, to Sep 2026."""
    t = ledger.transactions
    net = t.groupby(t["ts"].dt.to_period("M"))["amount"].sum()
    first = net.index.min() + (0 if t["ts"].min().day == 1 else 1)
    return net.reindex(pd.period_range(first, "2026-09", freq="M"), fill_value=0)


def expected_median(ledger: Ledger) -> float:
    return round(float(full_months(ledger).tail(12).median()), 2)


# Listing


def test_goals_are_listed_with_status_and_what_they_need(page: Tools, ledger: Ledger) -> None:
    data = page.call("list_goals", {}).data
    listed = {g["name"]: g for g in data["goals"]}

    assert [g["name"] for g in data["goals"]] == ["College fund", "Vacation fund", "New laptop"]
    assert {n: g["status"] for n, g in listed.items()} == {
        "Vacation fund": "active",
        "New laptop": "reached",
        "College fund": "ended",
    }
    vacation = listed["Vacation fund"]
    assert (vacation["months_left"], vacation["needed_per_month"]) == (3, 287.0)
    assert vacation["saved"] == 2140.0
    assert vacation["origin"] == "existing"
    assert vacation["undo_revision_id"] is None
    assert listed["New laptop"]["needed_per_month"] is None
    assert data["median_monthly_savings_12m"] == pytest.approx(expected_median(ledger), abs=0.01)
    assert data["months_of_history"] == len(full_months(ledger))
    assert data["forecast"] == "not_available"


def test_ended_and_archived_goals_are_listed_on_request(page: Tools) -> None:
    assert "College fund" not in goals_by_name(page, include_ended=False)
    page.call("archive_goal", {"goal_id": "g_x_1"})
    assert "Vacation fund" not in goals_by_name(page)
    archived = goals_by_name(page, include_archived=True)["Vacation fund"]
    assert archived["status"] == "archived"
    assert archived["undo_revision_id"]


# Checking


def test_a_check_states_the_facts_for_a_new_goal(page: Tools, ledger: Ledger) -> None:
    result = page.call("check_goal", TRIP)
    data = result.data

    assert data["valid"]
    assert data["problems"] == []
    assert data["target_date"] == "2027-06-30"  # due at the end of its month
    assert (data["months_left"], data["needed_per_month"]) == (9, 334.0)
    assert data["other_goals_per_month"] == 287.0  # the vacation fund; reached goals need nothing
    assert data["all_goals_per_month"] == 621.0
    assert data["median_monthly_savings_12m"] == pytest.approx(expected_median(ledger), abs=0.01)
    assert result.source.title == "Goal check · Trip"
    assert "Trip" not in goals_by_name(page)  # nothing written


def test_a_check_reports_problems_by_field(page: Tools) -> None:
    data = page.call(
        "check_goal", {"name": "vacation FUND", "target_amount": 20, "target_date": "2026-09-01"}
    ).data

    assert not data["valid"]
    assert {(p["field"], p["code"]) for p in data["problems"]} == {
        ("name", "name_in_use"),
        ("target_amount", "amount_range"),
        ("target_date", "date_too_soon"),
    }
    assert "needed_per_month" not in data
    assert data["median_monthly_savings_12m"] is not None  # the form shows it anyway


def test_an_empty_form_is_problems_not_an_error(page: Tools) -> None:
    codes = {p["code"] for p in page.call("check_goal", {}).data["problems"]}
    assert codes == {"name_missing", "amount_invalid", "date_invalid"}


def test_checking_an_edit_leaves_the_goal_out_of_other_goals(page: Tools) -> None:
    data = page.call("check_goal", {"goal_id": "g_x_1", "target_amount": 3300}).data

    assert data["valid"]
    assert (data["name"], data["needed_per_month"]) == ("Vacation fund", 387.0)
    assert data["other_goals_per_month"] == 0.0
    with pytest.raises(ToolError, match="no goal 'g_y_9'"):
        page.call("check_goal", {"goal_id": "g_y_9"})


def test_amounts_must_be_numbers(page: Tools) -> None:
    with pytest.raises(ToolError, match="target_amount must be a number"):
        page.call("check_goal", {**TRIP, "target_amount": "3000"})
    with pytest.raises(ToolError, match="must be a number"):
        page.call("check_goal", {**TRIP, "saved": True})


# Writing from the Goals page


def test_the_page_creates_a_goal_on_submit(page: Tools, store: GoalStore, ledger: Ledger) -> None:
    data = page.call("create_goal", {**TRIP, "saved": 300}).data

    assert data["status"] == "applied"
    goal = data["goal"]
    assert goal["goal_id"].startswith("gu_")
    assert (goal["name"], goal["target_date"], goal["saved"]) == ("Trip", "2027-06-30", 300.0)
    assert goal["needed_per_month"] == 300.0
    assert goal["undo_revision_id"] == data["revision_id"]
    assert [r.source for r in store.revisions("s1", ledger.user_id)] == ["edit"]


def test_a_refused_write_carries_its_problems(page: Tools) -> None:
    with pytest.raises(GoalProblemsError) as error:
        page.call("create_goal", {**TRIP, "name": "New Laptop", "saved": 5000})
    assert {p.code for p in error.value.problems} == {"name_in_use", "saved_range"}
    assert "You already have a goal called New Laptop." in str(error.value)


def test_updates_change_only_the_fields_given(page: Tools) -> None:
    data = page.call("update_goal", {"goal_id": "g_x_1", "saved": 3000}).data

    assert data["goal"]["status"] == "reached"
    assert data["goal"]["target_amount"] == 3000.0
    with pytest.raises(ToolError, match="at least one"):
        page.call("update_goal", {"goal_id": "g_x_1"})
    with pytest.raises(GoalProblemsError) as error:
        page.call("update_goal", {"goal_id": "g_x_3", "name": "Uni"})
    assert [p.code for p in error.value.problems] == ["goal_not_editable"]


def test_archive_and_undo_through_the_tools(page: Tools) -> None:
    archived = page.call("archive_goal", {"goal_id": "g_x_1"}).data
    assert archived["goal"]["status"] == "archived"

    undone = page.call("undo_goal_change", {"revision_id": archived["revision_id"]}).data
    assert undone["removed"] is False
    assert undone["goal"]["status"] == "active"

    created = page.call("create_goal", TRIP).data
    removed = page.call("undo_goal_change", {"revision_id": created["revision_id"]}).data
    assert (removed["removed"], removed["goal"]) == (True, None)
    with pytest.raises(ToolError, match="already undone"):
        page.call("undo_goal_change", {"revision_id": created["revision_id"]})


def test_an_undo_that_breaks_a_rule_is_refused_with_its_problems(page: Tools) -> None:
    archived = page.call("archive_goal", {"goal_id": "g_x_1"}).data
    page.call("create_goal", {**TRIP, "name": "Vacation fund"})
    with pytest.raises(GoalProblemsError) as error:
        page.call("undo_goal_change", {"revision_id": archived["revision_id"]})
    assert [p.code for p in error.value.problems] == ["name_in_use"]


# Writing from the coach and assistants


@pytest.mark.parametrize("source", ["coach", "assistant"])
def test_assistants_preview_every_change_until_confirmed(
    ledger: Ledger, store: GoalStore, source: str
) -> None:
    tools = tools_as(ledger, store, source)

    preview = tools.call("create_goal", TRIP).data
    assert preview["status"] == "needs_confirmation"
    assert (preview["target_date"], preview["needed_per_month"]) == ("2027-06-30", 334.0)
    assert "confirm: true" in preview["message"]
    for name, args in (
        ("update_goal", {"goal_id": "g_x_1", "target_amount": 3500}),
        ("archive_goal", {"goal_id": "g_x_1"}),
    ):
        assert tools.call(name, args).data["status"] == "needs_confirmation"
    assert store.revisions("s1", ledger.user_id) == []  # nothing written

    applied = tools.call("create_goal", {**TRIP, "confirm": True}).data
    assert applied["status"] == "applied"
    assert [r.source for r in store.revisions("s1", ledger.user_id)] == [source]


def test_an_undo_from_the_coach_applies_directly(coach: Tools, store: GoalStore) -> None:
    created = coach.call("create_goal", {**TRIP, "confirm": True}).data
    undone = coach.call("undo_goal_change", {"revision_id": created["revision_id"]}).data
    assert undone["status"] == "applied"


def test_an_invalid_goal_is_refused_before_any_preview(coach: Tools) -> None:
    with pytest.raises(GoalProblemsError, match="Goals start at"):
        coach.call("create_goal", {**TRIP, "target_amount": 10})


# Read-only and isolation


def test_without_goal_access_the_tools_are_read_only(ledger: Ledger) -> None:
    tools = Tools(ledger)

    assert set(goals_by_name(tools)) == {"Vacation fund", "New laptop", "College fund"}
    assert tools.call("check_goal", TRIP).data["valid"]
    for name, args in (
        ("create_goal", TRIP),
        ("update_goal", {"goal_id": "g_x_1", "name": "x"}),
        ("archive_goal", {"goal_id": "g_x_1"}),
        ("undo_goal_change", {"revision_id": "r"}),
    ):
        with pytest.raises(ToolError, match="isn't available for this sign-in"):
            tools.call(name, args)


def test_another_session_sees_only_the_generated_goals(
    ledger: Ledger, store: GoalStore, page: Tools
) -> None:
    created = page.call("create_goal", TRIP).data
    theirs = tools_as(ledger, store, subject="s2")

    assert "Trip" not in goals_by_name(theirs, include_archived=True)
    goal_id = created["goal"]["goal_id"]
    with pytest.raises(ToolError, match="no goal"):
        theirs.call("update_goal", {"goal_id": goal_id, "name": "Mine"})
    with pytest.raises(ToolError, match="no goal"):
        theirs.call("forecast_goal", {"goal_id": goal_id})
    with pytest.raises(ToolError, match="no change"):
        theirs.call("undo_goal_change", {"revision_id": created["revision_id"]})
    with pytest.raises(ToolError, match="takes no subject"):
        theirs.call("list_goals", {"subject": "s1"})


def test_forecasts_are_not_available_yet_for_any_goal(page: Tools) -> None:
    created = page.call("create_goal", TRIP).data
    result = page.call("forecast_goal", {"goal_id": created["goal"]["goal_id"]})
    assert result.data["status"] == "not_available"
    assert result.source.detail == "Trip"


def test_the_first_month_counts_when_it_starts_on_the_first() -> None:
    """Three full months of history give a median; a first month that starts late doesn't count
    (review on #43)."""

    def ledger_from(first_day: str) -> pd.DataFrame:
        days = [first_day, "2026-08-10", "2026-09-10"]
        return pd.DataFrame({"ts": pd.to_datetime(days), "amount": [100.0, 200.0, 300.0]})

    assert median_monthly_savings(ledger_from("2026-07-01"), TODAY) == (200_00, 3)
    assert median_monthly_savings(ledger_from("2026-07-02"), TODAY) == (None, 2)
