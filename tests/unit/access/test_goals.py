"""FR-10 §1-3: the goal record, validation, the store's replay and undo, isolation."""

import threading
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from smart_financial_coach.access.goals import (
    ACTIVE_MAX,
    Goal,
    GoalDraft,
    GoalError,
    GoalStore,
    check_draft,
    generated_goals,
    month_end,
    months_left,
    to_cents,
)
from smart_financial_coach.access.ledger import DataSources, Ledger

TODAY = date(2026, 9, 30)  # the small dataset's last day, as in the demo


def generated(*rows: tuple[str, float, str, float]) -> pd.DataFrame:
    """Generated goals for a ledger: (name, target, target_date, saved), saved as of today."""
    return pd.DataFrame(
        [
            {
                "goal_id": f"g_x_{i + 1}",
                "user_id": "u",
                "name": name,
                "target_amount": target,
                "created_date": "2026-01-15",
                "target_date": target_date,
                "as_of_date": TODAY.isoformat() if target_date > TODAY.isoformat() else target_date,
                "current_balance": saved,
            }
            for i, (name, target, target_date, saved) in enumerate(rows)
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


VACATION = ("Vacation fund", 3000.0, "2026-12-31", 2140.0)
LAPTOP_REACHED = ("New laptop", 2100.0, "2027-05-31", 2100.0)
COLLEGE_ENDED = ("College fund", 5900.0, "2025-01-31", 5606.21)


@pytest.fixture
def real(sources: DataSources, two_users: tuple[str, str]) -> Ledger:
    return Ledger.load(sources, two_users[0])


@pytest.fixture
def ledger(real: Ledger) -> Ledger:
    """A real user's ledger with known generated goals and today fixed."""
    return replace(real, goals=generated(VACATION, LAPTOP_REACHED, COLLEGE_ENDED), as_of=TODAY)


@pytest.fixture
def store(tmp_path: Path) -> GoalStore:
    return GoalStore(tmp_path / "goals.sqlite")


def draft(
    name: object = "Trip", amount: object = 3000, when: object = "2027-06-30", saved: object = 0
) -> GoalDraft:
    return GoalDraft(name, amount, when, saved)


def codes(problems: object) -> list[str]:
    return [p.code for p in problems]  # type: ignore[attr-defined]


def find(goals: list[Goal], name: str) -> Goal:
    return next(g for g in goals if g.name == name)


# Months and money


@pytest.mark.parametrize(
    ("today", "target", "expected"),
    [
        (date(2026, 9, 30), date(2027, 6, 30), 9),  # the design's example
        (date(2026, 9, 15), date(2027, 6, 30), 10),  # Sep 30 counts from mid-September
        (date(2026, 9, 30), date(2026, 10, 31), 1),
        (date(2026, 9, 30), date(2026, 9, 30), 0),
        (date(2026, 9, 30), date(2025, 1, 31), 0),
    ],
)
def test_months_left_counts_month_ends_after_today(
    today: date, target: date, expected: int
) -> None:
    assert months_left(today, target) == expected


def test_month_end_handles_february() -> None:
    assert month_end(date(2028, 2, 10)) == date(2028, 2, 29)
    assert month_end(date(2027, 2, 1)) == date(2027, 2, 28)


@pytest.mark.parametrize(
    ("amount", "cents"),
    [(19.99, 1999), (0.29, 29), (50, 5000), ("50.10", 5010), (Decimal("3227.8"), 322780)],
)
def test_to_cents_is_exact(amount: object, cents: int) -> None:
    assert to_cents(amount) == cents


@pytest.mark.parametrize("amount", [50.001, "0.005", True, None, "abc", float("nan"), "inf"])
def test_to_cents_refuses_what_isnt_whole_cents(amount: object) -> None:
    with pytest.raises(ValueError, match=r"amount|cent"):
        to_cents(amount)


# Validation


def test_a_valid_draft_becomes_a_month_end_goal() -> None:
    checked = check_draft(draft(when="2027-06-01"), TODAY, [])
    assert checked.valid
    goal = checked.goal
    assert goal is not None
    assert goal.target_date == date(2027, 6, 30)
    assert goal.goal_id.startswith("gu_")
    assert goal.origin == "yours"
    assert (goal.created_date, goal.saved_as_of) == (TODAY, TODAY)
    assert goal.months_left(TODAY) == 9
    assert goal.needed_per_month_cents(TODAY) == 334_00  # $3,000 / 9 = $333.33, rounded up


def test_june_1_and_june_30_are_the_same_goal() -> None:
    a = check_draft(draft(when="2027-06-01"), TODAY, []).goal
    b = check_draft(draft(when=date(2027, 6, 30)), TODAY, []).goal
    assert a is not None
    assert b is not None
    assert a.target_date == b.target_date
    assert a.needed_per_month_cents(TODAY) == b.needed_per_month_cents(TODAY)


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"name": "  "}, "name_missing"),
        ({"name": None}, "name_missing"),
        ({"name": "x" * 41}, "name_too_long"),
        ({"amount": 49.99}, "amount_range"),
        ({"amount": 1_000_000.01}, "amount_range"),
        ({"amount": 50.001}, "amount_cents"),
        ({"amount": "lots"}, "amount_invalid"),
        ({"when": "2026-09-15"}, "date_too_soon"),
        ({"when": "2036-10-01"}, "date_too_far"),
        ({"when": "next June"}, "date_invalid"),
        ({"saved": -1}, "saved_range"),
        ({"saved": 3000}, "saved_range"),  # creating at the whole amount
        ({"saved": 0.001}, "amount_cents"),
    ],
)
def test_each_rule_refuses_past_its_boundary(kwargs: dict[str, object], code: str) -> None:
    checked = check_draft(draft(**kwargs), TODAY, [])
    assert code in codes(checked.problems)
    assert checked.goal is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "x" * 40},
        {"name": "  Trip  "},
        {"amount": 50},
        {"amount": 1_000_000},
        {"amount": 19.99 + 50},
        {"when": "2026-10-01"},  # the first allowed month
        {"when": "2036-09-30"},  # 120 months ahead
        {"saved": 2999.99},
        {"saved": 0.29},
    ],
)
def test_each_rule_allows_its_boundary(kwargs: dict[str, object]) -> None:
    checked = check_draft(draft(**kwargs), TODAY, [])
    assert checked.valid, checked.problems


def test_date_too_soon_names_the_first_month_allowed() -> None:
    (problem,) = check_draft(draft(when="2026-09-30"), TODAY, []).problems
    assert problem.message == "Pick October 2026 or later."
    assert problem.field == "target_date"


def test_names_are_unique_among_running_goals_ignoring_case() -> None:
    others = generated_goals(generated(VACATION, LAPTOP_REACHED, COLLEGE_ENDED))
    assert codes(check_draft(draft("vacation FUND"), TODAY, others).problems) == ["name_in_use"]
    assert codes(check_draft(draft("New Laptop"), TODAY, others).problems) == ["name_in_use"]
    assert check_draft(draft("College fund"), TODAY, others).valid  # ended goals free the name
    archived = [replace(g, archived=True) for g in others]
    assert check_draft(draft("Vacation fund"), TODAY, archived).valid


def test_ten_active_goals_is_the_limit_and_reached_ones_dont_count() -> None:
    active = [(f"Goal {i}", 1000.0, "2027-06-30", 0.0) for i in range(ACTIVE_MAX)]
    ten = generated_goals(generated(*active))
    assert codes(check_draft(draft(), TODAY, ten).problems) == ["too_many_goals"]
    nine_and_reached = generated_goals(generated(*active[:-1], LAPTOP_REACHED))
    assert check_draft(draft(), TODAY, nine_and_reached).valid


def test_status_is_ended_first_then_reached_then_active() -> None:
    goals = generated_goals(
        generated(VACATION, LAPTOP_REACHED, COLLEGE_ENDED, ("Old", 100.0, "2025-01-31", 500.0))
    )
    assert [g.status(TODAY) for g in goals] == ["active", "reached", "ended", "ended"]
    assert [g.needed_per_month_cents(TODAY) for g in goals] == [287_00, 0, 0, 0]


def test_generated_balances_convert_to_cents_exactly(real: Ledger) -> None:
    goals = generated_goals(real.goals)
    assert len(goals) == len(real.goals) > 0
    for g, row in zip(goals, real.goals.to_dict("records"), strict=True):
        assert g.saved_cents == round(row["current_balance"] * 100)
        assert g.target_cents == round(row["target_amount"] * 100)
        assert g.origin == "existing"
        assert g.undo_revision_id is None


# The store


def test_a_created_goal_is_listed_with_its_undo(store: GoalStore, ledger: Ledger) -> None:
    rev = store.create(ledger, "s1", draft(when="2027-06-01"), source="edit")
    trip = find(store.goals(ledger, "s1"), "Trip")
    assert (trip.goal_id, trip.origin, trip.undo_revision_id) == (
        rev.goal_id,
        "yours",
        rev.revision_id,
    )
    assert trip.target_date == date(2027, 6, 30)
    assert [r.source for r in store.revisions("s1", ledger.user_id)] == ["edit"]


def test_editing_a_generated_goal_keeps_it_existing(store: GoalStore, ledger: Ledger) -> None:
    vacation = find(store.goals(ledger, "s1"), "Vacation fund")
    store.update(ledger, "s1", vacation.goal_id, {"target_amount": 3500}, source="edit")
    edited = find(store.goals(ledger, "s1"), "Vacation fund")
    assert (edited.origin, edited.target_cents, edited.saved_cents) == (
        "existing",
        3500_00,
        2140_00,
    )
    assert edited.saved_as_of == vacation.saved_as_of  # balance unchanged, so is its day


def test_a_new_saved_amount_is_recorded_as_of_today(store: GoalStore, ledger: Ledger) -> None:
    trip = store.create(ledger, "s1", draft(), source="edit")
    later = replace(ledger, as_of=date(2026, 10, 31))
    store.update(later, "s1", trip.goal_id, {"saved": 400}, source="edit")
    goal = store.goal(later, "s1", trip.goal_id)
    assert (goal.saved_cents, goal.saved_as_of, goal.created_date) == (400_00, later.as_of, TODAY)


def test_saving_the_whole_amount_reaches_a_goal(store: GoalStore, ledger: Ledger) -> None:
    vacation = find(store.goals(ledger, "s1"), "Vacation fund")
    store.update(ledger, "s1", vacation.goal_id, {"saved": 3100}, source="edit")
    assert store.goal(ledger, "s1", vacation.goal_id).status(TODAY) == "reached"
    with pytest.raises(GoalError) as error:
        store.update(ledger, "s1", vacation.goal_id, {"saved": -5}, source="edit")
    assert codes(error.value.problems) == ["saved_range"]


def test_ended_goals_can_be_archived_but_not_edited(store: GoalStore, ledger: Ledger) -> None:
    college = find(store.goals(ledger, "s1"), "College fund")
    with pytest.raises(GoalError) as error:
        store.update(ledger, "s1", college.goal_id, {"name": "Uni"}, source="edit")
    assert codes(error.value.problems) == ["goal_not_editable"]
    store.archive(ledger, "s1", college.goal_id, source="edit")
    assert "College fund" not in [g.name for g in store.goals(ledger, "s1")]


def test_archive_hides_and_undo_restores(store: GoalStore, ledger: Ledger) -> None:
    vacation = find(store.goals(ledger, "s1"), "Vacation fund")
    rev = store.archive(ledger, "s1", vacation.goal_id, source="edit")
    assert vacation.goal_id not in [g.goal_id for g in store.goals(ledger, "s1")]
    archived = find(store.goals(ledger, "s1", include_archived=True), "Vacation fund")
    assert archived.status(TODAY) == "archived"
    assert archived.undo_revision_id == rev.revision_id
    with pytest.raises(GoalError, match="no goal"):
        store.update(ledger, "s1", vacation.goal_id, {"name": "x"}, source="edit")

    restored = store.undo(ledger, "s1", rev.revision_id)
    assert restored == vacation  # the generated goal again, with nothing left to undo


def test_undoing_a_creation_removes_the_goal(store: GoalStore, ledger: Ledger) -> None:
    rev = store.create(ledger, "s1", draft(), source="coach")
    assert store.undo(ledger, "s1", rev.revision_id) is None
    assert "Trip" not in [g.name for g in store.goals(ledger, "s1", include_archived=True)]
    with pytest.raises(GoalError, match="already undone"):
        store.undo(ledger, "s1", rev.revision_id)


def test_undo_goes_back_one_change_at_a_time(store: GoalStore, ledger: Ledger) -> None:
    created = store.create(ledger, "s1", draft(), source="edit")
    renamed = store.update(ledger, "s1", created.goal_id, {"name": "Lisbon"}, source="edit")
    with pytest.raises(GoalError, match="latest change"):
        store.undo(ledger, "s1", created.revision_id)
    back = store.undo(ledger, "s1", renamed.revision_id)
    assert back is not None
    assert (back.name, back.undo_revision_id) == ("Trip", created.revision_id)
    assert [r.undone for r in store.revisions("s1", ledger.user_id)] == [True, False]


def test_undo_cant_bring_back_a_duplicate_name(store: GoalStore, ledger: Ledger) -> None:
    vacation = find(store.goals(ledger, "s1"), "Vacation fund")
    archived = store.archive(ledger, "s1", vacation.goal_id, source="edit")
    store.create(ledger, "s1", draft("Vacation fund"), source="edit")
    with pytest.raises(GoalError) as error:
        store.undo(ledger, "s1", archived.revision_id)
    assert codes(error.value.problems) == ["name_in_use"]
    assert not any(r.undone for r in store.revisions("s1", ledger.user_id))  # nothing changed
    assert [g.name for g in store.goals(ledger, "s1")].count("Vacation fund") == 1


def test_undo_cant_bring_back_an_eleventh_active_goal(store: GoalStore, ledger: Ledger) -> None:
    ten = replace(
        ledger,
        goals=generated(*[(f"Goal {i}", 1000.0, "2027-06-30", 0.0) for i in range(ACTIVE_MAX)]),
    )
    archived = store.archive(ten, "s1", "g_x_1", source="edit")
    store.create(ten, "s1", draft(), source="edit")
    with pytest.raises(GoalError) as error:
        store.undo(ten, "s1", archived.revision_id)
    assert codes(error.value.problems) == ["too_many_goals"]


def test_a_reached_goal_cant_become_an_eleventh_active_one(
    store: GoalStore, ledger: Ledger
) -> None:
    rows = [(f"Goal {i}", 1000.0, "2027-06-30", 0.0) for i in range(ACTIVE_MAX)]
    full = replace(ledger, goals=generated(*rows, LAPTOP_REACHED))
    laptop = find(store.goals(full, "s1"), "New laptop")
    for change in ({"target_amount": 2500}, {"saved": 100}):
        with pytest.raises(GoalError) as error:
            store.update(full, "s1", laptop.goal_id, change, source="edit")
        assert codes(error.value.problems) == ["too_many_goals"]
    store.update(full, "s1", laptop.goal_id, {"name": "Laptop"}, source="edit")  # still reached


def test_writes_are_validated_inside_the_lock(store: GoalStore, ledger: Ledger) -> None:
    errors: list[GoalError] = []

    def create() -> None:
        try:
            store.create(ledger, "s1", draft(), source="edit")
        except GoalError as error:
            errors.append(error)

    threads = [threading.Thread(target=create) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [g.name for g in store.goals(ledger, "s1")].count("Trip") == 1
    assert len(errors) == 7
    assert {codes(e.problems)[0] for e in errors} == {"name_in_use"}


def test_unknown_sources_and_fields_are_refused(store: GoalStore, ledger: Ledger) -> None:
    with pytest.raises(GoalError, match="source"):
        store.create(ledger, "s1", draft(), source="model")
    vacation = find(store.goals(ledger, "s1"), "Vacation fund")
    with pytest.raises(GoalError, match="can't change"):
        store.update(ledger, "s1", vacation.goal_id, {"user_id": "u2"}, source="edit")


# Isolation


def test_another_subject_sees_only_the_generated_goals(store: GoalStore, ledger: Ledger) -> None:
    vacation = find(store.goals(ledger, "s1"), "Vacation fund")
    trip = store.create(ledger, "s1", draft(), source="edit")
    store.archive(ledger, "s1", vacation.goal_id, source="edit")

    theirs = store.goals(ledger, "s2", include_archived=True)
    assert sorted(g.name for g in theirs) == ["College fund", "New laptop", "Vacation fund"]
    assert all(g.undo_revision_id is None for g in theirs)
    for attempt in (
        lambda: store.update(ledger, "s2", trip.goal_id, {"name": "Mine"}, source="edit"),
        lambda: store.archive(ledger, "s2", trip.goal_id, source="edit"),
        lambda: store.goal(ledger, "s2", trip.goal_id),
    ):
        with pytest.raises(GoalError, match="no goal"):
            attempt()
    with pytest.raises(GoalError, match="no change"):
        store.undo(ledger, "s2", trip.revision_id)


def test_one_subjects_changes_stay_on_the_user_they_were_made_for(
    store: GoalStore, ledger: Ledger, sources: DataSources, two_users: tuple[str, str]
) -> None:
    trip = store.create(ledger, "s1", draft(), source="edit")
    other = replace(Ledger.load(sources, two_users[1]), as_of=TODAY)
    assert "Trip" not in [g.name for g in store.goals(other, "s1", include_archived=True)]
    with pytest.raises(GoalError, match="no goal"):
        store.goal(other, "s1", trip.goal_id)
    with pytest.raises(GoalError, match="no change"):
        store.undo(other, "s1", trip.revision_id)
