"""FR-5/FR-6 §3: corrections, overrides, effective categories, undo, isolation; review items."""

from pathlib import Path

import pandas as pd
import pytest

from smart_financial_coach.access.feedback import (
    Correction,
    FeedbackError,
    FeedbackStore,
    Overrides,
    effective_categories,
)
from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.review_items import open_review_items


@pytest.fixture
def store(sources: DataSources, tmp_path: Path) -> FeedbackStore:
    return FeedbackStore(tmp_path / "feedback.sqlite", sources.categories())


@pytest.fixture
def ledger(sources: DataSources, two_users: tuple[str, str]) -> Ledger:
    return Ledger.load(sources, two_users[0])


def flagged_merchant(ledger: Ledger) -> pd.DataFrame:
    """The rows at the most-flagged spending merchant string, flagged ones first."""
    t = ledger.transactions
    spending = t[(t["amount"] < 0) & t["needs_review"]]
    key = spending["merchant_key"].value_counts().index[0]
    rows: pd.DataFrame = t[t["merchant_key"] == key]
    return rows.sort_values("needs_review", ascending=False, kind="stable")


def other(category: str) -> str:
    return "Shopping" if category != "Shopping" else "Groceries"


def correct(
    store: FeedbackStore, subject: str, ledger: Ledger, row: pd.Series, **kw: str | None
) -> Correction:
    args = {
        "action": "correct",
        "scope": "merchant",
        "merchant_key": row["merchant_key"],
        "from_category": row["category"],
        "to_category": other(row["category"]),
        "source": "review",
        "model_version": "stub",
    } | kw
    return store.record(subject, ledger.user_id, **args)


def test_a_merchant_correction_changes_every_transaction_there_at_once(
    store: FeedbackStore, ledger: Ledger
) -> None:
    rows = flagged_merchant(ledger)
    first = rows.iloc[0]

    correct(store, "s1", ledger, first)
    seen = ledger.seen_by(store, "s1").transactions.set_index("transaction_id")

    mine = seen.loc[rows["transaction_id"]]
    assert set(mine["category"]) == {other(first["category"])}
    assert set(mine["category_source"]) == {"you"}
    assert not mine["needs_review"].any()  # the override settles review
    assert mine["model_category"].tolist() == rows["category"].tolist()
    rest = seen.drop(index=rows["transaction_id"].tolist())
    assert set(rest["category_source"]) == {"model"}
    assert (
        rest["category"].tolist()
        == ledger.transactions.set_index("transaction_id")
        .drop(index=rows["transaction_id"].tolist())["category"]
        .tolist()
    )


def test_a_transaction_override_wins_over_a_merchant_override(
    store: FeedbackStore, ledger: Ledger
) -> None:
    rows = flagged_merchant(ledger)
    one = rows.iloc[0]
    correct(store, "s1", ledger, one, to_category="Travel")
    correct(
        store,
        "s1",
        ledger,
        one,
        scope="transaction",
        transaction_id=one["transaction_id"],
        to_category="Entertainment",
    )

    seen = ledger.seen_by(store, "s1").transactions.set_index("transaction_id")

    assert seen.loc[one["transaction_id"], "category"] == "Entertainment"
    assert set(seen.loc[rows["transaction_id"].iloc[1:], "category"]) == {"Travel"}


def test_undo_restores_the_previous_state_and_keeps_the_log(
    store: FeedbackStore, ledger: Ledger
) -> None:
    row = flagged_merchant(ledger).iloc[0]
    first = correct(store, "s1", ledger, row, to_category="Travel")
    second = correct(store, "s1", ledger, row, to_category="Entertainment")

    store.undo("s1", second.correction_id)
    after_one = ledger.seen_by(store, "s1").transactions.set_index("transaction_id")
    store.undo("s1", first.correction_id)
    after_both = ledger.seen_by(store, "s1").transactions.set_index("transaction_id")

    assert after_one.loc[row["transaction_id"], "category"] == "Travel"
    assert after_both.loc[row["transaction_id"], "category"] == row["category"]
    assert after_both.loc[row["transaction_id"], "needs_review"]  # flagged again
    log = store.corrections("s1", ledger.user_id)
    assert [c.correction_id for c in log] == [second.correction_id, first.correction_id]
    assert all(c.undone for c in log)
    with pytest.raises(FeedbackError, match="already undone"):
        store.undo("s1", first.correction_id)


def test_two_sessions_on_one_account_never_see_each_others_feedback(
    store: FeedbackStore, ledger: Ledger
) -> None:
    """The demo's visitors share accounts: feedback is scoped by session, not by user."""
    row = flagged_merchant(ledger).iloc[0]
    mine = correct(store, "visitor-a", ledger, row)

    theirs = ledger.seen_by(store, "visitor-b").transactions

    assert theirs["category"].tolist() == ledger.transactions["category"].tolist()
    assert set(theirs["category_source"]) == {"model"}
    assert store.corrections("visitor-b", ledger.user_id) == []
    with pytest.raises(FeedbackError, match="no correction"):
        store.undo("visitor-b", mine.correction_id)
    assert not store.corrections("visitor-a", ledger.user_id)[0].undone


def test_bad_feedback_is_rejected(store: FeedbackStore, ledger: Ledger) -> None:
    row = flagged_merchant(ledger).iloc[0]
    with pytest.raises(FeedbackError, match="unknown category"):
        correct(store, "s1", ledger, row, to_category="Crypto")
    with pytest.raises(FeedbackError, match="keeps the category"):
        correct(store, "s1", ledger, row, action="confirm")
    with pytest.raises(FeedbackError, match="already"):
        correct(store, "s1", ledger, row, to_category=row["category"])
    with pytest.raises(FeedbackError, match="exactly one transaction"):
        correct(store, "s1", ledger, row, scope="transaction")
    with pytest.raises(FeedbackError, match="subject"):
        correct(store, "", ledger, row)
    assert store.corrections("s1", ledger.user_id) == []


def test_a_confirmation_pins_the_category_and_closes_the_item(
    store: FeedbackStore, ledger: Ledger
) -> None:
    row = flagged_merchant(ledger).iloc[0]
    before = open_review_items(ledger.transactions, ledger.user_id)
    assert row["merchant_key"] in set(before["merchant_key"])

    correct(store, "s1", ledger, row, action="confirm", to_category=row["category"])
    seen = ledger.seen_by(store, "s1").transactions
    after = open_review_items(seen, ledger.user_id)

    assert row["merchant_key"] not in set(after["merchant_key"])
    assert len(after) == len(before) - 1
    # Merchant scope pins every transaction there, including ones the model put elsewhere (an
    # ambiguous merchant); "just this one" is the scope for exceptions
    pinned = seen[seen["merchant_key"] == row["merchant_key"]]
    assert set(pinned["category_source"]) == {"you"}
    assert set(pinned["category"]) == {row["category"]}


def test_review_items_are_one_per_merchant_string_ranked_by_unreviewed_spend(
    ledger: Ledger,
) -> None:
    items = open_review_items(ledger.transactions, ledger.user_id)
    t = ledger.transactions

    assert items["merchant_key"].is_unique
    assert set(items["merchant_key"]) == set(t.loc[t["needs_review"], "merchant_key"])
    assert items["unreviewed_spend"].is_monotonic_decreasing
    assert (items["flagged_count"] <= items["transaction_count"]).all()
    assert set(items["reason"]) <= {"new_merchant", "low_confidence"}
    assert items["item_id"].is_unique
    top = items.iloc[0]
    at = t[t["merchant_key"] == top["merchant_key"]]
    assert top["transaction_count"] == len(at)
    assert top["unreviewed_spend"] == pytest.approx(
        (-at.loc[at["needs_review"], "amount"]).clip(lower=0).sum()
    )


def event(seq: int, scope: str, to: str, transaction_id: str | None = None) -> Correction:
    return Correction(
        f"c{seq}", seq, "s1", "u1", "correct", scope, "shop", transaction_id, "Shopping", to,
        "edit", "v", "2026-10-03T00:00:00+00:00", None,
    )  # fmt: skip


SHOP = pd.DataFrame(
    {
        "transaction_id": ["t1", "t2", "t3"],
        "merchant_key": ["shop", "shop", "shop"],
        "category": ["Shopping", "Shopping", "Income"],  # t3: a refund the model called Income
        "needs_review": [True, True, False],
    }
)


def test_a_merchant_override_never_moves_income() -> None:
    """Review covers spending rows (owner, on #32): settling one leaves the refund as Income."""
    seen = effective_categories(SHOP, Overrides.replay([event(1, "merchant", "Travel")]))

    assert seen["category"].tolist() == ["Travel", "Travel", "Income"]
    assert seen["category_source"].tolist() == ["you", "you", "model"]
    one = Overrides.replay([event(1, "transaction", "Dining", "t3")])
    assert effective_categories(SHOP, one)["category"].tolist()[2] == "Dining"  # one at a time


def test_every_transaction_means_every_one_including_a_row_set_on_its_own() -> None:
    row_then_all = [event(1, "transaction", "Travel", "t1"), event(2, "merchant", "Dining")]
    all_then_row = [event(1, "merchant", "Dining"), event(2, "transaction", "Travel", "t1")]

    later_merchant = effective_categories(SHOP, Overrides.replay(row_then_all))
    later_row = effective_categories(SHOP, Overrides.replay(all_then_row))

    assert later_merchant["category"].tolist()[:2] == ["Dining", "Dining"]
    assert later_row["category"].tolist()[:2] == ["Travel", "Dining"]
