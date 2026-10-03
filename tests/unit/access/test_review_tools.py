"""FR-5/FR-6 tools: review items, corrections, undo, and totals that follow them (#15 §3, Tools)."""

from pathlib import Path

import pytest

from smart_financial_coach.access.feedback import FeedbackStore
from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.tools import Feedback, ToolError, Tools


@pytest.fixture
def store(sources: DataSources, tmp_path: Path) -> FeedbackStore:
    return FeedbackStore(tmp_path / "fb.sqlite", sources.categories())


def tools_for(
    sources: DataSources, user: str, store: FeedbackStore, subject: str = "s1", source: str = "edit"
) -> Tools:
    return Tools(Ledger.load(sources, user), Feedback(store, subject, source))


def whole_range(tools: Tools) -> dict[str, str]:
    t = tools.ledger.transactions
    return {"start_date": t["day"].min().isoformat(), "end_date": t["day"].max().isoformat()}


def by_category(tools: Tools) -> dict[str, float]:
    data = tools.call("get_spending_summary", whole_range(tools)).data
    return {c["category"]: c["amount"] for c in data["by_category"]}


def new_category(item: dict[str, object]) -> str:
    return "Travel" if item["suggested_category"] != "Travel" else "Entertainment"


def test_correcting_an_item_moves_its_spending_and_closes_it(
    sources: DataSources, two_users: tuple[str, str], store: FeedbackStore
) -> None:
    tools = tools_for(sources, two_users[0], store)
    listed = tools.call("list_review_items", {"limit": 5}).data
    item = listed["items"][0]
    before = by_category(tools)
    unreviewed = tools.call("get_spending_summary", whole_range(tools)).data["unreviewed_spend"]
    target = new_category(item)

    applied = tools.call(
        "resolve_review_item", {"item_id": item["item_id"], "action": "correct", "category": target}
    ).data

    assert applied["status"] == "applied"  # the dashboard applies at once; no preview
    assert applied["correction"]["source"] == "review"
    moved = sum(m["amount"] for m in applied["spend_moved"])
    after = by_category(tools)
    assert after.get(target, 0) == pytest.approx(before.get(target, 0) + moved, abs=0.01)
    assert sum(after.values()) == pytest.approx(sum(before.values()), abs=0.01)
    again = tools.call("list_review_items", {"limit": 25}).data
    assert again["open_items"] == listed["open_items"] - 1
    assert item["item_id"] not in {i["item_id"] for i in again["items"]}
    summary = tools.call("get_spending_summary", whole_range(tools)).data
    assert summary["unreviewed_spend"] == pytest.approx(
        unreviewed - item["unreviewed_spend"], abs=0.01
    )


def test_undo_restores_the_totals_and_the_item(
    sources: DataSources, two_users: tuple[str, str], store: FeedbackStore
) -> None:
    tools = tools_for(sources, two_users[0], store)
    item = tools.call("list_review_items", {}).data["items"][0]
    before = by_category(tools)
    applied = tools.call(
        "resolve_review_item",
        {"item_id": item["item_id"], "action": "correct", "category": new_category(item)},
    ).data

    undone = tools.call(
        "undo_correction", {"correction_id": applied["correction"]["correction_id"]}
    ).data

    assert undone["transactions_changed"] == applied["transactions_changed"]
    assert by_category(tools) == pytest.approx(before)
    assert item["item_id"] in {
        i["item_id"] for i in tools.call("list_review_items", {"limit": 25}).data["items"]
    }
    (logged,) = tools.call("list_corrections", {}).data["corrections"]
    assert not logged["undoable"]
    with pytest.raises(ToolError, match="already undone"):
        tools.call("undo_correction", {"correction_id": logged["correction_id"]})


def test_the_coach_previews_a_bulk_change_until_the_user_agrees(
    sources: DataSources, two_users: tuple[str, str], store: FeedbackStore
) -> None:
    coach = tools_for(sources, two_users[0], store, source="coach")
    items = coach.call("list_review_items", {"limit": 25}).data["items"]
    item = next(i for i in items if i["transaction_count"] > 1)
    args = {"item_id": item["item_id"], "action": "correct", "category": new_category(item)}

    preview = coach.call("resolve_review_item", args).data

    assert preview["status"] == "needs_confirmation"
    assert preview["transactions"] == item["transaction_count"]
    assert coach.call("list_corrections", {}).data["corrections"] == []  # nothing applied
    applied = coach.call("resolve_review_item", {**args, "confirm": True}).data
    assert applied["status"] == "applied"
    assert applied["correction"]["source"] == "coach"


def test_correcting_one_transaction_leaves_the_rest_of_its_merchant(
    sources: DataSources, two_users: tuple[str, str], store: FeedbackStore
) -> None:
    tools = tools_for(sources, two_users[0], store)
    t = tools.ledger.transactions
    counts = t["merchant_key"].value_counts()
    row = t[t["merchant_key"] == counts.index[0]].iloc[0]
    target = "Travel" if row["category"] != "Travel" else "Entertainment"

    applied = tools.call(
        "correct_category",
        {"transaction_id": row["transaction_id"], "category": target, "scope": "transaction"},
    ).data

    assert applied["transactions_changed"] == 1
    seen = tools.ledger.transactions.set_index("transaction_id")
    assert seen.loc[row["transaction_id"], "category"] == target
    assert seen.loc[row["transaction_id"], "category_source"] == "you"
    rest = seen[
        (seen["merchant_key"] == row["merchant_key"]) & (seen.index != row["transaction_id"])
    ]
    assert set(rest["category_source"]) == {"model"}
    listed = tools.call(
        "get_transactions", {**whole_range(tools), "search": row["merchant"], "limit": 50}
    ).data["transactions"]
    mine = next(x for x in listed if x["transaction_id"] == row["transaction_id"])
    assert mine["category_source"] == "you"
    assert mine["needs_review"] is False


def test_bad_calls_are_tool_errors(
    sources: DataSources, two_users: tuple[str, str], store: FeedbackStore
) -> None:
    tools = tools_for(sources, two_users[0], store)
    item = tools.call("list_review_items", {}).data["items"][0]
    with pytest.raises(ToolError, match="unknown category"):
        tools.call(
            "resolve_review_item",
            {"item_id": item["item_id"], "action": "correct", "category": "Crypto"},
        )
    with pytest.raises(ToolError, match="needs the new category"):
        tools.call("resolve_review_item", {"item_id": item["item_id"], "action": "correct"})
    with pytest.raises(ToolError, match="no open review item"):
        tools.call("resolve_review_item", {"item_id": "nope", "action": "confirm"})
    with pytest.raises(ToolError, match="must be a boolean"):
        tools.call(
            "resolve_review_item", {"item_id": item["item_id"], "action": "confirm", "confirm": 1}
        )
    with pytest.raises(ToolError, match="takes no user_id"):
        tools.call("list_review_items", {"user_id": two_users[1]})
    read_only = Tools(Ledger.load(sources, two_users[0]))
    with pytest.raises(ToolError, match="isn't available"):
        read_only.call("list_corrections", {})


def test_one_users_feedback_never_reaches_another_users_ledger(
    sources: DataSources, two_users: tuple[str, str], store: FeedbackStore
) -> None:
    mine = tools_for(sources, two_users[0], store, subject="s1")
    item = mine.call("list_review_items", {}).data["items"][0]
    mine.call("resolve_review_item", {"item_id": item["item_id"], "action": "confirm"})

    theirs = tools_for(sources, two_users[1], store, subject="s1")  # even with the same subject

    assert theirs.call("list_corrections", {}).data["corrections"] == []
    assert set(theirs.ledger.transactions["category_source"]) == {"model"}


def test_every_transaction_from_a_row_with_its_own_category_moves_it_too(
    sources: DataSources, two_users: tuple[str, str], store: FeedbackStore
) -> None:
    tools = tools_for(sources, two_users[0], store)
    t = tools.ledger.transactions
    key = t.loc[t["category"] != "Income", "merchant_key"].value_counts().index[0]
    row = t[t["merchant_key"] == key].iloc[0]
    held = t.loc[t["merchant_key"] == key, "category"].mode().iloc[0]
    tools.call(
        "correct_category",
        {"transaction_id": row["transaction_id"], "category": "Travel", "scope": "transaction"},
    )

    applied = tools.call(
        "correct_category", {"transaction_id": row["transaction_id"], "category": "Entertainment"}
    ).data

    assert applied["correction"]["from_category"] == held  # what the merchant held, not Travel
    seen = tools.ledger.transactions
    at = seen[(seen["merchant_key"] == key) & (seen["model_category"] != "Income")]
    assert set(at["category"]) == {"Entertainment"}
