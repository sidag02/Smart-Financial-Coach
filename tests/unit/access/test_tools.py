"""The ledger and tools read one user's data only, and their numbers add up."""

from datetime import date

import pytest

from smart_financial_coach.access.ledger import DataSources, Ledger, display_name
from smart_financial_coach.access.tools import DASH, TOOL_SPECS, ToolError, Tools, span_label


@pytest.fixture
def tools(sources: DataSources, two_users: tuple[str, str]) -> Tools:
    return Tools(Ledger.load(sources, two_users[0]))


def month(tools: Tools) -> dict[str, str]:
    end = tools.ledger.as_of
    return {"start_date": end.replace(day=1).isoformat(), "end_date": end.isoformat()}


def test_a_ledger_holds_only_its_users_rows(
    sources: DataSources, two_users: tuple[str, str]
) -> None:
    mine, theirs = (Ledger.load(sources, u) for u in two_users)

    assert len(mine.transactions) > 0
    assert set(mine.transactions["transaction_id"]).isdisjoint(
        theirs.transactions["transaction_id"]
    )
    assert mine.as_of == date(2026, 9, 30)
    with pytest.raises(LookupError):
        Ledger.load(sources, "u_nobody")


def test_tools_take_no_user_id(tools: Tools, two_users: tuple[str, str]) -> None:
    for spec in TOOL_SPECS:
        assert "user_id" not in spec["input_schema"]["properties"]
        assert spec["input_schema"]["additionalProperties"] is False
    with pytest.raises(ToolError, match="takes no user_id"):
        tools.call("get_transactions", {**month(tools), "user_id": two_users[1]})


def test_transactions_never_include_another_users_rows(
    sources: DataSources, two_users: tuple[str, str]
) -> None:
    mine, theirs = (Tools(Ledger.load(sources, u)) for u in two_users)
    span = {"start_date": "2023-01-01", "end_date": "2026-09-30", "limit": 50}
    ids = {t["transaction_id"] for t in mine.call("get_transactions", span).data["transactions"]}

    assert ids
    assert ids.isdisjoint(theirs.ledger.transactions["transaction_id"])


def test_spending_summary_adds_up(tools: Tools) -> None:
    data = tools.call("get_spending_summary", month(tools)).data
    rows = tools.ledger.between(tools.ledger.as_of.replace(day=1), tools.ledger.as_of)

    assert data["currency"] == "USD"
    assert data["transactions"] == len(rows)
    assert data["spending"] == pytest.approx(sum(c["amount"] for c in data["by_category"]))
    assert data["spending"] == pytest.approx(-rows[rows["category"] != "Income"]["amount"].sum())
    assert data["income"] == pytest.approx(rows[rows["category"] == "Income"]["amount"].sum())
    assert data["net"] == pytest.approx(data["income"] - data["spending"])
    assert [m["month"] for m in data["by_month"]] == ["2026-09"]


def test_spending_summary_by_month_covers_the_range(tools: Tools) -> None:
    data = tools.call(
        "get_spending_summary", {"start_date": "2026-07-01", "end_date": "2026-09-30"}
    ).data

    assert [m["month"] for m in data["by_month"]] == ["2026-07", "2026-08", "2026-09"]
    assert sum(m["spending"] for m in data["by_month"]) == pytest.approx(data["spending"])


def test_transactions_filter_sort_and_limit(tools: Tools) -> None:
    args = {**month(tools), "category": "Dining", "sort": "largest", "limit": 3}
    data = tools.call("get_transactions", args).data
    amounts = [abs(t["amount"]) for t in data["transactions"]]

    assert data["shown"] == len(amounts) <= 3
    assert data["matched"] >= data["shown"]
    assert amounts == sorted(amounts, reverse=True)
    assert {t["category"] for t in data["transactions"]} == {"Dining"}
    assert all(t["needs_review"] == (t["confidence"] < 0.6) for t in data["transactions"])


def test_bad_arguments_are_tool_errors(tools: Tools) -> None:
    with pytest.raises(ToolError, match="unknown tool"):
        tools.call("delete_everything", {})
    with pytest.raises(ToolError, match="needs"):
        tools.call("get_spending_summary", {"start_date": "2026-09-01"})
    with pytest.raises(ToolError, match="YYYY-MM-DD"):
        tools.call("get_spending_summary", {"start_date": "Sept", "end_date": "2026-09-30"})
    with pytest.raises(ToolError, match="before"):
        tools.call("get_spending_summary", {"start_date": "2026-09-30", "end_date": "2026-09-01"})
    with pytest.raises(ToolError, match="no data after"):
        tools.call("get_spending_summary", {"start_date": "2027-01-01", "end_date": "2027-01-31"})
    with pytest.raises(ToolError, match="unknown category"):
        tools.call("get_transactions", {**month(tools), "category": "Crypto"})


def test_end_dates_past_the_data_are_cut_at_as_of(tools: Tools) -> None:
    data = tools.call(
        "get_spending_summary", {"start_date": "2026-09-01", "end_date": "2026-12-31"}
    ).data

    assert data["end_date"] == "2026-09-30"


def test_unbuilt_services_say_not_available_and_give_no_numbers(tools: Tools) -> None:
    for name, args in (("detect_anomalies", month(tools)), ("forecast_goal", {"goal_name": "x"})):
        result = tools.call(name, args)
        assert result.data["status"] == "not_available"
        assert not any(isinstance(v, int | float) for v in result.data.values())
        assert "not available" in result.source.title


def test_goals_are_listed(tools: Tools) -> None:
    data = tools.call("list_goals", {}).data

    assert data["forecast"] == "not_available"
    assert len(data["goals"]) == len(tools.ledger.goals)


def test_display_names() -> None:
    assert display_name("POS DEBIT SQ *BLUE BOTTLE #4321") == "Blue Bottle"
    assert display_name("TARGET 00012345 AUSTIN TX") == "Target Austin TX"
    assert display_name("CVS/PHARMACY") == "Cvs/Pharmacy"
    assert display_name("PEET'S COFFEE") == "Peet's Coffee"


def test_span_labels() -> None:
    assert span_label(date(2026, 9, 1), date(2026, 9, 30)) == "Sep 2026"
    assert span_label(date(2026, 9, 24), date(2026, 9, 30)) == f"Sep 24{DASH}30, 2026"
    assert span_label(date(2026, 7, 1), date(2026, 9, 30)) == f"Jul 1 {DASH} Sep 30, 2026"
    assert span_label(date(2025, 10, 1), date(2026, 9, 30)) == f"Oct 1, 2025 {DASH} Sep 30, 2026"
