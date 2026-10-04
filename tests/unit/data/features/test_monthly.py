import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.data.features.monthly import (
    AGGREGATE_COLUMNS,
    CLIP,
    HISTORY_COLUMNS,
    month_index,
    month_start,
    monthly_aggregates,
    period_history,
)


def txns(rows: list[tuple[str, str, float, str]]) -> pd.DataFrame:
    """(user, ts, amount, category) rows."""
    return pd.DataFrame(rows, columns=["user_id", "ts", "amount", "category"])


Row = tuple[str, str, float, str]


def monthly(user: str, category: str, counts: list[int], amount: float = 10.0) -> list[Row]:
    """`counts[i]` purchases of `amount` in month i, from Jan 2025."""
    rows = []
    for i, n in enumerate(counts):
        y, m = 2025 + i // 12, i % 12 + 1
        rows += [(user, f"{y}-{m:02d}-1{k % 10} 12:00", -amount, category) for k in range(n)]
    return rows


def by_id(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("period_id")


def test_months_round_trip() -> None:
    starts = pd.Series(["2024-12-01", "2025-01-01"])
    assert month_index(starts).tolist() == [2024 * 12 + 11, 2025 * 12]
    assert month_start(month_index(starts)).tolist() == starts.tolist()


def test_aggregates_net_refunds_count_outflows_and_skip_income() -> None:
    p = by_id(
        monthly_aggregates(
            txns(
                [
                    ("u1", "2025-01-03 12:00", -50.0, "Shopping"),
                    ("u1", "2025-01-09 12:00", 20.0, "Shopping"),  # a refund
                    ("u1", "2025-01-15 12:00", 3000.0, "Income"),
                    ("u1", "2025-03-02 12:00", -5.0, "Dining"),
                ]
            )
        )
    )

    assert tuple(p.reset_index().columns) == AGGREGATE_COLUMNS
    jan = p.loc["u1|Shopping|2025-01-01"]
    assert (jan["spend"], jan["count"], jan["income"]) == (30.0, 1, 3000.0)
    # A category runs from its first purchase to the user's last month, zero-filled; Income never
    # is a period
    assert "u1|Dining|2025-01-01" not in p.index
    assert p.loc["u1|Shopping|2025-03-01", "count"] == 0
    assert set(p["category"]) == {"Shopping", "Dining"}
    assert len(p) == 4


def test_only_complete_months_are_kept() -> None:
    rows = monthly("u1", "Dining", [1, 1, 1, 1])
    assert monthly_aggregates(txns(rows), as_of="2025-03-31")["period_start"].tolist() == [
        "2025-01-01",
        "2025-02-01",
        "2025-03-01",
    ]
    # Mid-March, March is still in progress
    assert monthly_aggregates(txns(rows), as_of="2025-03-30")["period_start"].iloc[-1] == (
        "2025-02-01"
    )


def test_a_category_starts_at_its_first_purchase() -> None:
    rows = [*monthly("u1", "Dining", [3] * 6), ("u1", "2025-04-05 12:00", -40.0, "Travel")]
    p = monthly_aggregates(txns(rows))
    travel = p[p["category"] == "Travel"]
    assert travel["period_start"].tolist() == ["2025-04-01", "2025-05-01", "2025-06-01"]


def test_history_uses_earlier_months_only() -> None:
    rows = monthly("u1", "Dining", [4, 6, 8, 10, 12])
    short = by_id(period_history(monthly_aggregates(txns(rows[:28]))))  # the first 4 months
    full = by_id(period_history(monthly_aggregates(txns(rows))))

    pd.testing.assert_frame_equal(
        full.loc[short.index, list(HISTORY_COLUMNS)], short[list(HISTORY_COLUMNS)]
    )
    apr = full.loc["u1|Dining|2025-04-01"]
    assert apr["usual_months"] == 3
    assert apr["usual_count"] == pytest.approx(6.0)
    assert apr["usual"] == pytest.approx(60.0)
    assert apr["count_var"] == pytest.approx(4.0)


def test_the_window_is_twelve_months() -> None:
    p = by_id(period_history(monthly_aggregates(txns(monthly("u1", "Dining", [100] + [2] * 13)))))
    m14 = p.loc["u1|Dining|2026-02-01"]
    assert m14["usual_months"] == 12
    assert m14["usual_count"] == pytest.approx(2.0)  # January 2025's 100 is out of the window


def test_season_observations_need_three_months_and_are_clipped() -> None:
    june = ("u1", "2025-06-05 12:00", -9.0, "Shopping")
    rows = [*monthly("u1", "Dining", [2, 2, 2, 2, 60]), june]
    p = by_id(period_history(monthly_aggregates(txns(rows))))
    assert pd.isna(p.loc["u1|Dining|2025-03-01", "log_ratio"])  # two earlier months
    assert p.loc["u1|Dining|2025-04-01", "log_ratio"] == pytest.approx(0.0)
    assert p.loc["u1|Dining|2025-05-01", "log_ratio"] == pytest.approx(CLIP)  # 30x usual
    assert p.loc["u1|Dining|2025-06-01", "log_ratio"] == pytest.approx(-CLIP)  # none bought


def test_own_season_is_the_same_month_a_year_earlier() -> None:
    counts = [4] * 6 + [8] + [4] * 11 + [4] * 6 + [12] + [4] * 6  # Julys of 2025-2027
    p = by_id(period_history(monthly_aggregates(txns(monthly("u1", "Dining", counts)))))
    jul25, jul26, jul27 = (p.loc[f"u1|Dining|{y}-07-01"] for y in (2025, 2026, 2027))

    assert jul25["own_years"] == 0
    assert pd.isna(jul25["own_season"])
    assert jul26["own_years"] == 1
    assert jul26["own_season"] == pytest.approx(jul25["log_ratio"])
    assert jul27["own_years"] == 1  # only the year before counts (FR-8 §2)
    assert jul27["own_season"] == pytest.approx(jul26["log_ratio"])


def test_income_ratio_is_the_previous_two_months_over_the_window() -> None:
    rows = monthly("u1", "Dining", [3] * 5)
    pay = [1000.0, 1000.0, 1000.0, 4000.0, 1000.0]
    rows += [("u1", f"2025-{i + 1:02d}-01 09:00", a, "Income") for i, a in enumerate(pay)]
    p = by_id(period_history(monthly_aggregates(txns(rows))))

    assert pd.isna(p.loc["u1|Dining|2025-01-01", "income_ratio"])
    # May: March and April average 2,500 against 1,750 over January to April
    assert p.loc["u1|Dining|2025-05-01", "income_ratio"] == pytest.approx(2500 / 1750)


def test_history_spread_uses_every_earlier_month() -> None:
    p = by_id(
        period_history(monthly_aggregates(txns(monthly("u1", "Dining", [1] * 13 + [3] + [1]))))
    )
    last = p.loc["u1|Dining|2026-03-01"]  # after 13 months of $10 and one of $30, all counted
    assert last["history_mean"] == pytest.approx(np.mean([10.0] * 13 + [30.0]))
    assert last["history_sd"] == pytest.approx(np.std([10.0] * 13 + [30.0], ddof=1))
    assert pd.isna(p.loc["u1|Dining|2025-02-01", "history_sd"])  # one earlier month
