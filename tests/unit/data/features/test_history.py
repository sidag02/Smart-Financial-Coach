import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.data.features.history import COLUMNS, MAD_TO_SD, history_features


def txns(rows: list[tuple[str, str, str, float, str]]) -> pd.DataFrame:
    """(transaction_id, user_id, ts, amount, merchant_raw) rows as model-visible transactions."""
    df = pd.DataFrame(rows, columns=["transaction_id", "user_id", "ts", "amount", "merchant_raw"])
    return df.assign(currency="USD", channel="card")


ROWS = [
    ("t1", "u1", "2025-01-01 09:00", -10.0, "SQ *BLUE BOTTLE #1"),
    ("t2", "u1", "2025-01-02 09:00", -12.0, "SQ *BLUE BOTTLE #2"),
    ("t3", "u1", "2025-01-03 09:00", 2000.0, "ACME PAYROLL"),  # income: never history
    ("t4", "u1", "2025-01-04 09:00", -11.0, "SQ *BLUE BOTTLE #3"),
    ("t5", "u1", "2025-01-05 09:00", -90.0, "SQ *BLUE BOTTLE #4"),
    ("t6", "u1", "2025-01-05 09:30", -90.0, "SQ *BLUE BOTTLE #4"),
    ("t7", "u2", "2025-01-01 10:00", -500.0, "DELTA AIR"),
    ("t8", "u1", "2025-01-06 09:00", -40.0, "TARGET 00012 AUSTIN TX"),
]


def by_id(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("transaction_id")


def test_one_row_per_outflow_in_input_order() -> None:
    f = history_features(txns(ROWS))

    assert tuple(f.columns) == COLUMNS
    assert f["transaction_id"].tolist() == ["t1", "t2", "t4", "t5", "t6", "t7", "t8"]


def test_overall_history_counts_only_earlier_outflows() -> None:
    f = by_id(history_features(txns(ROWS)))

    assert f.loc["t1", "prior_charges"] == 0
    assert pd.isna(f.loc["t1", "rank_in_history"])
    assert f.loc["t4", "prior_charges"] == 2  # income t3 doesn't count
    assert f.loc["t4", "rank_in_history"] == 0.5  # larger than t1 only
    assert f.loc["t4", "largest_since"] == "2025-01-02 09:00"  # t2 is the latest at least as large
    assert pd.isna(f.loc["t5", "largest_since"])  # the largest so far
    assert f.loc["t6", "largest_since"] == "2025-01-05 09:00"  # equal counts as at least as large
    assert f.loc["t8", "rank_in_history"] == pytest.approx(3 / 5)


def test_history_at_the_merchant_key() -> None:
    f = by_id(history_features(txns(ROWS)))
    x = np.log([10.0, 12.0, 11.0])

    assert f.loc["t5", "merchant_key"] == "blue bottle"
    assert f.loc["t5", "key_prior"] == 3
    assert f.loc["t5", "key_median"] == pytest.approx(np.median(x))
    assert f.loc["t5", "key_spread"] == pytest.approx(
        MAD_TO_SD * np.median(np.abs(x - np.median(x)))
    )
    assert pd.isna(f.loc["t1", "key_median"])
    assert pd.isna(f.loc["t2", "key_spread"])
    assert f.loc["t8", "key_prior"] == 0  # first visit


def test_window_keeps_only_recent_charges_at_the_key() -> None:
    rows = [
        (f"t{i}", "u1", f"2025-01-{i + 1:02d} 09:00", -float(a), "CAFE")
        for i, a in enumerate([100, 100, 100, 10, 10])
    ]
    f = by_id(history_features(txns(rows), window=2))

    assert f.loc["t4", "key_median"] == pytest.approx(np.log(10 * 100) / 2)  # 100 and 10 only


def test_exact_repeats() -> None:
    f = by_id(history_features(txns(ROWS)))

    assert f.loc["t6", "repeat_of"] == "t5"
    assert f.loc["t6", "minutes_since_repeat"] == 30
    assert pd.isna(f.loc["t5", "repeat_of"])  # different amount and text from earlier charges


def test_point_in_time_later_rows_change_nothing() -> None:
    rng = np.random.default_rng(0)
    n = 300
    rows = [
        (
            f"t{i:03d}",
            f"u{i % 3}",
            str(pd.Timestamp("2025-01-01") + pd.Timedelta(hours=7 * i)),
            -float(np.round(rng.lognormal(3, 1), 2)),
            f"STORE {rng.integers(5)}",
        )
        for i in range(n)
    ]
    full = by_id(history_features(txns(rows)))
    early = by_id(history_features(txns(rows[: n // 2])))

    pd.testing.assert_frame_equal(early, full.loc[early.index])


def test_input_order_does_not_matter() -> None:
    df = txns(ROWS)
    shuffled = df.sample(frac=1, random_state=3)

    pd.testing.assert_frame_equal(
        by_id(history_features(df)).sort_index(), by_id(history_features(shuffled)).sort_index()
    )


def test_window_must_hold_a_spread() -> None:
    with pytest.raises(ValueError, match="window"):
        history_features(txns(ROWS), window=1)


def test_a_same_minute_repeat_is_found_whichever_id_sorts_first() -> None:
    rows = [
        ("tb", "u1", "2025-01-05 09:00", -9.0, "LYFT"),
        ("ta", "u1", "2025-01-05 09:00", -9.0, "LYFT"),  # sorts first within the minute
    ]
    f = by_id(history_features(txns(rows)))

    assert f.loc["ta", "repeat_of"] == "tb"
    assert f.loc["tb", "repeat_of"] == "ta"
    assert f.loc["ta", "minutes_since_repeat"] == 0
