import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.data.features.history import MAD_TO_SD
from smart_financial_coach.data.features.merchant_profiles import COLUMNS, profile_features


def txns(rows: list[tuple[str, str, str, float, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["transaction_id", "user_id", "ts", "amount", "merchant_raw"])
    return df.assign(currency="USD", channel="card")


def charge(
    i: int, user: str, month: str, amount: float, merchant: str = "CAFE"
) -> tuple[str, str, str, float, str]:
    return (f"t{i}", user, f"2025-{month}-10 12:00", -amount, merchant)


def by_id(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("transaction_id")


def test_a_profile_needs_min_users_other_users() -> None:
    rows = [charge(i, f"u{i}", "01", 10.0 + i) for i in range(3)]
    scored = txns([*rows, charge(9, "u9", "02", 50.0), charge(8, "u0", "02", 50.0)])
    p = by_id(profile_features(scored, scored))

    assert tuple(p.reset_index().columns) == COLUMNS
    assert p.loc["t9", "profile_users"] == 3
    assert p.loc["t9", "profile_typical"] == pytest.approx(np.log(11.0))
    assert p.loc["t8", "profile_users"] == 2  # u0's own January charge is left out
    assert pd.isna(p.loc["t8", "profile_typical"])


def test_the_scored_users_own_charges_never_count() -> None:
    rows = [charge(i, f"u{i}", "01", 10.0) for i in range(3)]
    rows += [charge(10 + i, "u9", "01", 1000.0) for i in range(5)]
    scored = txns([*rows, charge(99, "u9", "02", 12.0)])
    p = by_id(profile_features(scored, scored))

    assert p.loc["t99", "profile_typical"] == pytest.approx(np.log(10.0))


def test_profiles_are_as_of_the_first_of_the_month() -> None:
    rows = [charge(i, f"u{i}", "02", 10.0) for i in range(3)]  # same month as the scored charge
    scored = txns([*rows, charge(99, "u9", "02", 12.0)])
    p = by_id(profile_features(scored, scored))

    assert p.loc["t99", "profile_users"] == 0
    assert pd.isna(p.loc["t99", "profile_typical"])


def test_spread_needs_min_users_with_two_charges() -> None:
    rows = [charge(i, f"u{i}", "01", 10.0) for i in range(3)]  # one charge each: no spread
    scored = txns([*rows, charge(99, "u9", "02", 12.0)])
    assert pd.isna(by_id(profile_features(scored, scored)).loc["t99", "profile_spread"])

    rows += [charge(10 + i, f"u{i}", "01", 20.0) for i in range(3)]
    scored = txns([*rows, charge(99, "u9", "02", 12.0)])
    spread = by_id(profile_features(scored, scored)).loc["t99", "profile_spread"]
    assert spread == pytest.approx(MAD_TO_SD * np.log(2.0) / 2)


def test_the_pool_decides_whose_charges_count() -> None:
    train = txns([charge(i, f"u{i}", "01", 10.0) for i in range(3)])
    test = txns([charge(10 + i, f"v{i}", "01", 500.0) for i in range(3)])
    scored = txns([charge(99, "u9", "02", 12.0)])

    validation = by_id(profile_features(scored, train))
    serving = by_id(profile_features(scored, pd.concat([train, test])))
    assert validation.loc["t99", "profile_typical"] == pytest.approx(np.log(10.0))
    assert serving.loc["t99", "profile_users"] == 6


def test_income_and_refunds_are_not_scored_or_pooled() -> None:
    rows = [charge(i, f"u{i}", "01", 10.0) for i in range(3)]
    rows.append(("r1", "u5", "2025-01-11 12:00", 900.0, "CAFE"))  # a refund at the same key
    scored = txns([*rows, charge(99, "u9", "02", 12.0)])
    p = profile_features(scored, scored)

    assert "r1" not in set(p["transaction_id"])
    assert by_id(p).loc["t99", "profile_users"] == 3


def _brute_force(scored: pd.DataFrame, pool: pd.DataFrame, min_users: int) -> pd.DataFrame:
    """The definition, one charge at a time."""
    from smart_financial_coach.data.features.merchant_text import normalize_merchant

    pool = pool[pool["amount"] < 0].assign(
        key=lambda d: d["merchant_raw"].map(normalize_merchant), x=lambda d: np.log(-d["amount"])
    )
    out = []
    for r in scored[scored["amount"] < 0].itertuples():
        key, month = normalize_merchant(str(r.merchant_raw)), str(r.ts)[:7]
        others = pool[(pool["key"] == key) & (pool["ts"].str[:7] < month)]
        others = others[others["user_id"] != r.user_id]
        per = others.groupby("user_id")["x"]
        medians = per.median()
        deviation = others.assign(d=(others["x"] - per.transform("median")).abs())
        mads = deviation.groupby("user_id")["d"].median()[per.size() >= 2]
        out.append(
            {
                "transaction_id": r.transaction_id,
                "profile_users": len(medians),
                "profile_typical": medians.median() if len(medians) >= min_users else np.nan,
                "profile_spread": MAD_TO_SD * mads.median() if len(mads) >= min_users else np.nan,
            }
        )
    return pd.DataFrame(out)


@pytest.mark.parametrize("min_users", [1, 3])
def test_matches_the_definition_on_random_data(min_users: int) -> None:
    rng = np.random.default_rng(min_users)
    n = 400
    rows = [
        (
            f"t{i:03d}",
            f"u{rng.integers(8)}",
            f"2025-{rng.integers(1, 7):02d}-{rng.integers(1, 28):02d} 12:00",
            -float(np.round(rng.lognormal(3, 0.7), 2)),
            f"SHOP {rng.integers(4)}",
        )
        for i in range(n)
    ]
    scored = txns(rows)
    pool = scored[scored["user_id"] != "u0"]  # a pool that leaves one scored user out

    got = profile_features(scored, pool, min_users=min_users)
    want = _brute_force(scored, pool, min_users)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)


def test_min_users_must_be_positive() -> None:
    with pytest.raises(ValueError, match="min_users"):
        profile_features(txns([]), txns([]), min_users=0)
