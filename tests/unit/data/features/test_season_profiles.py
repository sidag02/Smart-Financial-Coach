import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.data.features.monthly import monthly_aggregates, period_history
from smart_financial_coach.data.features.season_profiles import (
    COLUMNS,
    season_features,
    season_profiles,
    season_table,
    user_terms,
)


def ledger(users: dict[str, list[int]], category: str = "Shopping") -> pd.DataFrame:
    """Each user's monthly purchase counts from Jan 2025, one $10 purchase per count."""
    rows = []
    for user, counts in users.items():
        for i, n in enumerate(counts):
            y, m = 2025 + i // 12, i % 12 + 1
            rows += [(user, f"{y}-{m:02d}-1{k % 10} 12:00", -10.0, category) for k in range(n)]
    return pd.DataFrame(rows, columns=["user_id", "ts", "amount", "category"])


def periods(users: dict[str, list[int]]) -> pd.DataFrame:
    return period_history(monthly_aggregates(ledger(users)))


# 14 months: a busy December (month 12) in the first year, then January and February 2026
BUSY_DECEMBER = [4] * 11 + [12] + [4, 4]


def at(feats: pd.DataFrame, period_id: str) -> pd.Series:
    rows = feats.set_index("period_id")
    return rows.loc[rows.index == period_id].iloc[0]


def test_a_profile_needs_min_users_other_users() -> None:
    p = periods({f"u{i}": BUSY_DECEMBER + [4] * 10 + [4] for i in range(4)})
    feats = season_profiles(p, p, min_users=3)
    dec26 = at(feats, "u0|Shopping|2026-12-01")

    assert tuple(feats.columns) == COLUMNS
    assert dec26["profile_users"] == 3  # u0 is left out
    assert dec26["profile_season"] == pytest.approx(np.log(3.0))  # 12 against 4, clipped at 3x
    assert at(season_profiles(p, p, min_users=4), "u0|Shopping|2026-12-01")["profile_season"] == 0


def test_profiles_are_as_of_the_month() -> None:
    p = periods({f"u{i}": BUSY_DECEMBER for i in range(4)})
    feats = season_profiles(p, p, min_users=1)
    # December 2025 is the first December observed: it can't shape its own month's profile
    assert at(feats, "u0|Shopping|2025-12-01")["profile_users"] == 0
    assert at(feats, "u0|Shopping|2025-12-01")["profile_season"] == 0


def test_leaving_a_user_out_equals_rebuilding_without_them() -> None:
    users = {
        "a": BUSY_DECEMBER + [4] * 10 + [9] + [4],
        "b": [3] * 11 + [6] + [3] * 11 + [3] + [3],
        "c": [5] * 11 + [20] + [5] * 11 + [5] + [5],
        "d": [2] * 11 + [2] + [2] * 11 + [8] + [2],
    }
    p = periods(users)
    left_out = season_profiles(p, p, min_users=1).set_index("period_id")
    for user in users:
        mine = p[p["user_id"] == user]
        rebuilt = season_features(mine, season_table(p[p["user_id"] != user]), min_users=1)
        rebuilt = rebuilt.set_index("period_id")
        np.testing.assert_allclose(
            left_out.loc[rebuilt.index, "profile_season"], rebuilt["profile_season"], atol=1e-12
        )
        assert (left_out.loc[rebuilt.index, "profile_users"] == rebuilt["profile_users"]).all()


def test_a_users_term_is_their_mean_so_far() -> None:
    p = periods({"a": [4] * 11 + [8] + [4] * 11 + [16] + [4]})
    t = user_terms(p)
    dec = t[t["moy"] == 11]
    first, second = np.log(2.0), np.log(3.0)  # 16 against 52 / 12 is clipped at 3x
    assert dec["term"].tolist() == pytest.approx([first, (first + second) / 2])
    assert dec["valid_from"].tolist() == [2025 * 12 + 12, 2026 * 12 + 12]


def test_an_own_term_missing_from_the_table_is_an_error() -> None:
    p = periods({f"u{i}": BUSY_DECEMBER + [4] * 11 for i in range(3)})
    others = season_table(p[p["user_id"] != "u0"])
    with pytest.raises(ValueError, match="wrong pool"):
        season_features(p, season_table(p.iloc[:0]), own=user_terms(p), min_users=1)
    # Without `own`, a user outside the pool is scored against everyone in it
    mine = p[p["user_id"] == "u0"]
    assert (
        at(season_features(mine, others, min_users=1), "u0|Shopping|2026-12-01")["profile_users"]
        == 2
    )
