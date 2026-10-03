"""Merchant profiles: what many users usually pay at a merchant (FR-7 §2).

A profile is a feature table, built from model-visible rows, never fitted into a model:

- **As of the month.** A charge is scored against profiles built from pool charges before the
  first of its month, so it never sees later prices.
- **Leave the user out.** The scored user's own charges never count towards their profile.
- **Distinct users.** A merchant key gets a typical price only when at least `min_users` *other*
  users have charges there, and a spread only when at least `min_users` other users have two or
  more. A string that occurs for one user (`ZELLE TO <name>`) never gets a profile.

    profiles = profile_features(scored=transactions, pool=transactions)  # serving, test
    profiles = profile_features(scored=train, pool=train)  # validation: train users only

Columns, one row per outflow of `scored`, in input order:
    transaction_id
    profile_users     other users with charges at the key before the month
    profile_typical   median over those users of each user's median log amount (NaN: no profile)
    profile_spread    1.4826 x the median over users of each user's median absolute deviation of
                      log amount (NaN: fewer than `min_users` users with two or more charges)
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.data.features.history import MAD_TO_SD, outflows
from smart_financial_coach.data.features.merchant_text import normalize_merchant

COLUMNS = ("transaction_id", "profile_users", "profile_typical", "profile_spread")
DEFAULT_MIN_USERS = 3


def _charges(transactions: pd.DataFrame) -> pd.DataFrame:
    out = outflows(transactions)
    return pd.DataFrame(
        {
            "transaction_id": out["transaction_id"].to_numpy(),
            "user_id": out["user_id"].to_numpy(),
            "month": out["ts"].astype(str).str[:7].to_numpy(),
            "key": out["merchant_raw"].map(normalize_merchant).to_numpy(),
            "x": np.log(-out["amount"].to_numpy(dtype=np.float64)),
        }
    )


def _per_user(pool: pd.DataFrame) -> pd.DataFrame:
    """Each (key, user)'s median log amount, and its median absolute deviation (2+ charges)."""
    g = pool.groupby(["key", "user_id"], sort=False)["x"]
    stats = g.agg(["median", "size"]).reset_index()
    deviation = (pool["x"] - g.transform("median")).abs()
    mad = deviation.groupby([pool["key"], pool["user_id"]], sort=False).median()
    stats["mad"] = mad.to_numpy()
    stats.loc[stats["size"] < 2, "mad"] = np.nan
    return stats


def _loo_median(
    stats: pd.DataFrame, value: str, pairs: pd.DataFrame, min_users: int
) -> npt.NDArray[np.float64]:
    """Median of `value` over each pair's key, leaving the pair's own user out.

    `pairs` has `key` and `user_id`. With the key's values sorted, dropping the user's own value
    at rank r shifts every later index by one, so the median is read straight from the array.
    """
    v = stats[["key", "user_id", value]].dropna().sort_values(["key", value], kind="mergesort")
    a: npt.NDArray[np.float64] = v[value].to_numpy(dtype=np.float64)
    first = v.groupby("key", sort=False).cumcount().rsub(np.arange(len(v)))  # row of key's 1st
    start = pd.Series(first.to_numpy(), index=v["key"].to_numpy()).groupby(level=0).first()
    size = v.groupby("key", sort=False).size()
    own = pd.Series(
        np.arange(len(v)) - first.to_numpy(), index=pd.MultiIndex.from_frame(v[["key", "user_id"]])
    )

    p_start = pairs["key"].map(start).to_numpy(dtype=np.float64)
    p_size = pairs["key"].map(size).fillna(0).to_numpy(dtype=np.int64)
    rank = own.reindex(pd.MultiIndex.from_frame(pairs[["key", "user_id"]])).to_numpy()
    has_own = ~np.isnan(rank)
    m = p_size - has_own  # values left after leaving the user out
    rank = np.where(has_own, rank, m).astype(np.int64)  # without an own value nothing shifts
    ok = m >= min_users
    result = np.full(len(pairs), np.nan)
    if not ok.any():
        return result
    base, r, mm = p_start[ok].astype(np.int64), rank[ok], m[ok]

    def at(i: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        values: npt.NDArray[np.float64] = a[base + i + (i >= r)]
        return values

    lo, hi = (mm - 1) // 2, mm // 2
    result[ok] = (at(lo) + at(hi)) / 2
    return result


def profile_features(
    scored: pd.DataFrame, pool: pd.DataFrame, min_users: int = DEFAULT_MIN_USERS
) -> pd.DataFrame:
    """Profile columns for every outflow of `scored`, from `pool` charges before its month."""
    if min_users < 1:
        raise ValueError(f"min_users must be at least 1, got {min_users}")
    s, p = _charges(scored), _charges(pool)
    out = pd.DataFrame(
        {
            "transaction_id": s["transaction_id"],
            "profile_users": 0,
            "profile_typical": np.nan,
            "profile_spread": np.nan,
        }
    )
    p = p[p["key"].isin(set(s["key"]))]
    for month, rows in s.groupby("month", sort=True):
        before = p[(p["month"] < month) & p["key"].isin(set(rows["key"]))]
        if before.empty:
            continue
        stats = _per_user(before)
        pairs = rows[["key", "user_id"]].drop_duplicates().reset_index(drop=True)
        users = stats.groupby("key")["user_id"].size()
        own = pd.MultiIndex.from_frame(stats[["key", "user_id"]])
        mine = pd.MultiIndex.from_frame(pairs[["key", "user_id"]]).isin(own)
        pairs["profile_users"] = pairs["key"].map(users).fillna(0).astype(int) - mine
        pairs["profile_typical"] = _loo_median(stats, "median", pairs, min_users)
        pairs["profile_spread"] = MAD_TO_SD * _loo_median(stats, "mad", pairs, min_users)
        joined = rows[["key", "user_id"]].merge(pairs, on=["key", "user_id"], how="left")
        out.loc[rows.index, list(COLUMNS[1:])] = joined[list(COLUMNS[1:])].to_numpy()
    out = out.astype({"profile_users": int, "profile_typical": float, "profile_spread": float})
    return out.reset_index(drop=True)
