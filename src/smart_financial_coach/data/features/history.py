"""Point-in-time history features for unusual-transaction scoring (FR-7 §1, §2).

Every feature of a charge uses only the same user's *earlier* outflows, in (ts, transaction_id)
order, so appending later rows never changes it. Income and refunds (amount >= 0) are never
scored and never count as history.

    feats = history_features(transactions)  # one row per outflow, in input order

Columns:
    transaction_id, user_id, merchant_key
    prior_charges          the user's earlier outflows
    rank_in_history        share of those that were smaller (NaN without any)
    largest_since          ts of the most recent earlier outflow at least as large (None: largest)
    key_prior              the user's earlier outflows at this merchant key
    key_median             median log amount of the last `window` of them (NaN without any)
    key_spread             1.4826 x their median absolute deviation (NaN with fewer than 2)
    repeat_of              the latest earlier outflow with the same raw text and amount, or one
                           in the same minute (whose order within the minute is unknowable)
    minutes_since_repeat   minutes since that charge (NaN without one)
"""

import bisect

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.data.features.merchant_text import normalize_merchant

COLUMNS = (
    "transaction_id",
    "user_id",
    "merchant_key",
    "prior_charges",
    "rank_in_history",
    "largest_since",
    "key_prior",
    "key_median",
    "key_spread",
    "repeat_of",
    "minutes_since_repeat",
)
MAD_TO_SD = 1.4826  # median absolute deviation to standard deviation, for a normal
DEFAULT_WINDOW = 50  # earlier charges at a merchant that describe its usual amount


def outflows(transactions: pd.DataFrame) -> pd.DataFrame:
    """The rows that are scored: amount < 0, in input order."""
    return transactions[transactions["amount"] < 0]


def _mad(window: npt.NDArray[np.float64]) -> float:
    values = window[~np.isnan(window)]
    if len(values) < 2:
        return float("nan")
    return float(np.median(np.abs(values - np.median(values))))


def _overall(
    amounts: npt.NDArray[np.float64], ts: npt.NDArray[np.object_]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.object_]]:
    """Rank among earlier amounts, and the ts of the latest earlier amount >= this one.

    One user's charges in time order. The stack keeps earlier charges in decreasing amount, so its
    top after popping smaller ones is the nearest earlier charge at least as large.
    """
    rank = np.full(len(amounts), np.nan)
    since = np.full(len(amounts), None, dtype=object)
    seen: list[float] = []
    stack: list[int] = []
    for i, value in enumerate(amounts):
        if seen:
            rank[i] = bisect.bisect_left(seen, value) / len(seen)
        bisect.insort(seen, value)
        while stack and amounts[stack[-1]] < value:
            stack.pop()
        if stack:
            since[i] = ts[stack[-1]]
        stack.append(i)
    return rank, since


def history_features(transactions: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """Point-in-time features for every outflow in `transactions`, in input order."""
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    out = outflows(transactions)
    f = pd.DataFrame(
        {
            "transaction_id": out["transaction_id"].to_numpy(),
            "user_id": out["user_id"].to_numpy(),
            "ts": out["ts"].astype(str).to_numpy(),
            "raw": out["merchant_raw"].to_numpy(),
            "amount": -out["amount"].to_numpy(dtype=np.float64),
        }
    )
    f["merchant_key"] = f["raw"].map(normalize_merchant)
    f["x"] = np.log(f["amount"])
    order = f.sort_values(["user_id", "ts", "transaction_id"], kind="mergesort").index
    s = f.loc[order]

    s["prior_charges"] = s.groupby("user_id").cumcount()
    rank = np.full(len(s), np.nan)
    since = np.full(len(s), None, dtype=object)
    amounts, times = s["amount"].to_numpy(), s["ts"].to_numpy()
    for idx in s.groupby("user_id", sort=False).indices.values():
        rank[idx], since[idx] = _overall(amounts[idx], times[idx])
    s["rank_in_history"], s["largest_since"] = rank, since

    by_key = s.groupby(["user_id", "merchant_key"], sort=False)
    s["key_prior"] = by_key.cumcount()
    earlier = by_key["x"].shift()  # each charge sees only the ones before it
    keys = [s["user_id"], s["merchant_key"]]
    rolling = earlier.groupby(keys, sort=False).rolling(window, min_periods=1)
    s["key_median"] = rolling.median().droplevel([0, 1])
    s["key_spread"] = MAD_TO_SD * rolling.apply(_mad, raw=True).droplevel([0, 1])

    same = s.groupby(["user_id", "raw", "amount"], sort=False)
    s["repeat_of"] = same["transaction_id"].shift()
    previous = pd.to_datetime(same["ts"].shift())
    s["minutes_since_repeat"] = (pd.to_datetime(s["ts"]) - previous).dt.total_seconds() / 60
    # Within one minute the order is unknowable (ts has minute resolution, IDs are arbitrary), so
    # the first of a same-minute pair repeats the next one too: both were known by then
    later = same["transaction_id"].shift(-1)
    same_minute = s["repeat_of"].isna() & later.notna() & (same["ts"].shift(-1) == s["ts"])
    s.loc[same_minute, "repeat_of"] = later[same_minute]
    s.loc[same_minute, "minutes_since_repeat"] = 0.0

    return s.loc[f.index, list(COLUMNS)].reset_index(drop=True)
