"""Monthly net savings and a goal's balance under a share of them (FR-11 and FR-12 design, §2-§3).

A goal's balance moves by `share x net` each month and never goes below zero: FR-1's stage-9
rule, kept for v1 because goals are notional (owner decision on #50). A bad month draws a goal
down in proportion to its share.

    net = monthly_net(transactions)                      # one value per calendar month
    final = run_balance(0.0, 0.42, net.to_numpy())[-1]   # the balance after the last month
    share = infer_share(net.to_numpy(), 0.0, 1250.0)      # the share that reaches 1,250
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

Floats = npt.NDArray[np.float64]

SHARE_CAP = 1.0  # a goal never takes more than all of a month's net savings
_BISECTION_STEPS = 60


def monthly_net(transactions: pd.DataFrame) -> pd.Series:
    """Net savings per calendar month: the sum of the month's amounts, so categories (and FR-6
    corrections) never change it. Indexed by `pd.Period` month, every month from the first
    transaction's to the last's, a month without transactions as 0."""
    if transactions.empty:
        return pd.Series(dtype=float, index=pd.PeriodIndex([], freq="M"))
    months = pd.to_datetime(transactions["ts"]).dt.to_period("M")
    net = transactions["amount"].astype(float).groupby(months.to_numpy()).sum()
    span = pd.period_range(months.min(), months.max(), freq="M")
    return net.reindex(span, fill_value=0.0)


def run_balance(start: float | Floats, share: float | Floats, net: Floats) -> Floats:
    """The balance after each month of `net` (last axis), from `start`.

    `net` may be one path (months,) or many (paths, months); `start` and `share` broadcast over
    the paths. Each month: balance = max(0, balance + share x net).
    """
    net = np.asarray(net, dtype=float)
    balance = np.broadcast_to(np.asarray(start, dtype=float), net.shape[:-1]).copy()
    share_ = np.asarray(share, dtype=float)
    out = np.empty_like(net)
    for m in range(net.shape[-1]):
        balance = np.maximum(0.0, balance + share_ * net[..., m])
        out[..., m] = balance
    return out


def final_balance(start: float, share: float, net: Floats) -> float:
    return float(run_balance(start, share, net)[-1]) if len(net) else float(start)


def infer_share(net: Floats, start: float, end: float, cap: float = SHARE_CAP) -> float:
    """The share at which `start`, run over `net`, ends at `end`. From $0 at a goal's creation
    for a generated goal's track record, or from the first saved entry for a goal the user keeps
    up (§3).

    Bisection is sound because the final balance is the largest of straight lines in the share
    (the floor resets the sum after each emptying month), so it's convex, and it starts at
    `start < end`. A convex function that starts below a level crosses it at most once.

    0 when the balance never grew (`end <= start`) or there are no months; `cap` when even that
    share falls short.
    """
    net = np.asarray(net, dtype=float)
    if len(net) == 0 or end <= start:
        return 0.0
    if final_balance(start, cap, net) < end:
        return cap
    lo, hi = 0.0, cap
    for _ in range(_BISECTION_STEPS):
        mid = (lo + hi) / 2
        if final_balance(start, mid, net) < end:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
