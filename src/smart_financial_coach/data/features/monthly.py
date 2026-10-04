"""Monthly aggregates and their point-in-time history, for spending-spike scoring (FR-8 §1, §2).

A period is one (user, spending category, month). The category comes from the `category` column
the caller fills: true categories in evaluation, the user's effective categories in serving. The
scorer doesn't know which (FR-8 §2).

    periods = monthly_aggregates(transactions, as_of="2026-09-30")  # complete months, zero-filled
    periods = period_history(periods)  # adds each period's history, from earlier months only

`monthly_aggregates` columns:
    period_id      "<user_id>|<category>|<period_start>"
    user_id, category
    period_start   the month's first day, "YYYY-MM-01"
    month          months since year 0 (period arithmetic without dates)
    spend          net outflow in the category: outflows minus refunds
    count          outflows in the category
    income         the user's Income inflows that month (the same on each of the user's rows)

Each (user, category) runs from the user's first purchase in the category to their last month,
with zeros where they bought nothing. A category adopted later has no zeros before its first use,
so they can't pull its usual level down (review on #59). With `as_of`, only months complete by
that date are kept: v1 never scores a month in progress (FR-8 §1, decision 10). Income is never a
period.

`period_history` adds, from the user's earlier months only, so appending later months never
changes a period's history:
    usual_months   earlier months in the window (at most `window`)
    usual          their mean spend: the reason's "usual" (FR-8 §7)
    usual_count    their mean count
    count_var      their count variance (NaN with fewer than 2)
    log_ratio      log(count / usual_count) clipped to ±log 3: this month's season observation
                   (NaN without a usual count or with fewer than `MIN_HISTORY` earlier months)
    own_season     `log_ratio` in the same month a year earlier (FR-8 §2; NaN without one)
    own_years      1 when there is one, else 0: the shrinkage weight's `years`
    income_ratio   mean income over the previous `income_months` over the mean over the window
                   (NaN without income in the window)
"""

import numpy as np
import pandas as pd

INCOME = "Income"
WINDOW = 12  # months that describe a usual month (FR-8 §2)
YEAR = 12
MIN_HISTORY = 3  # earlier months before a period is scored (FR-8 §1; the label warm-up's length)
INCOME_MONTHS = 2  # spending follows the previous two months' income
CLIP = float(np.log(3.0))  # a season observation is at most 3x or a third of usual
KEY = ["user_id", "category"]

AGGREGATE_COLUMNS = (
    "period_id",
    "user_id",
    "category",
    "period_start",
    "month",
    "spend",
    "count",
    "income",
)
HISTORY_COLUMNS = (
    "usual_months",
    "usual",
    "usual_count",
    "count_var",
    "log_ratio",
    "own_season",
    "own_years",
    "income_ratio",
)


def month_index(period_start: pd.Series) -> pd.Series:
    """'2026-08-01' -> 2026 * 12 + 7."""
    s = period_start.astype(str)
    return s.str[:4].astype(int) * 12 + s.str[5:7].astype(int) - 1


def month_start(month: pd.Series | np.ndarray) -> pd.Series:
    """The inverse of `month_index`."""
    m = pd.Series(np.asarray(month, dtype=np.int64))
    return (m // 12).astype(str).str.zfill(4) + "-" + (m % 12 + 1).astype(str).str.zfill(2) + "-01"


def period_ids(user_id: pd.Series, category: pd.Series, period_start: pd.Series) -> pd.Series:
    return user_id.astype(str) + "|" + category.astype(str) + "|" + period_start.astype(str)


def last_complete_month(as_of: str) -> int:
    """The latest month that has ended by `as_of` (a date): its own month when `as_of` is the
    month's last day, otherwise the month before."""
    day = pd.Timestamp(as_of)
    month = day.year * 12 + day.month - 1
    return month if day.is_month_end else month - 1


def monthly_aggregates(transactions: pd.DataFrame, as_of: str | None = None) -> pd.DataFrame:
    """One row per (user, spending category, month), zero-filled, sorted by user, category, month.

    `transactions` has `user_id`, `ts`, `amount` and `category`. `as_of` (a date) keeps only
    months complete by then, and drops later transactions.
    """
    if missing := [c for c in ("user_id", "ts", "amount", "category") if c not in transactions]:
        raise ValueError(f"transactions lack {missing}")
    t = transactions[["user_id", "ts", "amount", "category"]].copy()
    t["month"] = month_index(t["ts"].astype(str).str[:7] + "-01")
    if as_of is not None:
        t = t[t["month"] <= last_complete_month(as_of)]
    last = t.groupby("user_id")["month"].max()
    income = t[t["category"] == INCOME].groupby(["user_id", "month"])["amount"].sum()
    spending = t[t["category"] != INCOME]
    g = spending.groupby([*KEY, "month"])["amount"]
    agg = pd.DataFrame(
        {"spend": -g.sum(), "count": g.apply(lambda a: int((a < 0).sum()))}
    ).reset_index()

    used = spending.groupby(KEY)["month"].min().rename("first").reset_index()
    used["last"] = used["user_id"].map(last).to_numpy()
    reps = (used["last"] - used["first"] + 1).to_numpy()
    grid = used.loc[used.index.repeat(reps), KEY].reset_index(drop=True)
    first = np.repeat(used["first"].to_numpy(), reps)
    offset = grid.groupby(KEY, sort=False).cumcount().to_numpy()
    grid["month"] = first + offset

    p = grid.merge(agg, on=[*KEY, "month"], how="left").fillna({"spend": 0.0, "count": 0})
    p["count"] = p["count"].astype(int)
    p["income"] = (
        pd.MultiIndex.from_frame(p[["user_id", "month"]]).map(income.to_dict()).fillna(0.0)
    )
    p["income"] = p["income"].astype(float)
    p["period_start"] = month_start(p["month"]).to_numpy()
    p["period_id"] = period_ids(p["user_id"], p["category"], p["period_start"])
    p = p.sort_values([*KEY, "month"], kind="mergesort").reset_index(drop=True)
    return p[list(AGGREGATE_COLUMNS)]


def _trailing(values: pd.Series, groups: list[pd.Series], fn: str, window: int) -> pd.Series:
    """`fn` over the previous `window` rows of each group (shifted: earlier months only)."""
    return values.groupby(groups, sort=False).transform(
        lambda x: getattr(x.shift(1).rolling(window, min_periods=1), fn)()
    )


def period_history(
    periods: pd.DataFrame, window: int = WINDOW, income_months: int = INCOME_MONTHS
) -> pd.DataFrame:
    """`periods` (from `monthly_aggregates`) with the history columns, from earlier months only."""
    p = periods.sort_values([*KEY, "month"], kind="mergesort").reset_index(drop=True)
    keys = [p["user_id"], p["category"]]
    p["usual_months"] = p.groupby(KEY, sort=False).cumcount().clip(upper=window).astype(int)
    p["usual"] = _trailing(p["spend"], keys, "mean", window)
    p["usual_count"] = _trailing(p["count"].astype(float), keys, "mean", window)
    p["count_var"] = (
        p["count"]
        .astype(float)
        .groupby(keys, sort=False)
        .transform(lambda x: x.shift(1).rolling(window, min_periods=2).var())
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(p["count"].to_numpy(dtype=float) / p["usual_count"].to_numpy())
    observed = (p["usual_count"].to_numpy() > 0) & (p["usual_months"].to_numpy() >= MIN_HISTORY)
    p["log_ratio"] = np.clip(np.where(observed, ratio, np.nan), -CLIP, CLIP)

    # The same month a year earlier (FR-8 §2): rows are consecutive months within a category
    p["own_season"] = p["log_ratio"].groupby(keys, sort=False).shift(YEAR)
    p["own_years"] = p["own_season"].notna().astype(int)

    users = p.drop_duplicates(["user_id", "month"])[["user_id", "month", "income"]]
    by_user = users.groupby("user_id", sort=False)["income"]
    recent = by_user.transform(lambda x: x.shift(1).rolling(income_months, min_periods=1).mean())
    year = by_user.transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    ratio_by_month = (recent / year.where(year > 0)).to_numpy()
    lookup = dict(
        zip(zip(users["user_id"], users["month"], strict=True), ratio_by_month, strict=True)
    )
    p["income_ratio"] = [lookup[k] for k in zip(p["user_id"], p["month"], strict=True)]
    return p
