"""Monthly aggregates and their point-in-time history, for spending-spike scoring (FR-8 §1, §2).

A period is one (user, spending category, month). The category comes from the `category` column
the caller fills: true categories in evaluation, the user's effective categories in serving. The
scorer doesn't know which (FR-8 §2).

    periods = monthly_aggregates(transactions)  # one row per period, zero-filled
    periods = period_history(periods)  # adds each period's history, from earlier months only

`monthly_aggregates` columns:
    period_id      "<user_id>|<category>|<period_start>"
    user_id, category
    period_start   the month's first day, "YYYY-MM-01"
    month          months since year 0 (period arithmetic without dates)
    spend          net outflow in the category: outflows minus refunds
    count          outflows in the category
    income         the user's Income inflows that month (the same on each of the user's rows)

Each user's grid runs from their first month to their last (or `through`), for every spending
category they've used, with zeros where they bought nothing. Income is never a period.

`period_history` adds, from the user's earlier months only, so appending later months never
changes a period's history:
    usual_months   earlier months in the window (at most `window`)
    usual          their mean spend: the reason's "usual" (FR-8 §7)
    usual_count    their mean count
    count_var      their count variance (NaN with fewer than 2)
    log_ratio      log(count / usual_count) clipped to ±log 3: this month's season observation
                   (NaN without a usual count or with fewer than `MIN_HISTORY` earlier months)
    own_season     mean `log_ratio` in the same month of earlier years (NaN without one)
    own_years      how many such months there are
    income_ratio   mean income over the previous `income_months` over the mean over the window
                   (NaN without income in the window)
"""

import numpy as np
import pandas as pd

INCOME = "Income"
WINDOW = 12  # months that describe a usual month (FR-8 §2)
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


def monthly_aggregates(transactions: pd.DataFrame, through: str | None = None) -> pd.DataFrame:
    """One row per (user, spending category, month), zero-filled, sorted by user, category, month.

    `transactions` has `user_id`, `ts`, `amount` and `category`. `through` (a month start) drops
    later months, for scoring as of a date.
    """
    if missing := [c for c in ("user_id", "ts", "amount", "category") if c not in transactions]:
        raise ValueError(f"transactions lack {missing}")
    t = transactions[["user_id", "ts", "amount", "category"]].copy()
    t["month"] = month_index(t["ts"].astype(str).str[:7] + "-01")
    if through is not None:
        t = t[t["month"] <= int(month_index(pd.Series([through])).iloc[0])]
    span = t.groupby("user_id")["month"].agg(["min", "max"])
    income = t[t["category"] == INCOME].groupby(["user_id", "month"])["amount"].sum()
    spending = t[t["category"] != INCOME]
    g = spending.groupby([*KEY, "month"])["amount"]
    agg = pd.DataFrame(
        {"spend": -g.sum(), "count": g.apply(lambda a: int((a < 0).sum()))}
    ).reset_index()

    used = spending[KEY].drop_duplicates()
    used = used.join(span, on="user_id")
    reps = (used["max"] - used["min"] + 1).to_numpy()
    grid = used.loc[used.index.repeat(reps), KEY].reset_index(drop=True)
    first = np.repeat(used["min"].to_numpy(), reps)
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

    # The same month of earlier years: a running mean per (user, category, month of year),
    # excluding the period itself
    same = [p["user_id"], p["category"], p["month"] % 12]
    value = p["log_ratio"].fillna(0.0)
    seen = p["log_ratio"].notna().astype(int)
    total = value.groupby(same, sort=False).cumsum() - value
    years = seen.groupby(same, sort=False).cumsum() - seen
    p["own_years"] = years.astype(int)
    p["own_season"] = (total / years).where(years > 0)

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
