"""Category season profiles: how much more or less people buy in a category in each month of the
year (FR-8 §2). A feature table built from model-visible rows, never fitted into a model:

- **As of the month.** A period is scored against observations from months before its own.
- **Leave the user out, exactly.** A user's *term* for a (category, month of year) is the mean of
  their season observations (`period_history`'s `log_ratio`) in that month of earlier years. Each
  cell stores the sum of users' terms and the number of users, so the scored user's own term is
  subtracted, not approximated. A median (the POC's) can't be left out this way.
- **Distinct users.** A profile needs at least `min_users` *other* users; with fewer, the season is
  1 (a log season of 0). It's an aggregate over many people's buying, never a merchant or a price.

    table = season_table(pool)  # pool: `period_history` rows on the pool's category basis
    feats = season_features(scored, table, own=user_terms(pool_rows_of_the_scored_users))
    feats = season_profiles(scored, pool)  # the same, when the scored users' rows are in `pool`

In evaluation the pool is train users (validation) or everyone (test), on true categories. In
serving it's every user on the promoted categorizer's predictions, which carry nobody's
corrections, and the served user's own term is computed on that same basis (FR-8 §2).

`season_features` columns, one row per scored period, in input order:
    period_id
    profile_season   mean of other users' terms (log; 0 below `min_users`)
    profile_users    other users with a term
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_MIN_USERS = 20  # FR-8 §2: stability, beyond FR-7's 3-user privacy floor
CELL = ["category", "moy"]
COLUMNS = ("period_id", "profile_season", "profile_users")


def user_terms(periods: pd.DataFrame) -> pd.DataFrame:
    """Each user's term per (category, month of year), and the first as-of month it applies to.

    One row per season observation: `term` is the user's mean observation in that cell so far,
    valid from the month after the observation (`valid_from`) until their next one.
    """
    obs = periods.loc[periods["log_ratio"].notna(), ["user_id", "category", "month", "log_ratio"]]
    obs = obs.assign(moy=obs["month"] % 12).sort_values(
        ["user_id", *CELL, "month"], kind="mergesort"
    )
    g = obs.groupby(["user_id", *CELL], sort=False)["log_ratio"]
    return pd.DataFrame(
        {
            "user_id": obs["user_id"].to_numpy(),
            "category": obs["category"].to_numpy(),
            "moy": obs["moy"].to_numpy(),
            "valid_from": obs["month"].to_numpy() + 1,
            "term": (g.cumsum() / (g.cumcount() + 1)).to_numpy(),
            "first": (g.cumcount() == 0).to_numpy(),
        }
    )


@dataclass(frozen=True)
class SeasonTable:
    """Per (category, month of year, as-of month): the sum of users' terms and their number."""

    cells: pd.DataFrame  # category, moy, as_of, total, users

    def lookup(self, category: pd.Series, moy: pd.Series, as_of: pd.Series) -> pd.DataFrame:
        """`total` and `users` for each row; zeros where the table has nothing yet."""
        keys = pd.DataFrame({"category": category, "moy": moy, "as_of": as_of})
        if self.cells.empty:
            return pd.DataFrame({"total": 0.0, "users": 0}, index=keys.index)
        cells = self.cells.sort_values("as_of", kind="mergesort")
        found = pd.merge_asof(
            keys.reset_index().sort_values("as_of", kind="mergesort"),
            cells,
            on="as_of",
            by=CELL,
            direction="backward",
        ).set_index("index")
        out = found.reindex(keys.index)[["total", "users"]]
        return out.fillna({"total": 0.0, "users": 0}).astype({"users": int})


def season_table(pool: pd.DataFrame) -> SeasonTable:
    """The table from `pool`'s `period_history` rows: one cell row per change in a cell."""
    t = user_terms(pool)
    if t.empty:
        empty = pd.DataFrame(columns=["category", "moy", "as_of", "total", "users"])
        return SeasonTable(empty.astype({"moy": int, "as_of": int, "total": float, "users": int}))
    previous = t.groupby(["user_id", *CELL], sort=False)["term"].shift().fillna(0.0)
    deltas = pd.DataFrame(
        {
            "category": t["category"],
            "moy": t["moy"],
            "as_of": t["valid_from"],
            "total": t["term"] - previous,  # a user's term replaces their previous one
            "users": t["first"].astype(int),
        }
    )
    cells = deltas.groupby([*CELL, "as_of"], sort=True)[["total", "users"]].sum().reset_index()
    cells[["total", "users"]] = cells.groupby(CELL, sort=False)[["total", "users"]].cumsum()
    return SeasonTable(cells.astype({"users": int}))


def _own(scored: pd.DataFrame, own: pd.DataFrame) -> pd.Series:
    """The scored user's own term as of each scored period's month (NaN without one)."""
    keys = pd.DataFrame(
        {
            "user_id": scored["user_id"].to_numpy(),
            "category": scored["category"].to_numpy(),
            "moy": (scored["month"] % 12).to_numpy(),
            "as_of": scored["month"].to_numpy(),
        }
    ).reset_index()
    terms = own.rename(columns={"valid_from": "as_of"})[["user_id", *CELL, "as_of", "term"]]
    found = pd.merge_asof(
        keys.sort_values("as_of", kind="mergesort"),
        terms.sort_values("as_of", kind="mergesort"),
        on="as_of",
        by=["user_id", *CELL],
        direction="backward",
    ).set_index("index")
    return found["term"].reindex(keys["index"]).set_axis(scored.index)


def season_features(
    scored: pd.DataFrame,
    table: SeasonTable,
    own: pd.DataFrame | None = None,
    min_users: int = DEFAULT_MIN_USERS,
) -> pd.DataFrame:
    """Profile columns for `scored` periods, leaving each scored user's `own` terms out.

    `own` is `user_terms` of the scored users' rows on the table's category basis; without it the
    scored users must not be in the table's pool.
    """
    if min_users < 1:
        raise ValueError(f"min_users must be at least 1, got {min_users}")
    cell = table.lookup(scored["category"], scored["month"] % 12, scored["month"])
    mine = _own(scored, own) if own is not None else pd.Series(np.nan, index=scored.index)
    has_own = mine.notna().to_numpy()
    total = cell["total"].to_numpy() - np.where(has_own, mine.fillna(0.0).to_numpy(), 0.0)
    users = cell["users"].to_numpy() - has_own.astype(int)
    if (users < 0).any():
        raise ValueError("a scored user's own term isn't in the table: wrong pool or basis")
    enough = users >= min_users
    season = np.where(enough, total / np.maximum(users, 1), 0.0)
    return pd.DataFrame(
        {
            "period_id": scored["period_id"].to_numpy(),
            "profile_season": season,
            "profile_users": users,
        }
    )


def season_profiles(
    scored: pd.DataFrame, pool: pd.DataFrame, min_users: int = DEFAULT_MIN_USERS
) -> pd.DataFrame:
    """`season_features` when the scored users' own rows are part of `pool`, on its basis."""
    users = set(scored["user_id"])
    own = user_terms(pool[pool["user_id"].isin(users)])
    return season_features(scored, season_table(pool), own=own, min_users=min_users)
