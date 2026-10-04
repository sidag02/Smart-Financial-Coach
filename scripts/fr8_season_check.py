"""FR-8 milestone 1: does the POC's result hold with the clipped-mean season profile?

Decision 13 on #58 (owner, Oct 4, 2026) fixed the pass condition before this check ran. At 0.035
flags per post-warm-up user-month, the POC leader with the clipped-mean profile must have
    recall    within 0.506-0.578, and
    precision within 0.737-0.860,
the POC's user-bootstrap intervals at that rate (`poc/fr-8-spending-spikes` at `bd5ae63`). If
either falls outside, the statistic goes back to the owner before the round; it isn't swapped.

Train users only, as in the POC: every truth table is cut to train users before anything is
computed. The POC's simplifications stay (thresholds and the income elasticity on the same
users); what changes is only the season profile, built by milestone 1's feature pipeline: a
clipped mean of log ratios, as of each month, each scored user left out, at least 20 other users.
The row that rebuilds the POC's median profile reproduced the POC exactly (0.545, 0.809) on the
POC's grid; since review on #59 categories start at their first purchase, which moves it to
0.543 and 0.805.

    uv run python scripts/fr8_season_check.py data/synthetic/default.sqlite
"""

import sys
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import poisson

from smart_financial_coach.data.labels import PERIOD_KEY, Truth, load_truth
from smart_financial_coach.data.store import load_meta
from smart_financial_coach.intelligence.spikes.contract import (
    MIN_USUAL_COUNT,
    SPEND_FLOOR,
    scoring_periods,
)

RATE = 0.035
RECALL = (0.506, 0.578)
PRECISION = (0.737, 0.860)
KAPPA = 1.0  # the POC's shrinkage: one earlier year counts as much as the profile


def train_only(truth: Truth) -> Truth:
    users = truth.users[truth.users["split"] == "train"]
    keep = set(users["user_id"])

    def cut(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["user_id"].isin(keep)].reset_index(drop=True)

    return replace(
        truth,
        transactions=cut(truth.transactions),
        periods=cut(truth.periods),
        expected=cut(truth.expected),
        users=users.reset_index(drop=True),
        preferences=cut(truth.preferences),
    )


def poc_season(rows: pd.DataFrame) -> np.ndarray:
    """The POC's season: a median of ratios over all users and months, shrunk with last year."""
    ratio = rows["count"] / rows["usual_count"].where(rows["usual_count"] > 0)
    ratio = ratio.where(rows["log_ratio"].notna())
    pop = ratio.groupby([rows["category"], rows["month"] % 12]).median()
    pop_v = pd.MultiIndex.from_arrays([rows["category"], rows["month"] % 12]).map(pop.to_dict())
    pop_v = np.asarray(pop_v, dtype=float)
    own = ratio.groupby([rows["user_id"], rows["category"]]).shift(12).to_numpy()
    has = ~np.isnan(own)
    w = has / (has + 1.0)
    return np.where(has, w * np.nan_to_num(own, nan=1.0) + (1 - w) * pop_v, pop_v)


def clipped_season(rows: pd.DataFrame) -> np.ndarray:
    """Milestone 1's season: the user's own log season a year earlier, shrunk toward the
    profile."""
    years = rows["own_years"].to_numpy(dtype=float)
    w = years / (years + KAPPA)
    own = rows["own_season"].fillna(0.0).to_numpy()
    season: np.ndarray = np.exp(w * own + (1 - w) * rows["profile_season"].to_numpy())
    return season


def pooled_clipped_season(rows: pd.DataFrame) -> np.ndarray:
    """A step between the two, for the breakdown: the clipped mean pooled like the POC's median
    (every user, every month, nobody left out), with milestone 1's own season."""
    cell = [rows["category"], rows["month"] % 12]
    pop = rows["log_ratio"].groupby(cell).mean()
    pop_v = np.asarray(pd.MultiIndex.from_arrays(cell).map(pop.to_dict()), dtype=float)
    years = rows["own_years"].to_numpy(dtype=float)
    w = years / (years + KAPPA)
    season: np.ndarray = np.exp(w * rows["own_season"].fillna(0.0).to_numpy() + (1 - w) * pop_v)
    return season


def fit_beta(rows: pd.DataFrame, season: np.ndarray) -> float:
    """The POC's pooled income elasticity of counts (reads no labels)."""
    ok = (rows["usual_count"] >= 3) & rows["income_ratio"].between(0.5, 1.6)
    d, s = rows[ok], season[ok.to_numpy()]
    y = np.log((d["count"] + 0.5) / (d["usual_count"] * s))
    x = np.log(d["income_ratio"])
    good = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[good], y[good], 1)[0])


def leader(rows: pd.DataFrame, season: np.ndarray, plain_floor: bool) -> np.ndarray:
    beta = fit_beta(rows, season)
    income = rows["income_ratio"].clip(0.5, 1.6).fillna(1.0).to_numpy() ** beta
    lam = np.clip(np.clip(rows["usual_count"].to_numpy() * season, 0.3, None) * income, 0.3, None)
    score = -poisson.logsf(rows["count"].to_numpy() - 1, lam)
    usual = rows["usual"].to_numpy() * (1.0 if plain_floor else season)
    allowed = (rows["spend"].to_numpy() >= SPEND_FLOOR * np.clip(usual, 1, None)) & (
        rows["usual_count"].to_numpy() >= MIN_USUAL_COUNT
    )
    return np.where(allowed, score, np.nan)


def at_rate(score: np.ndarray, outcome: pd.Series, user_months: int) -> tuple[float, float, int]:
    keep = (outcome != "ignored").to_numpy() & ~np.isnan(score)
    k = round(RATE * user_months)
    order = np.argsort(-score[keep], kind="mergesort")[:k]
    hits = (outcome[keep].to_numpy() == "tp")[order]
    return float(hits.sum()), float(hits.mean()), k


def main(path: str) -> None:
    truth = train_only(load_truth(path))
    tx = truth.transactions
    rows = scoring_periods(tx, tx, as_of=load_meta(path)["calendar_end"])
    rows = rows[rows["period_start"] >= truth.warmup_end_month].reset_index(drop=True)
    scored = truth.score_periods(rows[PERIOD_KEY])
    outcome = rows.merge(scored[scored["outcome"] != "fn"], on=PERIOD_KEY, how="left")["outcome"]
    n_labels = len(truth.spikes("month"))
    user_months = rows.drop_duplicates(["user_id", "period_start"]).shape[0]
    print(
        f"train users {truth.users.shape[0]}, periods {len(rows)}, user-months {user_months}, "
        f"labels {n_labels}"
    )
    variants = {
        "POC median profile, POC floor (the POC's own grid gave 0.545 and 0.809)": (
            poc_season(rows),
            False,
        ),
        "clipped mean pooled like the POC, milestone 1's own season (breakdown)": (
            pooled_clipped_season(rows),
            False,
        ),
        "clipped-mean profile, POC floor (the check)": (clipped_season(rows), False),
        "clipped-mean profile, floor on the plain usual (as designed)": (
            clipped_season(rows),
            True,
        ),
    }
    result = {}
    for name, (season, plain) in variants.items():
        tp, precision, k = at_rate(leader(rows, season, plain), outcome, user_months)
        result[name] = (tp / n_labels, precision)
        print(f"- {name}: flags {k}, recall {tp / n_labels:.3f}, precision {precision:.3f}")
    few = rows["profile_users"] < 20
    print(
        f"- periods whose profile has fewer than 20 other users (season 1): {few.mean():.3f}, "
        f"all in {sorted(rows.loc[few, 'period_start'].str[:4].unique())}"
    )
    recall, precision = result["clipped-mean profile, POC floor (the check)"]
    holds = RECALL[0] <= recall <= RECALL[1] and PRECISION[0] <= precision <= PRECISION[1]
    print(
        f"\nPass condition (decision 13): recall in {RECALL}, precision in {PRECISION} -> "
        f"{'HOLDS' if holds else 'DOES NOT HOLD: back to the owner before the round'}"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/default.sqlite")
