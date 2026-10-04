"""FR-9 feasibility: what "Less often" and "More often" do at 0.5x and 2x the promoted alert rate.

Owner decisions (Oct 4, 2026, decision 4 and review on #65): the presets are alert rates per
post-warm-up user-month, Less = 0.5x and More = 2x the promoted model's, Balanced = the promoted
cutoff unchanged, for both unusual charges (FR-7) and spending spikes (FR-8). A preset's cutoff is
placed as `sfc-model presets` will place it: count the Balanced flags on post-warm-up rows, then
take the score of the k-th highest post-warm-up row, k = 0.5x or 2x that count, with `top_k`'s
tie-break by id. The warm-up is each user's first 90 days (charges) or first 3 months (periods),
counted from their first transaction, a rule that needs no labels; on the synthetic data it's the
label contract's warm-up, since every user starts on the calendar's first day.

This script measures two things with the promoted models:

1. **Precision and recall per preset, train users only,** through the label contract. FR-7's
   merchant profiles and FR-8's season profiles come from train users only, and FR-8 runs on true
   categories, as validation does. Both promoted cutoffs were placed on train users, so these
   numbers are in-sample for the cutoff: an estimate. Milestone 1 measures them out of fold. Flags
   in the warm-up are counted separately: the label contract scores nothing there, so they're
   outside the precision. No test user's labels are read.
2. **What the demo shows,** serving's way: every user scored against the whole pool, FR-8 on the
   promoted categorizer's predictions, preset cutoffs placed on the pool, and each demo account's
   flags in "Worth a look"'s 60-day window. It also counts the users whose window changes, which
   is how the fourth demo account was chosen. No label is read.

Exact repeats score +inf, so every preset flags every duplicate.

    uv run python scripts/fr9_presets_feasibility.py data/synthetic/default.sqlite \\
        configs/web/demo_accounts.yaml u_te_fb_0000,u_te_fb_0019,u_te_fb_0023
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.data import store
from smart_financial_coach.data.features.monthly import (
    MIN_HISTORY,
    monthly_aggregates,
    period_history,
)
from smart_financial_coach.data.features.season_profiles import (
    season_features,
    season_table,
    user_terms,
)
from smart_financial_coach.data.labels import load_truth, metrics
from smart_financial_coach.data.predictions import load_categories
from smart_financial_coach.experience.accounts import load_accounts
from smart_financial_coach.intelligence.anomaly.contract import EVIDENCE, scoring_rows
from smart_financial_coach.intelligence.anomaly.reasons import reason as charge_reason
from smart_financial_coach.intelligence.anomaly.threshold import top_k
from smart_financial_coach.intelligence.categorization.batch import categorize_dataset
from smart_financial_coach.intelligence.service import load_service
from smart_financial_coach.intelligence.spikes.contract import INPUT_COLUMNS, evidence_for
from smart_financial_coach.intelligence.spikes.reasons import reason as spike_reason

PRESETS = {"less": 0.5, "balanced": 1.0, "more": 2.0}
WINDOW_DAYS = 60  # "Worth a look" (FR-7 §8)
WARMUP_DAYS = 90  # FR-2's warm-up for charges
WARMUP_MONTHS = 3  # and for periods
DAYS_PER_MONTH = 30.4

Scores = npt.NDArray[np.float64]
Mask = npt.NDArray[np.bool_]


def preset_cutoffs(
    score: Scores, post: Mask, ids: npt.NDArray[np.str_], balanced: float
) -> dict[str, float]:
    """Cutoffs that flag 0.5x and 2x as many post-warm-up rows as the promoted cutoff does."""
    eligible = np.where(post, score, -np.inf)
    n = int((eligible >= balanced).sum())
    out = {}
    for name, m in PRESETS.items():
        if m == 1.0:
            out[name] = balanced
            continue
        chosen = top_k(eligible, ids, round(n * m)) & (eligible > -np.inf)
        out[name] = float(eligible[chosen].min())
    return out


def charge_post(charges: pd.DataFrame) -> Mask:
    """Charges after each user's first 90 days."""
    ts = pd.to_datetime(charges["ts"])
    first = ts.groupby(charges["user_id"].to_numpy()).transform("min")
    return (ts >= first + pd.Timedelta(days=WARMUP_DAYS)).to_numpy()


def period_post(periods: pd.DataFrame, transactions: pd.DataFrame) -> Mask:
    """Periods starting at least 3 months after each user's first transaction's month."""
    first = pd.to_datetime(transactions.groupby("user_id")["ts"].min()).dt.to_period("M")
    start = pd.to_datetime(periods["period_start"]).dt.to_period("M")
    began = periods["user_id"].map(first)
    return ((start - began).map(lambda d: d.n) >= WARMUP_MONTHS).to_numpy()


def charge_months(charges: pd.DataFrame, post: Mask) -> float:
    """Post-warm-up user-months: each user's post-warm-up span, as FR-7's rate counts them."""
    rows = charges[post]
    ts = pd.to_datetime(rows["ts"])
    span = ts.groupby(rows["user_id"].to_numpy()).agg(lambda t: (t.max() - t.min()).days)
    return float(span.sum() / DAYS_PER_MONTH)


def charge_scores(transactions: pd.DataFrame, pool: pd.DataFrame) -> tuple[pd.DataFrame, Scores]:
    model = load_service("unusual_transactions").model
    rows = scoring_rows(transactions, pool=pool)
    scored = model._scores(rows).assign(user_id=rows["user_id"].to_numpy(), ts=rows["ts"])
    scored["merchant_raw"] = rows["merchant_raw"].to_numpy()
    scored["amount"] = rows["amount"].to_numpy()
    has_reason = scored["reason_code"].notna().to_numpy()
    return scored, np.where(has_reason, scored["score"].to_numpy(dtype=np.float64), -np.inf)


def period_scores(pool: pd.DataFrame, as_of: str) -> tuple[pd.DataFrame, Scores]:
    """Every user's scoring periods, their own season term left out, as serving builds them."""
    model = load_service("spending_spikes").model
    periods = period_history(monthly_aggregates(pool, as_of=as_of))
    season = season_features(periods, season_table(periods), own=user_terms(periods))
    rows = pd.concat([periods, season.drop(columns="period_id")], axis=1)
    rows = rows[rows["usual_months"] >= MIN_HISTORY].reset_index(drop=True)
    x = rows[list(INPUT_COLUMNS)]
    raw = np.nan_to_num(np.asarray(model.base.scores(x), dtype=np.float64), nan=-np.inf)
    return rows, np.where(model.eligible(x), raw, -np.inf)


def precision_on_train(data: Path) -> None:
    truth = load_truth(data)
    with sqlite3.connect(data) as conn:
        users = pd.read_sql_query("SELECT user_id, split FROM users", conn)
    train = set(users.loc[users["split"] == "train", "user_id"])
    transactions = store.load_transactions(data)
    mine = transactions[transactions["user_id"].isin(train)]

    print(f"## Precision and recall per preset: {len(train)} train users (in-sample cutoffs)\n")
    print(
        "| Half | Preset | Flags after warm-up | Per post-warm-up user-month | Precision "
        "| Recall | Flags in warm-up (outside precision) |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    model = load_service("unusual_transactions").model
    scored, score = charge_scores(mine, pool=mine)
    post = charge_post(scored)
    if not (post == (pd.to_datetime(scored["ts"]) >= truth.warmup_end_day).to_numpy()).all():
        raise RuntimeError("the per-user warm-up differs from the label contract's on this data")
    months = charge_months(scored, post)
    ids = scored["transaction_id"].to_numpy()
    owner = transactions.set_index("transaction_id")["user_id"]
    for name, cutoff in preset_cutoffs(score, post, ids, model.cutoff).items():
        flagged = score >= cutoff
        flags = scored.loc[flagged, ["transaction_id", "reason_code"]]
        outcomes = truth.score_transactions(flags)
        r = metrics(outcomes[outcomes["transaction_id"].map(owner).isin(train)])
        after = int((flagged & post).sum())
        print(
            f"| Unusual charges | {name} | {after:,} | {after / months:.3f} "
            f"| {r['precision']:.3f} | {r['recall']:.3f} | {int((flagged & ~post).sum()):,} |"
        )

    spikes = load_service("spending_spikes").model
    pool = mine.merge(truth.transactions[["transaction_id", "category"]], on="transaction_id")
    rows, score = period_scores(pool, store.load_meta(data)["calendar_end"])
    post = period_post(rows, mine)
    months_p = rows.loc[post, ["user_id", "period_start"]].drop_duplicates().shape[0]
    ids = rows["period_id"].to_numpy()
    for name, cutoff in preset_cutoffs(score, post, ids, spikes.cutoff).items():
        flagged = score >= cutoff
        flags = rows.loc[flagged, ["user_id", "category", "period_start"]]
        outcomes = truth.score_periods(flags.astype({"period_start": str}))
        r = metrics(outcomes[outcomes["user_id"].isin(train)])
        after = int((flagged & post).sum())
        print(
            f"| Spending spikes | {name} | {after:,} | {after / months_p:.4f} "
            f"| {r['precision']:.3f} | {r['recall']:.3f} | {int((flagged & ~post).sum()):,} |"
        )


def _plain(value: object) -> object:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value.item() if isinstance(value, np.generic) else value


def demo_view(data: Path, accounts: Path, extra: list[str]) -> None:
    """Each demo account's "Worth a look" at each preset, scored as serving scores it."""
    meta = store.load_meta(data)
    as_of = pd.Timestamp(meta["calendar_end"])
    start = as_of - pd.Timedelta(days=WINDOW_DAYS - 1)
    transactions = store.load_transactions(data)
    shown = [a.user_id for a in load_accounts(accounts)] + extra

    model = load_service("unusual_transactions").model
    charges, charge_score = charge_scores(transactions, pool=transactions)
    in_window = (pd.to_datetime(charges["ts"]).dt.normalize() >= start).to_numpy()
    charge_ids = charges["transaction_id"].to_numpy()
    charge_cuts = preset_cutoffs(charge_score, charge_post(charges), charge_ids, model.cutoff)

    with tempfile.TemporaryDirectory() as tmp:
        predictions = Path(tmp) / "pool_predictions.sqlite"
        categorize_dataset(data, predictions, overwrite=True)
        categories = load_categories(predictions)[["transaction_id", "category"]]
    pool = transactions.merge(categories, on="transaction_id")
    spikes = load_service("spending_spikes").model
    periods, period_score = period_scores(pool, meta["calendar_end"])
    period_ids = periods["period_id"].to_numpy()
    post = period_post(periods, transactions)
    period_cuts = preset_cutoffs(period_score, post, period_ids, spikes.cutoff)
    month = pd.to_datetime(periods["period_start"]).dt.to_period("M")
    # Spikes in months that ended in the window (FR-8 §8)
    ended = (month.dt.end_time.dt.normalize() >= start).to_numpy()

    print(f'\n## The demo\'s "Worth a look": {start.date()} to {as_of.date()}\n')
    print(
        "| Preset | Unusual charges (all users) | Users with one in the window "
        "| Spikes (all users) | Users with one in the window |"
    )
    print("| --- | --- | --- | --- | --- |")
    for name in PRESETS:
        c = charge_score >= charge_cuts[name]
        p = period_score >= period_cuts[name]
        print(
            f"| {name} | {c.sum():,} | {charges.loc[c & in_window, 'user_id'].nunique()} "
            f"| {p.sum():,} | {periods.loc[p & ended, 'user_id'].nunique()} |"
        )

    def window(user: str, name: str) -> tuple[Mask, Mask]:
        c = (charges["user_id"] == user).to_numpy() & in_window
        p = (periods["user_id"] == user).to_numpy() & ended
        return c & (charge_score >= charge_cuts[name]), p & (period_score >= period_cuts[name])

    test_users = sorted(u for u in transactions["user_id"].unique() if u.startswith("u_te_"))
    both, grows = [], []
    for user in test_users:
        bc, bp = window(user, "balanced")
        mc, mp = window(user, "more")
        if mc.sum() + mp.sum() > bc.sum() + bp.sum():
            grows.append(user)
            if bp.sum() >= 1:
                both.append(user)
    print(
        f"\nTest users whose window grows at More often: {len(grows)} of {len(test_users)}; "
        f"of them, with a spike at Balanced too: {len(both)} {both}"
    )

    for user in shown:
        print(f"\n### {user}")
        for name in PRESETS:
            c, p = window(user, name)
            print(f"- {name}: {int(c.sum())} unusual charges, {int(p.sum())} spikes")
            for i in map(int, np.flatnonzero(c)):
                code = str(charges["reason_code"].iat[i])
                evidence = {k: _plain(charges[k].iat[i]) for k in EVIDENCE[code] if k in charges}
                r = charges.iloc[i]
                print(
                    f"  - {str(r['ts'])[:10]} {r['merchant_raw']} ${-r['amount']:.2f}: "
                    f"{charge_reason(code, evidence)}"
                )
            for i in map(int, np.flatnonzero(p)):
                (evidence,) = evidence_for(periods.iloc[[i]][list(INPUT_COLUMNS)])
                print(f"  - {periods['category'].iat[i]} {month.iat[i]}: {spike_reason(evidence)}")


def main() -> None:
    data, accounts = Path(sys.argv[1]), Path(sys.argv[2])
    extra = sys.argv[3].split(",") if len(sys.argv) > 3 else []
    precision_on_train(data)
    demo_view(data, accounts, extra)


if __name__ == "__main__":
    main()
