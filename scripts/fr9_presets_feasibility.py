"""FR-9 feasibility: what "Less often" and "More often" do at 0.5x and 2x the promoted alert rate.

Owner decision (Oct 4, 2026): the presets are alert rates, Less = 0.5x and More = 2x the
promoted model's, Balanced = the promoted cutoff unchanged, for both unusual charges (FR-7) and
spending spikes (FR-8). This script measures two things with the promoted models:

1. **Precision and recall per preset, train users only,** through the label contract. FR-7's
   merchant profiles and FR-8's season profiles come from train users only, and FR-8 runs on true
   categories, as validation does. Both promoted cutoffs were placed on train users, so these
   numbers are in-sample for the cutoff: an estimate. The design's milestone measures them out of
   fold. No test user's labels are read.
2. **What the demo shows,** serving's way: every user scored against the whole pool, FR-8 on the
   promoted categorizer's predictions, preset cutoffs matched to the pool's flag count, and each
   demo account's flags in "Worth a look"'s 60-day window. No label is read.

Exact repeats score +inf, so every preset flags every duplicate; the rate is matched over all
flags, duplicates included.

    uv run python scripts/fr9_presets_feasibility.py data/synthetic/default.sqlite \\
        configs/web/demo_accounts.yaml u_te_fb_0000,u_te_fb_0019
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
from smart_financial_coach.intelligence.categorization.batch import categorize_dataset
from smart_financial_coach.intelligence.service import load_service
from smart_financial_coach.intelligence.spikes.contract import INPUT_COLUMNS, evidence_for
from smart_financial_coach.intelligence.spikes.reasons import reason as spike_reason

PRESETS = {"less": 0.5, "balanced": 1.0, "more": 2.0}
WINDOW_DAYS = 60  # "Worth a look" (FR-7 §8)
DAYS_PER_MONTH = 30.4

Scores = npt.NDArray[np.float64]


def preset_cutoffs(score: Scores, balanced: float) -> dict[str, float]:
    """Cutoffs that flag 0.5x and 2x as many rows as the promoted cutoff does on these scores."""
    s = np.sort(score[score > -np.inf])[::-1]
    n = int((s >= balanced).sum())
    out = {}
    for name, m in PRESETS.items():
        k = min(max(round(n * m), 1), len(s))
        out[name] = balanced if m == 1.0 else float(s[k - 1])
    return out


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
    print("| Half | Preset | Flags | Flags per user-month | Precision | Recall |")
    print("| --- | --- | --- | --- | --- | --- |")
    model = load_service("unusual_transactions").model
    scored, score = charge_scores(mine, pool=mine)
    post = (pd.to_datetime(scored["ts"]) >= pd.Timestamp(truth.warmup_end_day)).to_numpy()
    ts = pd.to_datetime(scored.loc[post, "ts"])
    months = ts.groupby(scored.loc[post, "user_id"]).agg(lambda t: (t.max() - t.min()).days).sum()
    months /= DAYS_PER_MONTH
    for name, cutoff in preset_cutoffs(score, model.cutoff).items():
        flagged = score >= cutoff
        flags = scored.loc[flagged, ["transaction_id", "reason_code"]]
        outcomes = truth.score_transactions(flags)
        owner = transactions.set_index("transaction_id")["user_id"]
        r = metrics(outcomes[outcomes["transaction_id"].map(owner).isin(train)])
        rate = (flagged & post).sum() / months
        print(
            f"| Unusual charges | {name} | {flagged.sum():,} | {rate:.3f} "
            f"| {r['precision']:.3f} | {r['recall']:.3f} |"
        )

    spikes = load_service("spending_spikes").model
    pool = mine.merge(truth.transactions[["transaction_id", "category"]], on="transaction_id")
    rows, score = period_scores(pool, store.load_meta(data)["calendar_end"])
    post = (rows["period_start"].astype(str) >= truth.warmup_end_month).to_numpy()
    months = rows.loc[post, ["user_id", "period_start"]].drop_duplicates().shape[0]
    for name, cutoff in preset_cutoffs(score, spikes.cutoff).items():
        flagged = score >= cutoff
        flags = rows.loc[flagged, ["user_id", "category", "period_start"]]
        outcomes = truth.score_periods(flags.astype({"period_start": str}))
        r = metrics(outcomes[outcomes["user_id"].isin(train)])
        rate = (flagged & post).sum() / months
        print(
            f"| Spending spikes | {name} | {flagged.sum():,} | {rate:.4f} "
            f"| {r['precision']:.3f} | {r['recall']:.3f} |"
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
    charge_cuts = preset_cutoffs(charge_score, model.cutoff)

    with tempfile.TemporaryDirectory() as tmp:
        predictions = Path(tmp) / "pool_predictions.sqlite"
        categorize_dataset(data, predictions, overwrite=True)
        categories = load_categories(predictions)[["transaction_id", "category"]]
    pool = transactions.merge(categories, on="transaction_id")
    spikes = load_service("spending_spikes").model
    periods, period_score = period_scores(pool, meta["calendar_end"])
    period_cuts = preset_cutoffs(period_score, spikes.cutoff)
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
    for user in shown:
        print(f"\n### {user}")
        for name in PRESETS:
            c = (charges["user_id"] == user).to_numpy() & in_window
            c &= charge_score >= charge_cuts[name]
            p = (periods["user_id"] == user).to_numpy() & ended
            p &= period_score >= period_cuts[name]
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
