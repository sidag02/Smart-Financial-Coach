"""Data quality checks, run after generation and in CI. Fail loudly instead of feeding bad data on.

Checks are statistical where the data is random, with tolerances wide enough that a correct
generator passes on the small spec and narrow enough to catch a broken stage.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from smart_financial_coach.data.generator.catalog import load_catalog
from smart_financial_coach.data.generator.dataset import TABLES, Dataset
from smart_financial_coach.data.generator.spec import Spec, season_vector
from smart_financial_coach.data.generator.taxonomy import INCOME

MIN_HISTORY_MONTHS = 24
MIN_INCOME_MONTH_SHARE = 0.6  # freelancers have slow months with no payments
SAVINGS_TOLERANCE = 0.1
SAVINGS_MIN_SHARE_IN_RANGE = 0.85
SEASONAL_MIN_VARIATION = 1.5  # only check profiles that vary at least this much
SEASONAL_MIN_CORRELATION = 0.5
MESS_MIN_SHARE = {"light": 0.3, "realistic": 0.6, "heavy": 0.8}
EVENT_COUNT_TOLERANCE = 0.4
MIN_EVENTS_FOR_RATE_CHECK = 30
MIN_SPIKES_FOR_LIFT_CHECK = 10


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)


def _schema(ds: Dataset, report: Report) -> None:
    for name, table in TABLES.items():
        df = ds.tables.get(name)
        if df is None:
            report.errors.append(f"schema: missing table {name}")
            continue
        report.check(
            list(df.columns) == table.column_names, f"schema: {name} columns {list(df.columns)}"
        )
        for col in table.columns:
            if not col.nullable and col.name in df and df[col.name].isna().any():
                report.errors.append(f"schema: {name}.{col.name} has nulls")
        report.check(
            not df.duplicated(list(table.primary_key)).any(),
            f"schema: {name} primary key not unique",
        )
    ids = set(ds["transactions"]["transaction_id"])
    report.check(
        ids == set(ds["truth_transactions"]["transaction_id"]),
        "schema: transactions and truth rows differ",
    )


def _ledger(ds: Dataset) -> pd.DataFrame:
    tx = ds["transactions"].merge(
        ds["truth_transactions"], on="transaction_id", validate="one_to_one"
    )
    tx = tx.merge(ds["users"][["user_id", "persona", "split"]], on="user_id")
    tx["month"] = tx["ts"].str[:7]
    tx["moy"] = tx["ts"].str[5:7].astype(int)
    return tx


def _coverage(tx: pd.DataFrame, ds: Dataset, spec: Spec, report: Report) -> None:
    months = pd.period_range(spec.calendar.start, spec.calendar.end, freq="M")
    report.check(len(months) >= MIN_HISTORY_MONTHS, f"coverage: calendar has {len(months)} months")
    income_months = tx[tx["process"] == "income"].groupby("user_id")["month"].nunique()
    income_share = income_months.reindex(ds["users"]["user_id"], fill_value=0) / len(months)
    report.stats["min_income_month_share"] = float(income_share.min())
    if bad := sorted(income_share[income_share < MIN_INCOME_MONTH_SHARE].index):
        report.errors.append(f"coverage: users with income in too few months: {bad[:5]}")
    seen = tx.groupby("persona")["category"].agg(set)
    for name, persona in spec.personas.items():
        if name in seen.index and (missing := persona.categories() - seen[name]):
            report.errors.append(f"coverage: persona {name} never spends in {sorted(missing)}")


def _plausibility(tx: pd.DataFrame, ds: Dataset, spec: Spec, report: Report) -> None:
    users = ds["users"].set_index("user_id")
    income = tx[tx["amount"] > 0].groupby("user_id")["amount"].sum()
    net = tx.groupby("user_id")["amount"].sum()
    rate = (net / income).reindex(users.index)
    in_range = [
        spec.personas[p].savings_rate[0] - SAVINGS_TOLERANCE
        <= r
        <= spec.personas[p].savings_rate[1] + SAVINGS_TOLERANCE
        for r, p in zip(rate, users["persona"], strict=True)
    ]
    share = float(np.mean(in_range))
    report.stats["savings_rate_in_range_share"] = share
    report.check(
        share >= SAVINGS_MIN_SHARE_IN_RANGE,
        f"plausibility: only {share:.0%} of users near their savings target",
    )

    ordered = tx.sort_values(["user_id", "ts", "transaction_id"], kind="mergesort")
    running = ordered.groupby("user_id")["amount"].cumsum() + ordered["user_id"].map(
        users["starting_balance"]
    )
    report.check(
        float(running.min()) >= -0.01, f"plausibility: running balance reaches {running.min():.2f}"
    )

    per_month = tx.groupby(["user_id", "month"]).size()
    report.stats["median_txns_per_user_month"] = float(per_month.median())
    report.check(
        20 <= per_month.median() <= 250,
        f"plausibility: median {per_month.median()} transactions per user-month",
    )


def _correlated(expected: np.ndarray, realized: np.ndarray) -> bool:
    if expected.max() < SEASONAL_MIN_VARIATION * max(expected.min(), 1e-9) or realized.std() == 0:
        return True
    return float(np.corrcoef(expected, realized)[0, 1]) >= SEASONAL_MIN_CORRELATION


def _seasonality(tx: pd.DataFrame, spec: Spec, report: Report) -> None:
    """Realized month-of-year profiles must follow the spec's seasonal profiles."""
    catalog = load_catalog(spec.catalog.merchants, spec.catalog.holdout, spec.categories)
    prices = catalog.merchants.groupby(["category", "subtype"])["price_median"].mean()
    months = np.arange(1, 13)
    for name, persona in spec.personas.items():
        ptx = tx[tx["persona"] == name]
        if ptx.empty:
            continue
        mean_net = float(np.mean(persona.income.monthly_net))
        profiles: dict[tuple[str, str], np.ndarray] = {}
        for s in persona.discretionary:
            price = (
                float(prices.loc[s.category].reindex(s.subtypes or None).mean())
                if s.category in prices
                else 1.0
            )
            key = ("discretionary", s.category)
            profiles[key] = (
                profiles.get(key, np.zeros(12))
                + np.mean(s.weekly_rate) * price * season_vector(s.seasonality)[months]
            )
        for r in persona.recurring:
            if r.amount is not None:
                level = float(np.mean(r.amount))
            elif r.income_share is not None:
                level = float(np.mean(r.income_share)) * mean_net
            else:
                level = float(prices.loc[r.category].reindex(r.subtypes or None).mean())
            active = np.isin(months, r.months) if r.months is not None else np.ones(12, dtype=bool)
            key = ("recurring", r.category)
            profiles[key] = (
                profiles.get(key, np.zeros(12))
                + r.probability * level * active * season_vector(r.seasonality)[months]
            )
        for (process, category), expected in profiles.items():
            sub = ptx[(ptx["process"] == process) & (ptx["category"] == category)]
            by_month = sub.groupby(["month", "moy"])["amount"].sum().groupby("moy").mean()
            realized = -by_month.reindex(months, fill_value=0.0).to_numpy()
            if not _correlated(expected, realized):
                report.errors.append(
                    f"seasonality: {name} {process} {category} does not follow its seasonal profile"
                )


def _mess(tx: pd.DataFrame, ds: Dataset, spec: Spec, report: Report) -> None:
    names = ds["truth_merchants"].set_index("merchant_id")["canonical_name"]
    canonical = tx["merchant_id"].map(names)
    differs = float((tx["merchant_raw"] != canonical).mean())
    report.stats["merchant_raw_differs_share"] = differs
    distortion = spec.rendering.distortion
    if distortion == "none":
        report.check(
            differs == 0.0, "mess: distortion none but merchant_raw differs from canonical names"
        )
        return
    report.check(
        differs >= MESS_MIN_SHARE[distortion],
        f"mess: only {differs:.0%} of merchant_raw values are distorted",
    )
    variants = tx.groupby("merchant_id").agg(
        n=("merchant_raw", "size"), distinct=("merchant_raw", "nunique")
    )
    busy = variants[variants["n"] >= 30]
    if len(busy):
        multi = float((busy["distinct"] >= 2).mean())
        report.stats["busy_merchants_with_variants_share"] = multi
        report.check(
            multi >= 0.5, f"mess: only {multi:.0%} of busy merchants have several text variants"
        )


def _events(tx: pd.DataFrame, ds: Dataset, spec: Spec, report: Report) -> None:
    years = (spec.calendar.end - spec.calendar.start).days / 365.25
    n_users = len(ds["users"])
    expected = spec.events.unusual_charge.rate_per_user_year * years * n_users
    actual = int((tx["process"] == "unusual_charge").sum())
    report.stats["unusual_charges"] = actual
    if expected >= MIN_EVENTS_FOR_RATE_CHECK:
        report.check(
            abs(actual - expected) <= EVENT_COUNT_TOLERANCE * expected,
            f"events: {actual} unusual charges, expected about {expected:.0f}",
        )
    spikes = ds["truth_periods"]
    report.stats["spikes"] = len(spikes)
    if len(spikes) < MIN_SPIKES_FOR_LIFT_CHECK:
        return
    disc = tx[tx["process"] == "discretionary"].copy()
    day = pd.to_datetime(disc["ts"].str[:10])
    disc["month_start"] = day.dt.to_period("M").dt.start_time.dt.strftime("%Y-%m-%d")
    disc["week_start"] = (day - pd.to_timedelta(day.dt.dayofweek, unit="D")).dt.strftime("%Y-%m-%d")
    lifts = []
    for granularity, col in (("month", "month_start"), ("week", "week_start")):
        totals = -disc.groupby(["user_id", "category", col])["amount"].sum()
        baseline = totals.groupby(["user_id", "category"]).median()
        for s in spikes[spikes["granularity"] == granularity].itertuples():
            key = (s.user_id, s.category, s.period_start)
            if key in totals.index and (s.user_id, s.category) in baseline.index:
                lifts.append(totals[key] / max(baseline[(s.user_id, s.category)], 1e-9))
    if lifts:
        lift = float(np.median(lifts))
        floor = 1 + 0.4 * (spec.events.spending_spike.multiplier[0] - 1)
        report.stats["median_spike_lift"] = lift
        report.check(lift >= floor, f"events: median spike lift {lift:.2f} below {floor:.2f}")


def _splits(tx: pd.DataFrame, ds: Dataset, report: Report) -> None:
    holdout = set(ds["truth_merchants"].loc[ds["truth_merchants"]["holdout"] == 1, "merchant_id"])
    leaked = tx[(tx["split"] == "train") & tx["merchant_id"].isin(holdout)]
    report.check(leaked.empty, f"splits: {len(leaked)} train transactions at holdout merchants")
    test_spend = tx[(tx["split"] == "test") & (tx["category"] != INCOME)]
    if len(test_spend):
        report.stats["test_holdout_share"] = float(test_spend["merchant_id"].isin(holdout).mean())
    by_split = ds["users"].groupby("split")["user_id"].agg(set)
    if {"train", "test"} <= set(by_split.index):
        report.check(
            not (by_split["train"] & by_split["test"]), "splits: users in both train and test"
        )


def _goals(ds: Dataset, report: Report) -> None:
    truth = ds["truth_goals"]
    known = truth[truth["met"].notna()]
    report.stats["goals_with_known_outcome"] = len(known)
    on_track = known[known["outcome_class"] == "on_track"]["met"]
    off_track = known[known["outcome_class"] == "off_track"]["met"]
    report.check(
        bool((on_track == 1).all()), "goals: an on-track goal with a known outcome was not met"
    )
    report.check(
        bool((off_track == 0).all()), "goals: an off-track goal with a known outcome was met"
    )
    report.check(
        set(ds["goals"]["user_id"]) <= set(ds["users"]["user_id"]), "goals: unknown user_id"
    )


def validate(ds: Dataset, spec: Spec) -> Report:
    report = Report()
    _schema(ds, report)
    if not report.ok:
        return report
    tx = _ledger(ds)
    report.stats["users"] = len(ds["users"])
    report.stats["transactions"] = len(tx)
    _coverage(tx, ds, spec, report)
    _plausibility(tx, ds, spec, report)
    _seasonality(tx, spec, report)
    _mess(tx, ds, spec, report)
    _events(tx, ds, spec, report)
    _splits(tx, ds, report)
    _goals(ds, report)
    return report
