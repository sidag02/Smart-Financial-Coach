"""Data quality checks, run after generation and in CI. Fail loudly instead of feeding bad data on.

Checks are statistical where the data is random, with tolerances wide enough that a correct
generator passes on the small spec and narrow enough to catch a broken stage.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import chi2

from smart_financial_coach.data.generator.catalog import load_catalog
from smart_financial_coach.data.generator.dataset import TABLES, Dataset
from smart_financial_coach.data.generator.spec import Spec, season_vector, spec_hash
from smart_financial_coach.data.generator.taxonomy import INCOME
from smart_financial_coach.data.labels import Truth

MIN_HISTORY_MONTHS = 24
MIN_INCOME_MONTH_SHARE = 0.6  # freelancers have slow months with no payments
SAVINGS_TOLERANCE = 0.1
SAVINGS_MIN_SHARE_IN_RANGE = 0.85
SEASONAL_MIN_VARIATION = 1.5  # only check profiles that vary at least this much
SEASONAL_MIN_CORRELATION = 0.5
# Poisson goodness of fit across the 12 months; calibrated at any volume
SEASONAL_FIT_MIN_P = 1e-4
POISSON_PROCESSES = ("discretionary", "one_off")
MESS_MIN_SHARE = {"light": 0.3, "realistic": 0.6, "heavy": 0.8}
EVENT_COUNT_TOLERANCE = 0.4
MIN_EVENTS_FOR_RATE_CHECK = 30
MIN_SPIKES_FOR_LIFT_CHECK = 10
# The holdout share varies a lot between users; below this many test users it is only reported
MIN_TEST_USERS_FOR_HOLDOUT_CHECK = 30  # also the floor for the preference adoption-rate check
PREFERENCE_SIGMAS = 3.0  # adoption rate tolerance, in binomial standard deviations
# About 3% of amount outliers land inside the user's normal range at high-variance marketplaces;
# the small spec has few enough outliers that a stricter floor fails by chance
MIN_OUTLIER_ABOVE_NORMAL_SHARE = 0.85
MIN_SPIKE_GAP_DAYS = 8  # plan_spikes keeps a 7-day buffer around each spike in a category


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
    # Income as calibration measures it: refunds are positive but are not income
    income = tx[tx["process"] == "income"].groupby("user_id")["amount"].sum()
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


def _realized_vs_expected(
    tx: pd.DataFrame, ds: Dataset, report: Report
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Per (persona, category): expected and realized Poisson purchases by month of year.

    Months with a planted spike in the user's category are left out of both, since expected
    counts exclude spikes.
    """
    expected = ds["truth_expected"].merge(ds["users"][["user_id", "persona"]], on="user_id")
    expected["month"] = expected["period_start"].str[:7]
    spikes = ds["truth_periods"]
    spiked = pd.DataFrame(
        {
            "user_id": spikes["user_id"],
            "category": spikes["category"],
            "month": spikes["period_start"].str[:7],
            "spiked": True,
        }
    ).drop_duplicates()
    expected = expected.merge(spiked, on=["user_id", "category", "month"], how="left")
    expected = expected[expected["spiked"].isna()]
    purchases = tx[tx["process"].isin(POISSON_PROCESSES)]
    realized = purchases.groupby(["user_id", "category", "month"]).size().rename("realized")
    cells = expected.merge(realized.reset_index(), on=["user_id", "category", "month"], how="left")
    cells["moy"] = cells["month"].str[5:7].astype(int)
    totals = cells.groupby(["persona", "category", "moy"])[["expected_count", "realized"]].sum()
    out = {}
    for (persona, category), rows in totals.groupby(level=[0, 1]):
        by_moy = rows.droplevel([0, 1]).reindex(range(1, 13), fill_value=0.0)
        out[(str(persona), str(category))] = (
            by_moy["expected_count"].to_numpy(dtype=np.float64),
            by_moy["realized"].fillna(0).to_numpy(dtype=np.float64),
        )
    return out


def _seasonality(tx: pd.DataFrame, ds: Dataset, spec: Spec, report: Report) -> None:
    """Month-of-year profiles must follow the spec's seasonal profiles.

    Discretionary purchases get two checks that hold at any volume: the generator's exact expected
    counts must follow the spec's profile (structural), and realized counts must fit them
    (Poisson goodness of fit). Recurring bills, which are low-noise, are checked by correlating
    realized spend with the spec's profile.
    """
    catalog = load_catalog(spec.catalog.merchants, spec.catalog.holdout, spec.categories)
    prices = catalog.merchants.groupby(["category", "subtype"])["price_median"].mean()
    months = np.arange(1, 13)
    fits = _realized_vs_expected(tx, ds, report)
    tested = 0
    for name, persona in spec.personas.items():
        ptx = tx[tx["persona"] == name]
        if ptx.empty:
            continue
        mean_net = float(np.mean(persona.income.monthly_net))
        rates: dict[str, np.ndarray] = {}
        for s in persona.discretionary:
            rates[s.category] = (
                rates.get(s.category, np.zeros(12))
                + np.mean(s.weekly_rate) * season_vector(s.seasonality)[months]
            )
        for category, profile in rates.items():
            if (name, category) not in fits:
                continue
            expected, realized = fits[(name, category)]
            tested += 1
            if not _correlated(profile, expected):
                report.errors.append(
                    f"seasonality: {name} {category} expected purchases don't follow the spec"
                )
            fit = float(chi2.sf(((realized - expected) ** 2 / expected.clip(min=1e-9)).sum(), 12))
            if fit < SEASONAL_FIT_MIN_P:
                report.errors.append(
                    f"seasonality: {name} {category} purchases don't fit their expected counts "
                    f"(p = {fit:.1e})"
                )
        recurring: dict[str, np.ndarray] = {}
        for r in persona.recurring:
            if r.amount is not None:
                level = float(np.mean(r.amount))
            elif r.income_share is not None:
                level = float(np.mean(r.income_share)) * mean_net
            else:
                level = float(prices.loc[r.category].reindex(r.subtypes or None).mean())
            active = np.isin(months, r.months) if r.months is not None else np.ones(12, dtype=bool)
            recurring[r.category] = (
                recurring.get(r.category, np.zeros(12))
                + r.probability * level * active * season_vector(r.seasonality)[months]
            )
        for category, profile in recurring.items():
            sub = ptx[(ptx["process"] == "recurring") & (ptx["category"] == category)]
            if sub.empty:
                continue
            tested += 1
            by_month = sub.groupby(["month", "moy"])["amount"].sum().groupby("moy").mean()
            spend = -by_month.reindex(months, fill_value=0.0).to_numpy()
            if not _correlated(profile, spend):
                report.errors.append(
                    f"seasonality: {name} recurring {category} does not follow its seasonal profile"
                )
    report.stats["seasonal_profiles_tested"] = tested


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


def _splits(tx: pd.DataFrame, ds: Dataset, spec: Spec, report: Report) -> None:
    holdout = set(ds["truth_merchants"].loc[ds["truth_merchants"]["holdout"] == 1, "merchant_id"])
    leaked = tx[(tx["split"] == "train") & tx["merchant_id"].isin(holdout)]
    report.check(leaked.empty, f"splits: {len(leaked)} train transactions at holdout merchants")
    test_spend = tx[(tx["split"] == "test") & (tx["category"] != INCOME)]
    if len(test_spend):
        share = float(test_spend["merchant_id"].isin(holdout).mean())
        report.stats["test_holdout_share"] = share
        low, high = spec.catalog.holdout.test_share_range
        if test_spend["user_id"].nunique() >= MIN_TEST_USERS_FOR_HOLDOUT_CHECK:
            report.check(
                low <= share <= high,
                f"splits: {share:.1%} of test spending is at holdout merchants, "
                f"outside {low:.0%}-{high:.0%}",
            )
    by_split = ds["users"].groupby("split")["user_id"].agg(set)
    if {"train", "test"} <= set(by_split.index):
        report.check(
            not (by_split["train"] & by_split["test"]), "splits: users in both train and test"
        )


def _preferences(ds: Dataset, spec: Spec, report: Report) -> None:
    """Schema 4: only test users, whole subtypes, the remap's category, adoption near its share."""
    prefs = ds["truth_preferences"]
    users = ds["users"].set_index("user_id")["split"]
    test_users = users.index[users == "test"]
    split = prefs["user_id"].map(users)
    report.check(
        bool((split == "test").all()),
        f"preferences: {int((split != 'test').sum())} rows not for test users",
    )
    merchants = ds["truth_merchants"].set_index("merchant_id")
    rows = prefs.join(merchants[["category", "subtype"]], on="merchant_id", rsuffix="_true")
    remaps = {r.subtype: r for r in spec.preferences.remaps}
    expected = rows["subtype"].map({k: r.category for k, r in remaps.items()})
    report.check(
        bool((rows["category"] == expected).all()),
        "preferences: rows outside the spec's remaps, or with another category",
    )
    report.check(
        bool((rows["category"] != rows["category_true"]).all()),
        "preferences: a preference equal to the merchant's own category",
    )
    n = len(test_users)
    for subtype, remap in remaps.items():
        size = int((merchants["subtype"] == subtype).sum())
        held = rows[rows["subtype"] == subtype].groupby("user_id").size()
        report.check(
            bool((held == size).all()),
            f"preferences: users with only part of subtype {subtype}'s {size} merchants",
        )
        if n:
            share = len(held) / n
            report.stats[f"preference_share.{subtype}"] = share
            tolerance = PREFERENCE_SIGMAS * float(np.sqrt(remap.share * (1 - remap.share) / n))
            if n >= MIN_TEST_USERS_FOR_HOLDOUT_CHECK:
                report.check(
                    abs(share - remap.share) <= tolerance,
                    f"preferences: {share:.0%} of test users adopted {subtype}, "
                    f"expected {remap.share:.0%} ± {tolerance:.0%}",
                )
    if n:
        report.stats["test_users_with_preferences"] = prefs["user_id"].nunique() / n


def _goals(ds: Dataset, spec: Spec, report: Report) -> None:
    goals = ds["goals"]
    end = str(spec.calendar.end)
    report.check(bool((goals["as_of_date"] <= end).all()), "goals: as_of_date after the history")
    report.check(
        bool((goals["created_date"] <= goals["as_of_date"]).all()),
        "goals: as_of_date before created_date",
    )
    report.check(
        bool((goals["as_of_date"] < goals["target_date"]).all()),
        "goals: as_of_date not before target_date, so current_balance can reveal the outcome",
    )
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


def _label_consistency(truth: Truth, report: Report) -> None:
    tx = truth.transactions
    by_id = tx.set_index("transaction_id")
    unusual = tx["process"] == "unusual_charge"
    report.check(
        bool((tx["anomaly_kind"].notna() == unusual).all()),
        "labels: anomaly_kind set on a transaction that isn't an unusual charge, or missing",
    )

    def related(kind: pd.DataFrame) -> pd.DataFrame:
        orig = by_id.reindex(kind["related_transaction_id"])
        return orig.set_index(kind.index)

    dupes = tx[tx["anomaly_kind"] == "duplicate"]
    orig = related(dupes)
    minutes = (pd.to_datetime(dupes["ts"]) - pd.to_datetime(orig["ts"])).dt.total_seconds() / 60
    ok = (
        (orig["user_id"] == dupes["user_id"])
        & (orig["merchant_raw"] == dupes["merchant_raw"])
        & (orig["amount"] == dupes["amount"])
        & minutes.between(0, truth.contract.duplicate_window_minutes)
    )
    report.check(
        bool(ok.all()), f"labels: {int((~ok).sum())} duplicates don't match their original"
    )

    refunds = tx[tx["process"] == "refund"]
    orig = related(refunds)
    ok = (orig["user_id"] == refunds["user_id"]) & (orig["amount"] == -refunds["amount"])
    report.check(bool(ok.all()), f"labels: {int((~ok).sum())} refunds don't match their purchase")

    novel = tx[tx["anomaly_kind"] == "new_merchant_large"]
    visits = tx.groupby(["user_id", "merchant_id"]).size()
    repeat = visits.reindex(list(zip(novel["user_id"], novel["merchant_id"], strict=True))) > 1
    report.check(
        not bool(repeat.any()),
        f"labels: {int(repeat.sum())} new-merchant charges at merchants the user visits again",
    )

    normal = tx[tx["anomaly_kind"].isna() & (tx["amount"] < 0)]
    largest = (-normal["amount"]).groupby([normal["user_id"], normal["merchant_id"]]).max()
    outliers = tx[tx["anomaly_kind"] == "amount_outlier"]
    keys = list(zip(outliers["user_id"], outliers["merchant_id"], strict=True))
    above = (-outliers["amount"].to_numpy()) > largest.reindex(keys).fillna(0).to_numpy()
    report.check(
        bool(((outliers["tier"] == "clear").to_numpy() == above).all()),
        "labels: an amount outlier's tier disagrees with the user's normal charges",
    )
    report.check(
        bool((tx["tier"].notna() == tx["anomaly_kind"].notna()).all()),
        "labels: tier set on a transaction that isn't an unusual charge, or missing",
    )
    if len(outliers):
        share = float(above.mean())
        report.stats["amount_outlier_above_normal_share"] = share
        report.check(
            share >= MIN_OUTLIER_ABOVE_NORMAL_SHARE,
            f"labels: only {share:.0%} of amount outliers exceed the user's normal charges",
        )


def _spike_integrity(truth: Truth, spec: Spec, report: Report) -> None:
    spikes = truth.periods
    if spikes.empty:
        return
    start = pd.to_datetime(spikes["period_start"])
    end = pd.to_datetime(spikes["period_end"])
    week = spikes["granularity"] == "week"
    month_end = start + pd.offsets.MonthEnd(0)
    shape = np.where(
        week,
        (start.dt.dayofweek == 0) & (end - start == pd.Timedelta(days=6)),
        (start.dt.day == 1) & (end == month_end),
    )
    report.check(bool(shape.all()), "spikes: a period doesn't start on a Monday or the 1st")
    inside = (spikes["period_start"] >= truth.warmup_end_month) & (
        spikes["period_end"] <= str(spec.calendar.end)
    )
    report.check(bool(inside.all()), "spikes: a period is in the warm-up or after the calendar")

    ordered = spikes.assign(start=start, end=end).sort_values(["user_id", "category", "start"])
    same = (ordered[["user_id", "category"]] == ordered[["user_id", "category"]].shift()).all(
        axis=1
    )
    gap = (ordered["start"] - ordered["end"].shift()).dt.days
    report.check(
        not bool((same & (gap < MIN_SPIKE_GAP_DAYS)).any()),
        "spikes: two spikes in one category overlap or are within 7 days",
    )

    users = truth.users.set_index("user_id")["persona"]
    peaks: dict[tuple[str, str], np.ndarray] = {}
    for name, persona in spec.personas.items():
        for stream in persona.discretionary:
            key = (name, stream.category)
            peaks[key] = np.maximum(peaks.get(key, np.zeros(13)), season_vector(stream.seasonality))
    peak = [
        peaks.get((users[u], c), np.ones(13))[m]
        for u, c, m in zip(spikes["user_id"], spikes["category"], start.dt.month, strict=True)
    ]
    report.check(
        max(peak) <= spec.events.spending_spike.peak_threshold,
        "spikes: a spike falls in a seasonal peak of its category",
    )

    tx = truth.transactions[["user_id", "category", "ts", "amount"]]
    joined = spikes[["spike_id", "user_id", "category", "period_start", "period_end"]].merge(
        tx, on=["user_id", "category"]
    )
    day = joined["ts"].str[:10]
    joined = joined[(day >= joined["period_start"]) & (day <= joined["period_end"])]
    realized = (-joined.groupby("spike_id")["amount"].sum()).reindex(spikes["spike_id"]).fillna(0)
    recorded = (spikes["base_spend"] + spikes["extra_spend"]).to_numpy()
    off = np.abs(realized.to_numpy() - recorded) > 0.015
    report.check(
        not bool(off.any()), f"spikes: {int(off.sum())} spikes' spend doesn't match ledger"
    )


def _label_ceiling(truth: Truth, report: Report) -> None:
    c = truth.contract
    stats = truth.report()
    report.stats.update(stats)
    months = truth.spikes("month")
    if len(months) >= c.min_labels_for_oracle_check:
        weak = stats["monthly_weak_share"]
        report.check(
            weak <= c.max_weak_share,
            f"labels: {weak:.0%} of monthly spikes are weak, above {c.max_weak_share:.0%}",
        )
        precision = stats["oracle_month_precision"]
        report.check(
            precision >= c.oracle_min_precision,
            f"ceiling: monthly spike oracle precision {precision:.2f} "
            f"below {c.oracle_min_precision:.2f}",
        )
    if truth.transactions["anomaly_kind"].notna().sum() >= c.min_labels_for_oracle_check:
        precision = stats["oracle_transaction_precision"]
        report.check(
            precision >= c.oracle_min_precision,
            f"ceiling: unusual-charge oracle precision {precision:.2f} "
            f"below {c.oracle_min_precision:.2f}",
        )


def validate(ds: Dataset, spec: Spec) -> Report:
    report = Report()
    if (built_from := ds.meta.get("spec_hash")) != spec_hash(spec):
        # Checking against another spec's tolerances would be meaningless
        report.errors.append(f"spec: dataset was built from spec {built_from}, not this spec")
        return report
    _schema(ds, report)
    if not report.ok:
        return report
    tx = _ledger(ds)
    report.stats["users"] = len(ds["users"])
    report.stats["transactions"] = len(tx)
    _coverage(tx, ds, spec, report)
    _plausibility(tx, ds, spec, report)
    _seasonality(tx, ds, spec, report)
    _mess(tx, ds, spec, report)
    _events(tx, ds, spec, report)
    _splits(tx, ds, spec, report)
    _preferences(ds, spec, report)
    _goals(ds, spec, report)
    truth = Truth.from_dataset(ds)
    _label_consistency(truth, report)
    _spike_integrity(truth, spec, report)
    _label_ceiling(truth, report)
    report.stats["unusual_charges_dropped"] = int(ds.meta.get("unusual_charges_dropped", 0))
    return report
