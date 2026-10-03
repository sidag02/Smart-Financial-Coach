"""FR-2 label contract: how flags are scored against planted ground truth, and the oracle ceiling.

This module reads `truth_*` tables. Model code under `intelligence/` must never import it
(`tests/unit/test_truth_isolation.py` enforces that); the evaluation harness and `validate` do.

    truth = load_truth("data/synthetic/default.sqlite")
    truth.score_periods(flags)  # one row per flag or missed label: tp / fp / fn / ignored
    truth.score_transactions(flags)
    truth.user_categories()  # label contract 2: each transaction's category as its user sees it
"""

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from smart_financial_coach.data.generator.dataset import Dataset
from smart_financial_coach.data.generator.sqlite_io import read_sqlite

# Reason code a flag must carry for each planted kind (reason accuracy, NFR-7)
REASON_CODES = {
    "duplicate": "duplicate",
    "amount_outlier": "amount_unusual",
    "new_merchant_large": "new_merchant",
}
PERIOD_KEY = ["user_id", "category", "period_start"]
# Processes whose purchase counts are Poisson; recurring bills have scheduled counts
POISSON_PROCESSES = ("discretionary", "one_off")
INCOME = "Income"
MIN_ORACLE_SIGMA = 0.1  # log-amount spread floor for the transaction oracle


@dataclass(frozen=True)
class Contract:
    """Contract parameters, stored in `meta.label_contract` by the generator."""

    weak_lift: float
    max_weak_share: float
    max_drivers: int
    duplicate_window_minutes: int
    oracle_recall: float
    oracle_min_precision: float
    min_labels_for_oracle_check: int
    baseline_days: int
    baseline_months: int

    @classmethod
    def from_meta(cls, meta: dict[str, str]) -> "Contract":
        if "label_contract" not in meta:
            raise ValueError("dataset has no label contract; regenerate it with this generator")
        return cls(**json.loads(meta["label_contract"]))


def precision_at_recall(score: pd.Series, label: pd.Series, recall: float) -> float:
    """Precision at the first rank where recall reaches `recall`; NaN without positives."""
    order = np.argsort(-score.to_numpy(dtype=np.float64), kind="mergesort")
    hits = label.to_numpy(dtype=bool)[order]
    if not hits.any():
        return float("nan")
    tp = np.cumsum(hits)
    at = int(np.argmax(tp >= recall * hits.sum()))
    return float(tp[at] / (at + 1))


def metrics(outcomes: pd.DataFrame) -> dict[str, float]:
    """Precision and recall from scored outcomes; `recall_clear` and reason accuracy if present."""
    counts = outcomes["outcome"].value_counts()
    tp, fp, fn = (int(counts.get(k, 0)) for k in ("tp", "fp", "fn"))
    result = {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "ignored": float(counts.get("ignored", 0)),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
    }
    if "tier" in outcomes:
        clear = outcomes[outcomes["tier"] == "clear"]["outcome"]
        tp_c, fn_c = int((clear == "tp").sum()), int((clear == "fn").sum())
        result["recall_clear"] = tp_c / (tp_c + fn_c) if tp_c + fn_c else float("nan")
    if "reason_match" in outcomes:
        matched = outcomes.loc[outcomes["outcome"] == "tp", "reason_match"].dropna()
        result["reason_accuracy"] = float(matched.mean()) if len(matched) else float("nan")
    return result


def _month_start(ts: pd.Series) -> pd.Series:
    return ts.str[:7] + "-01"


def _check_months(df: pd.DataFrame) -> None:
    """Reject period starts that aren't the 1st of a month: a harness bug, not a false positive."""
    bad = ~df["period_start"].astype(str).str.fullmatch(r"\d{4}-\d{2}-01")
    if bad.any():
        example = df.loc[bad, "period_start"].iloc[0]
        raise ValueError(f"{int(bad.sum())} periods don't start on the 1st (e.g. {example!r})")


def _in_keys(df: pd.DataFrame, keys: set[tuple[str, str, str]]) -> pd.Series:
    """Whether each row's (user_id, category, period_start) is in `keys`."""
    rows = zip(*(df[c] for c in PERIOD_KEY), strict=True)
    return pd.Series([k in keys for k in rows], index=df.index, dtype=bool)


@dataclass(frozen=True)
class Truth:
    """Ground truth for one dataset, with the contract that scores flags against it."""

    transactions: pd.DataFrame  # model-visible columns + truth columns + `month`
    periods: pd.DataFrame
    expected: pd.DataFrame
    merchants: pd.DataFrame
    users: pd.DataFrame
    preferences: pd.DataFrame  # truth_preferences (schema 4): user, merchant, the user's category
    contract: Contract
    calendar_start: date

    @classmethod
    def from_dataset(cls, ds: Dataset) -> "Truth":
        tx = ds["transactions"].merge(
            ds["truth_transactions"], on="transaction_id", validate="one_to_one"
        )
        tx["month"] = _month_start(tx["ts"])
        return cls(
            transactions=tx,
            periods=ds["truth_periods"],
            expected=ds["truth_expected"],
            merchants=ds["truth_merchants"],
            users=ds["users"],
            preferences=ds["truth_preferences"],
            contract=Contract.from_meta(ds.meta),
            calendar_start=date.fromisoformat(ds.meta["calendar_start"]),
        )

    # --- The user's category (label contract 2) -------------------------------------------

    def user_categories(self) -> pd.Series:
        """Each transaction's category as its user sees it: the user's preference for that
        merchant where there is one, otherwise the true category. Aligned with `transactions`.

        For FR-5/FR-6 measures only (personal accuracy, global gain against users' own view).
        FR-4 and every other gate score against the true category.
        """
        tx = self.transactions
        preferred = self.preferences.set_index(["user_id", "merchant_id"])["category"]
        keys = pd.MultiIndex.from_frame(tx[["user_id", "merchant_id"]])
        mine = preferred.reindex(keys).to_numpy()
        values = np.where(pd.isna(mine), tx["category"].to_numpy(), mine)
        return pd.Series(values, index=tx.index, name="user_category")

    # --- Warm-up -------------------------------------------------------------------------------

    @property
    def warmup_end_day(self) -> str:
        """Transactions before this date are in the warm-up (nothing is planted there)."""
        return str(self.calendar_start + timedelta(days=self.contract.baseline_days))

    @property
    def warmup_end_month(self) -> str:
        """Periods starting before this month are in the warm-up."""
        months = self.calendar_start.year * 12 + self.calendar_start.month - 1
        months += self.contract.baseline_months
        return f"{months // 12:04d}-{months % 12 + 1:02d}-01"

    # --- Labels --------------------------------------------------------------------------------

    def spikes(self, granularity: str = "month", tier: str | None = None) -> pd.DataFrame:
        p = self.periods[self.periods["granularity"] == granularity]
        return p if tier is None else p[p["tier"] == tier]

    def _ignored_transaction_ids(self) -> set[str]:
        tx = self.transactions
        originals = tx.loc[tx["anomaly_kind"] == "duplicate", "related_transaction_id"]
        warmup = tx.loc[tx["ts"] < self.warmup_end_day, "transaction_id"]
        return set(originals.dropna()) | set(warmup)

    def _ignored_months(self) -> set[tuple[str, str, str]]:
        """Months not scored: with a planted weekly spike or unusual charge in the category."""
        weeks = self.spikes("week")
        keys = set(
            zip(
                weeks["user_id"],
                weeks["category"],
                _month_start(weeks["period_start"]),
                strict=True,
            )
        )
        tx = self.transactions
        unusual = tx[tx["process"] == "unusual_charge"]
        keys |= set(zip(unusual["user_id"], unusual["category"], unusual["month"], strict=True))
        return keys

    # --- Scoring -------------------------------------------------------------------------------

    def score_transactions(self, flags: pd.DataFrame) -> pd.DataFrame:
        """Score transaction flags (FR-7).

        `flags` has `transaction_id` and optionally `reason_code`. Returns one row per flagged
        transaction and one `fn` row per missed label, with `outcome`, the label's `tier` and
        `reason_match`.
        """
        tx = self.transactions[["transaction_id", "anomaly_kind", "tier"]]
        flags = flags.drop_duplicates("transaction_id")
        if unknown := set(flags["transaction_id"]) - set(tx["transaction_id"]):
            raise ValueError(f"{len(unknown)} flagged transactions are not in the dataset")
        ignored = self._ignored_transaction_ids()
        scored = flags.merge(tx, on="transaction_id", how="left")
        is_ignored = scored["transaction_id"].isin(ignored)
        scored["outcome"] = np.select(
            [scored["anomaly_kind"].notna(), is_ignored], ["tp", "ignored"], "fp"
        )
        if "reason_code" in scored:
            expected = scored["anomaly_kind"].map(REASON_CODES)
            scored["reason_match"] = (scored["reason_code"] == expected).where(
                scored["outcome"] == "tp"
            )
        missed = tx[
            tx["anomaly_kind"].notna()
            & ~tx["transaction_id"].isin(flags["transaction_id"])
            & ~tx["transaction_id"].isin(ignored)
        ].assign(outcome="fn")
        return pd.concat([scored, missed], ignore_index=True)

    def score_periods(self, flags: pd.DataFrame, granularity: str = "month") -> pd.DataFrame:
        """Score spending-spike flags (FR-8) on true categories.

        `flags` has `user_id`, `category` and `period_start` (first of the month). Returns one row
        per flagged period and one `fn` row per missed label, with `outcome` and `tier`.
        """
        if granularity != "month":
            raise ValueError("v1 scores monthly spikes only (FR-2 option F)")
        _check_months(flags)
        labels = self.spikes("month")[[*PERIOD_KEY, "tier"]]
        flags = flags[PERIOD_KEY].drop_duplicates()
        scored = flags.merge(labels, on=PERIOD_KEY, how="left")
        is_ignored = _in_keys(scored, self._ignored_months()) | (
            scored["period_start"] < self.warmup_end_month
        )
        scored["outcome"] = np.select([scored["tier"].notna(), is_ignored], ["tp", "ignored"], "fp")
        missed = labels.merge(flags, on=PERIOD_KEY, how="left", indicator=True)
        missed = missed[missed["_merge"] == "left_only"].drop(columns="_merge").assign(outcome="fn")
        return pd.concat([scored, missed], ignore_index=True)

    def _period_spend(self) -> pd.DataFrame:
        """Realized net outflow per (user, category, month), every transaction counted."""
        tx = self.transactions[self.transactions["category"] != INCOME]
        spend = -tx.groupby(["user_id", "category", "month"])["amount"].sum()
        return spend.rename("spend").reset_index().rename(columns={"month": "period_start"})

    def score_drivers(self, drivers: pd.DataFrame) -> pd.DataFrame:
        """Score driving transactions for flagged monthly spikes (FR-8).

        `drivers` has `user_id`, `category`, `period_start` and `transaction_id`. Returns one row
        per period: `valid` (at most `max_drivers` distinct transactions, all inside the period
        and category) and `coverage` (returned spend over the excess above expected spend, capped
        at 1). An invalid set covers nothing, so returning everything can't score well.
        """
        _check_months(drivers)
        drivers = drivers[[*PERIOD_KEY, "transaction_id"]].drop_duplicates()
        tx = self.transactions[["transaction_id", "user_id", "category", "month", "amount"]]
        d = drivers.merge(tx, on="transaction_id", how="left", suffixes=("", "_tx"))
        d["inside"] = (
            (d["user_id"] == d["user_id_tx"])
            & (d["category"] == d["category_tx"])
            & (d["period_start"] == d["month"])
        )
        d["outflow"] = (-d["amount"]).where(d["inside"], 0.0)
        per = (
            d.groupby(PERIOD_KEY)
            .agg(
                n=("transaction_id", "size"), inside=("inside", "all"), returned=("outflow", "sum")
            )
            .reset_index()
        )
        monthly = self.expected[self.expected["granularity"] == "month"]
        per = per.merge(monthly[[*PERIOD_KEY, "expected_spend"]], on=PERIOD_KEY, how="left")
        per = per.merge(self._period_spend(), on=PERIOD_KEY, how="left")
        excess = per["spend"] - per["expected_spend"]
        per["valid"] = per["inside"] & (per["n"] <= self.contract.max_drivers)
        coverage = (per["returned"] / excess).clip(upper=1.0).where(excess > 0)
        per["coverage"] = coverage.where(per["valid"] | coverage.isna(), 0.0)
        return per

    def baseline_drivers(self) -> pd.DataFrame:
        """The `max_drivers` largest outflows in each labeled monthly spike: no detection skill."""
        tx = self.transactions[
            (self.transactions["amount"] < 0) & (self.transactions["category"] != INCOME)
        ]
        spikes = self.spikes("month")[PERIOD_KEY]
        inside = tx.rename(columns={"month": "period_start"}).merge(spikes, on=PERIOD_KEY)
        top = inside.sort_values(["amount", "transaction_id"], kind="mergesort").groupby(PERIOD_KEY)
        return top.head(self.contract.max_drivers)[[*PERIOD_KEY, "transaction_id"]]

    # --- Oracle ceiling ------------------------------------------------------------------------

    def period_oracle(self) -> pd.DataFrame:
        """Monthly spike oracle: Poisson upper tail of the realized purchase count.

        Uses the generator's true expected count, so it is a ceiling, not a model. Ignored and
        warm-up periods are dropped, as `score_periods` would.
        """
        e = self.expected[self.expected["granularity"] == "month"]
        tx = self.transactions[self.transactions["process"].isin(POISSON_PROCESSES)]
        counts = (
            tx.groupby(["user_id", "category", "month"])
            .size()
            .rename("count")
            .reset_index()
            .rename(columns={"month": "period_start"})
        )
        o = e.merge(counts, on=PERIOD_KEY, how="left").fillna({"count": 0})
        o["score"] = -poisson.logsf(o["count"] - 1, o["expected_count"])
        labels = self.spikes("month")[PERIOD_KEY].assign(is_spike=True)
        o = o.merge(labels, on=PERIOD_KEY, how="left").fillna({"is_spike": False})
        o["is_spike"] = o["is_spike"].astype(bool)
        keep = o["is_spike"] | ~_in_keys(o, self._ignored_months())
        o = o[keep & (o["period_start"] >= self.warmup_end_month)]
        return o[[*PERIOD_KEY, "expected_count", "count", "score", "is_spike"]]

    def transaction_oracle(self) -> pd.DataFrame:
        """Unusual-charge oracle: a perfect duplicate check, then distance from normal amounts.

        A transaction's reference is the user's *normal* (unlabeled) charges at the same merchant,
        leaving the transaction itself out; with no such history, the merchant's catalog price.
        Score is +inf for an exact repeat (same text and amount) within the duplicate window,
        otherwise the z-score of the log amount against the reference.
        """
        tx = self.transactions
        tx = tx[(tx["amount"] < 0) & (tx["category"] != INCOME)].copy()
        tx["x"] = np.log(-tx["amount"])
        normal = tx["anomaly_kind"].isna()

        stats = (
            tx[normal]
            .assign(x2=lambda d: d["x"] ** 2)
            .groupby(["user_id", "merchant_id"])
            .agg(n=("x", "size"), s1=("x", "sum"), s2=("x2", "sum"))
            .reset_index()
        )
        tx = tx.merge(stats, on=["user_id", "merchant_id"], how="left")
        tx = tx.fillna({"n": 0, "s1": 0.0, "s2": 0.0})
        own = tx["anomaly_kind"].isna()  # leave the transaction itself out of its reference
        n = tx["n"] - own
        s1 = tx["s1"] - tx["x"].where(own, 0.0)
        s2 = tx["s2"] - (tx["x"] ** 2).where(own, 0.0)
        prices = self.merchants.set_index("merchant_id")
        cat_mu = np.log(tx["merchant_id"].map(prices["price_median"]).clip(lower=0.01))
        cat_sigma = tx["merchant_id"].map(prices["price_sigma"])
        safe_n = n.clip(lower=1)
        mean = (s1 / safe_n).where(n >= 1, cat_mu)
        sample_sd = np.sqrt((s2 / safe_n - (s1 / safe_n) ** 2).clip(lower=0)).where(n >= 2, 0.0)
        sd = np.maximum(np.maximum(sample_sd, cat_sigma), MIN_ORACLE_SIGMA)
        z = (tx["x"] - mean) / sd

        ts = pd.to_datetime(tx["ts"])
        order = tx.assign(t=ts).sort_values(
            ["user_id", "merchant_raw", "amount", "t"], kind="mergesort"
        )
        same = (
            order[["user_id", "merchant_raw", "amount"]]
            == order[["user_id", "merchant_raw", "amount"]].shift()
        ).all(axis=1)
        gap = order["t"].diff().dt.total_seconds() / 60
        repeat = (same & (gap <= self.contract.duplicate_window_minutes)).reindex(tx.index)

        tx["score"] = z.where(~repeat, np.inf)
        tx["is_anomaly"] = tx["anomaly_kind"].notna()
        ignored = self._ignored_transaction_ids()
        tx = tx[tx["is_anomaly"] | ~tx["transaction_id"].isin(ignored)]
        return tx[["transaction_id", "user_id", "score", "is_anomaly", "anomaly_kind"]]

    # --- Report --------------------------------------------------------------------------------

    def ambiguous_error_share(self) -> float:
        """Share of spending transactions no categorizer can get right from text and amount."""
        tx = self.transactions
        spend = tx[(tx["amount"] < 0) & (tx["category"] != INCOME)]
        ambiguous = set(self.merchants.loc[self.merchants["is_ambiguous"] == 1, "merchant_id"])
        a = spend[spend["merchant_id"].isin(ambiguous)]
        if spend.empty or a.empty:
            return 0.0
        per = a.groupby(["merchant_id", "category"]).size()
        majority = per.groupby(level=0).max()
        return float((per.groupby(level=0).sum() - majority).sum() / len(spend))

    def report(self) -> dict[str, float]:
        """Label counts, tiers and ceilings for `sfc-data labels` and `validate`."""
        c = self.contract
        tx = self.transactions
        months = self.spikes("month")
        out: dict[str, float] = {}
        for kind, n in tx["anomaly_kind"].value_counts().sort_index().items():
            out[f"unusual_{kind}"] = float(n)
        out["unusual_weak"] = float((tx["tier"] == "weak").sum())
        tiers = self.periods.groupby(["granularity", "tier"]).size()
        for granularity, tier, n in zip(
            tiers.index.get_level_values(0), tiers.index.get_level_values(1), tiers, strict=True
        ):
            out[f"spikes_{granularity}_{tier}"] = float(n)
        out["monthly_weak_share"] = (
            float((months["tier"] == "weak").mean()) if len(months) else float("nan")
        )
        po = self.period_oracle()
        out["oracle_month_precision"] = precision_at_recall(
            po["score"], po["is_spike"], c.oracle_recall
        )
        to = self.transaction_oracle()
        out["oracle_transaction_precision"] = precision_at_recall(
            to["score"], to["is_anomaly"], c.oracle_recall
        )
        baseline = self.score_drivers(self.baseline_drivers())
        out["driver_baseline_coverage"] = float(baseline["coverage"].mean())
        out["ambiguous_error_share"] = self.ambiguous_error_share()
        return out


def load_truth(path: str | Path) -> Truth:
    """Ground truth from a generated SQLite file. Evaluation and validation code only."""
    return Truth.from_dataset(read_sqlite(path))
