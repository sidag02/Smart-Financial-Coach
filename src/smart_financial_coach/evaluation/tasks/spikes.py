"""The spending-spikes task (FR-8 §4-§6): examples, splits, metrics, ties and gates.

Examples are every (user, spending category, month) after the warm-up, on **true** categories, so
categorizer errors can't leak into FR-8's metrics. Labels come from FR-2's label contract
(`data.labels`): `spike` for a planted monthly spike, `ignored` for what the contract ignores
(months with a planted weekly spike or unusual charge in the category), `normal` otherwise. Models
get them only through `fit`, where scorers never see them and `SpikeThresholded` uses them to place
its cutoff.

Simulated basket-size spikes (decision 1, the main risk) are extra examples labelled `basket`:
copies of normal months in the categories where spikes are planted, with spend multiplied by
U(1.8, 3.0) and the purchase count unchanged. They're never trained on, never count towards a rate
or precision, and feed one reported diagnostic: the share a run flags.

Splits:
- `train`: train users' periods; validation is 5 folds of whole train users, by persona.
- `test`: test users' periods, scored once by `finalize`.

Season profiles come from two pools (§2): train users' rows carry profiles from train users only,
so test users never shape a validation score; test users' rows carry profiles from everyone.

Runs are ranked at a common operating point (§4): out-of-fold recall when each held-out fold flags
its top `FLAG_RATE` eligible periods per user-month. Periods the contract ignores stay in that
budget, as a deployed cutoff can't know them (FR-7's rule, review on #34). Each run's precision at
its own tuned cutoff is reported beside it, and gates eligibility and promotion.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.data.labels import PERIOD_KEY, Truth, load_truth
from smart_financial_coach.data.labels import metrics as outcome_metrics
from smart_financial_coach.data.store import load_meta, load_transactions, load_users
from smart_financial_coach.evaluation.metrics.classification import interval
from smart_financial_coach.evaluation.splits import (
    TRAIN,
    Ids,
    Splits,
    group_kfold,
    ids,
    leak_errors,
)
from smart_financial_coach.evaluation.tasks.base import Examples, Gate, register_task
from smart_financial_coach.evaluation.tasks.categorization import content_hash
from smart_financial_coach.intelligence.anomaly.threshold import top_k
from smart_financial_coach.intelligence.models.contract import Checked
from smart_financial_coach.intelligence.spikes.contract import (
    INPUT_COLUMNS,
    SpikeModel,
    scoring_periods,
)
from smart_financial_coach.intelligence.spikes.threshold import (
    BASKET,
    IGNORED,
    NORMAL,
    SPIKE,
    user_months,
)

TEST = "test"
FLAG_RATE = 0.035  # the common operating point (§4), the POC leader's 0.80 point
PRECISION_GATE = 0.70  # the PRD's target, on test users (§5)
BASELINE = "mean_k_std"  # the run name every candidate must beat at FLAG_RATE
BOOTSTRAP_REPS = 1000
DEFAULTS: dict[str, Any] = {"k": 5}
BASKET_RANGE = (1.8, 3.0)  # the planted multipliers, applied to spend instead of the rate
BASKET_ROWS = {"train": 400, "test": 200}
BASKET_SEED = 0
MAX_DRIVERS = 5
COUNT_BANDS = ((0, 3.999, "under 4"), (4, 15.999, "4-15"), (16, 10**9, "16+"))
HISTORY_BANDS = ((0, 11, "< 12 months"), (12, 23, "12-23"), (24, 10**9, "24+"))
EVAL_COLUMNS = ["user_id", "category", "period_start", "label", "tier", "month", "split"]


def labels_for(truth: Truth, periods: pd.DataFrame) -> pd.DataFrame:
    """`label` (`spike`, `ignored` or `normal`) and its `tier` per period, by the contract."""
    scored = truth.score_periods(periods[PERIOD_KEY])
    scored = scored[scored["outcome"] != "fn"].drop_duplicates(PERIOD_KEY)
    out = periods[PERIOD_KEY].merge(scored, on=PERIOD_KEY, how="left", validate="one_to_one")
    label = out["outcome"].map({"tp": SPIKE, "ignored": IGNORED, "fp": NORMAL})
    return pd.DataFrame({"label": label.to_numpy(), "tier": out["tier"].to_numpy()})


def basket_rows(frame: pd.DataFrame, categories: set[str], n: int, seed: int) -> pd.DataFrame:
    """Simulated basket-size spikes: `n` normal months in `categories` with some spend and at
    least 2 purchases in a usual month, spend times U(BASKET_RANGE), counts unchanged (FR-8 §5)."""
    pool = frame[
        (frame["label"] == NORMAL)
        & frame["category"].isin(categories)
        & (frame["usual_count"] >= 2)
        & (frame["spend"] > 0)  # a month with nothing bought has no basket to grow
    ]
    if pool.empty:
        return pool.iloc[:0]
    rng = np.random.default_rng(seed)
    picked = pool.loc[rng.choice(pool.index, min(n, len(pool)), replace=False)].copy()
    picked["spend"] = picked["spend"] * rng.uniform(*BASKET_RANGE, len(picked))
    picked["period_id"] = picked["period_id"] + "|basket"
    return picked.assign(label=BASKET, tier=None)


class SpendingSpikesTask:
    """Metric choices and the alternatives considered: FR-8 design, "Metrics and why"."""

    name = "spending_spikes"
    selection_metric = "recall_at_rate"
    tuning_metric = "recall_at_rate"  # grid points are chosen at the same operating point
    # Among tied runs (lower is better): the own cutoff's shortfall from precision 1, then the
    # negative binomial before the Poisson (decision 6), then batch cost
    tiebreak_metrics: tuple[str, ...] = (
        "val_cutoff_shortfall",
        "fit.assumes_poisson",
        "latency_batch_ms",
    )
    shipping_params: tuple[str, ...] = ()  # scorers don't train on labels: no shipping twins
    serving_files_required: tuple[str, ...] = ()
    report_metrics: tuple[str, ...] = (
        "val_precision_at_rate",
        "val_precision",
        "val_recall",
        "val_flag_rate",
        "val_recall_clear",
        "val_average_precision",
        "val_basket_recall_at_rate",
        "val_excess_coverage",
        "val_usual_error",
        "latency_batch_ms",
    )
    test_report_metrics: tuple[str, ...] = (
        "test_precision",
        "test_precision_lo",
        "test_precision_hi",
        "test_recall",
        "test_recall_at_rate",
        "test_recall_at_rate_lo",
        "test_recall_at_rate_hi",
        "test_flag_rate",
        "test_recall_clear",
        "test_basket_recall",
    )
    required_baselines: tuple[str, ...] = (BASELINE,)
    bootstrap_unit = "user"
    diagnostics_scope = "at its own cutoff"

    def __init__(self, reps: int = BOOTSTRAP_REPS) -> None:
        self.reps = reps
        self.truth: Truth | None = None
        self.train_transactions: pd.DataFrame | None = None
        self.as_of = ""

    # --- Data and splits -------------------------------------------------------------------

    def load(self, data: Path) -> Examples:
        meta = load_meta(data)
        truth = load_truth(data)
        self.truth = truth
        users = load_users(data)[["user_id", "split", "persona"]]
        txns = load_transactions(data)
        category = truth.transactions.set_index("transaction_id")["category"]
        txns["category"] = txns["transaction_id"].map(category).to_numpy()
        split = txns["user_id"].map(users.set_index("user_id")["split"])
        train = txns[split == "train"]
        self.train_transactions = train
        # The dataset's last day ends a month, so every month in it is complete (decision 10)
        self.as_of = meta["calendar_end"]
        parts = [
            scoring_periods(train, train, as_of=self.as_of).assign(season_pool="train"),
            scoring_periods(txns[split == "test"], txns, as_of=self.as_of).assign(
                season_pool="all"
            ),
        ]
        frame = pd.concat(parts, ignore_index=True)
        frame = frame[frame["period_start"] >= truth.warmup_end_month].reset_index(drop=True)
        frame = frame.merge(users, on="user_id", how="left")
        frame = pd.concat([frame, labels_for(truth, frame)], axis=1)
        planted = set(truth.spikes("month")["category"])
        extra = [
            basket_rows(frame[frame["split"] == s], planted, n, BASKET_SEED + i)
            for i, (s, n) in enumerate(BASKET_ROWS.items())
        ]
        frame = pd.concat([frame, *extra], ignore_index=True)
        frame["first_month"] = frame.groupby("user_id")["month"].transform("min")
        return Examples(
            frame=frame,
            labels=frame["label"],
            id_column="period_id",
            model_columns=INPUT_COLUMNS,
            data_hash=content_hash(frame[list(INPUT_COLUMNS)], meta),
        )

    def split(self, examples: Examples, params: Mapping[str, Any], seed: int) -> Splits:
        p = DEFAULTS | dict(params)
        f = examples.frame
        train = f[f["split"] == "train"]
        folds = group_kfold(
            train,
            group="user_id",
            k=int(p["k"]),
            seed=seed,
            id_column="period_id",
            stratify="persona",
        )
        return Splits(
            sets={
                TRAIN: ids(train["period_id"]),
                TEST: ids(f.loc[f["split"] == "test", "period_id"]),
            },
            folds=folds,
        )

    def leak_errors(self, examples: Examples, splits: Splits) -> list[str]:
        f = examples.frame.set_index("period_id")
        errors = leak_errors(splits, examples.frame, id_column="period_id", group="user_id")
        for name, split, pool in ((TRAIN, "train", "train"), (TEST, "test", "all")):
            rows = f.loc[splits.sets[name].tolist()]
            if (rows["split"] != split).any():
                errors.append(
                    f"{name} has periods of {'test' if split == 'train' else 'train'} users"
                )
            if (rows["season_pool"] != pool).any():
                errors.append(
                    f"{name} rows carry season profiles from the wrong pool (want {pool})"
                )
        if shared := set(f.loc[splits.sets[TRAIN].tolist(), "user_id"]) & set(
            f.loc[splits.sets[TEST].tolist(), "user_id"]
        ):
            errors.append(f"users in both train and test: {sorted(shared)[:5]}")
        if forbidden := {"persona", "label", "tier"} & set(examples.model_columns):
            errors.append(f"model columns include {sorted(forbidden)}")
        errors += self._validation_profile_errors(examples, splits)
        return errors

    def _validation_profile_errors(self, examples: Examples, splits: Splits) -> list[str]:
        """Checked from the data, not the `season_pool` tag: rebuilding the train rows' profiles
        from train users' transactions alone must give exactly the loaded values, so no test user
        shaped a validation score. Leave-one-out and as-of are `season_profiles`' own tests."""
        if self.train_transactions is None:
            return ["load() first"]
        f = examples.frame
        train = f[f["period_id"].isin(set(splits.sets[TRAIN])) & (f["label"] != BASKET)]
        rebuilt = scoring_periods(
            self.train_transactions, self.train_transactions, as_of=self.as_of
        )
        rebuilt = rebuilt.set_index("period_id").reindex(train["period_id"])
        for column in ("profile_season", "profile_users"):
            if not np.allclose(
                train[column].to_numpy(dtype=float),
                rebuilt[column].to_numpy(dtype=float),
                equal_nan=True,
            ):
                return [f"validation {column} isn't built from train users alone"]
        return []

    def training_rows(
        self, examples: Examples, which: Ids, params: Mapping[str, Any], seed: int
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Simulated basket rows are scored, never trained on."""
        labels = examples.labels_for(which)
        assert labels is not None
        keep = (labels != BASKET).to_numpy()
        return examples.rows(which)[keep].reset_index(drop=True), labels[keep].reset_index(
            drop=True
        )

    # --- Metrics ---------------------------------------------------------------------------

    def _rows(self, examples: Examples, predictions: pd.DataFrame) -> pd.DataFrame:
        """Predictions with the evaluation columns they're scored on."""
        f = examples.frame.set_index("period_id")
        cols = [*EVAL_COLUMNS, "usual", "usual_count", "spend", "persona", "first_month"]
        joined = f.loc[predictions["period_id"].tolist(), cols].reset_index(drop=True)
        fold = (
            predictions["fold"] if "fold" in predictions else pd.Series(0, index=predictions.index)
        )
        evidence = (
            predictions["evidence"]
            if "evidence" in predictions
            else pd.Series(None, index=predictions.index, dtype=object)
        )
        out = pd.concat(
            [
                predictions[["period_id", "score", "is_flagged"]].reset_index(drop=True),
                evidence.reset_index(drop=True).rename("evidence"),
                joined,
                fold.reset_index(drop=True).rename("fold"),
            ],
            axis=1,
        )
        out["eligible"] = SpikeModel.eligible(out)
        out["history_months"] = out["month"] - out["first_month"]
        return out

    def _at_rate(self, rows: pd.DataFrame) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Each fold flags its top `FLAG_RATE` eligible periods per user-month. Returns those
        flags, and the basket rows at or above each fold's resulting cutoff."""
        flagged = np.zeros(len(rows), dtype=bool)
        basket = np.zeros(len(rows), dtype=bool)
        for _, idx in rows.groupby("fold").indices.items():
            part = rows.iloc[idx]
            counted = (part["label"] != BASKET).to_numpy()
            k = round(FLAG_RATE * user_months(part[counted]))
            score = part["score"].to_numpy(dtype=float)
            usable = counted & part["eligible"].to_numpy() & np.isfinite(score)
            chosen = top_k(np.where(usable, score, -np.inf), part["period_id"].to_numpy(), k)
            chosen &= usable
            flagged[idx] = chosen
            if chosen.any():
                cutoff = score[chosen].min()
                basket[idx] = ~counted & part["eligible"].to_numpy() & (score >= cutoff)
        return flagged, basket

    def _outcomes(self, rows: pd.DataFrame, flagged: npt.NDArray[np.bool_]) -> pd.DataFrame:
        """The label contract's outcome per flag and per missed label, for these rows' users."""
        assert self.truth is not None, "load() first"
        real = (rows["label"] != BASKET).to_numpy()
        outcomes = self.truth.score_periods(rows.loc[flagged & real, PERIOD_KEY])
        return outcomes[outcomes["user_id"].isin(set(rows["user_id"]))]

    def _drivers(self, keys: pd.DataFrame) -> pd.DataFrame:
        """The 5 largest outflows in each period and true category (FR-8 §7)."""
        assert self.truth is not None
        tx = self.truth.transactions
        tx = tx[tx["amount"] < 0].rename(columns={"month": "period_start"})
        inside = tx.merge(keys[PERIOD_KEY].drop_duplicates(), on=PERIOD_KEY)
        top = inside.sort_values(["amount", "transaction_id"], kind="mergesort").groupby(PERIOD_KEY)
        return top.head(MAX_DRIVERS)[[*PERIOD_KEY, "transaction_id"]]

    def _explanation(self, rows: pd.DataFrame, hits: pd.DataFrame) -> dict[str, float]:
        """Excess coverage of the drivers, and the error of the quoted usual, on true spikes."""
        assert self.truth is not None
        if hits.empty:
            return {"excess_coverage": float("nan"), "usual_error": float("nan")}
        coverage = self.truth.score_drivers(self._drivers(hits))
        expected = self.truth.expected[self.truth.expected["granularity"] == "month"]
        e = hits.merge(expected[[*PERIOD_KEY, "expected_spend"]], on=PERIOD_KEY, how="left")
        error = (e["usual"] - e["expected_spend"]).abs() / e["expected_spend"]
        return {
            "excess_coverage": float(coverage["coverage"].mean()),
            "usual_error": float(error.median()),
        }

    def _metrics(self, rows: pd.DataFrame) -> dict[str, float]:
        flagged = rows["is_flagged"].to_numpy(dtype=bool)
        real = (rows["label"] != BASKET).to_numpy()
        out = outcome_metrics(self._outcomes(rows, flagged))
        months = user_months(rows[real])
        out["flag_rate"] = float((flagged & real).sum() / max(months, 1))
        out["flags_without_evidence"] = float((flagged & rows["evidence"].isna().to_numpy()).sum())
        out["cutoff_shortfall"] = 1.0 - out["precision"]

        at_rate, basket_at_rate = self._at_rate(rows)
        r = outcome_metrics(self._outcomes(rows, at_rate))
        out["precision_at_rate"], out["recall_at_rate"] = r["precision"], r["recall"]
        baskets = ~real
        if baskets.any():
            out["basket_recall"] = float(flagged[baskets].mean())
            out["basket_recall_at_rate"] = float(basket_at_rate[baskets].mean())

        scored = rows[real & (rows["label"] != IGNORED).to_numpy()]
        score = np.where(scored["eligible"], scored["score"].to_numpy(dtype=float), -np.inf)
        order = np.argsort(-score, kind="mergesort")
        hits = (scored["label"] == SPIKE).to_numpy()[order]
        if hits.any():
            precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
            out["average_precision"] = float(precision[hits].mean())
        out |= self._explanation(rows, rows[flagged & (rows["label"] == SPIKE).to_numpy()])
        return out

    def validation_metrics(self, examples: Examples, pooled: pd.DataFrame) -> dict[str, float]:
        return self._metrics(self._rows(examples, pooled))

    def test_metrics(self, examples: Examples, splits: Splits, model: Checked) -> dict[str, float]:
        predictions = model.predict(examples.rows(splits.sets[TEST]))
        rows = self._rows(examples, predictions)
        out = self._metrics(rows)
        weights = self._weights(rows)
        own = self._per_user(rows, rows["is_flagged"].to_numpy(dtype=bool))
        at_rate = self._per_user(rows, self._at_rate(rows)[0])
        out["precision_lo"], out["precision_hi"] = interval(
            self._ratio(weights, own, "tp", "flags")
        )
        out["recall_at_rate_lo"], out["recall_at_rate_hi"] = interval(
            self._ratio(weights, at_rate, "tp", "positives")
        )
        return out

    # --- User bootstrap ----------------------------------------------------------------------

    def _per_user(self, rows: pd.DataFrame, flagged: npt.NDArray[np.bool_]) -> pd.DataFrame:
        """Per user: flags (scored ones), true positives and labels."""
        label = rows["label"].to_numpy()
        scored = ~np.isin(label, (IGNORED, BASKET))
        frame = pd.DataFrame(
            {
                "user_id": rows["user_id"].to_numpy(),
                "flags": flagged & scored,
                "tp": flagged & (label == SPIKE),
                "positives": label == SPIKE,
            }
        )
        return frame.groupby("user_id").sum()

    def _weights(self, rows: pd.DataFrame) -> npt.NDArray[np.float64]:
        """How often each user (sorted) is drawn in each bootstrap sample."""
        users = np.sort(rows["user_id"].unique())
        rng = np.random.default_rng(0)
        draws = rng.integers(len(users), size=(self.reps, len(users)))
        return np.stack([np.bincount(d, minlength=len(users)) for d in draws]).astype(float)

    @staticmethod
    def _ratio(
        weights: npt.NDArray[np.float64], per_user: pd.DataFrame, top: str, bottom: str
    ) -> npt.NDArray[np.float64]:
        per_user = per_user.sort_index()
        num = weights @ per_user[top].to_numpy(dtype=float)
        den = weights @ per_user[bottom].to_numpy(dtype=float)
        ratio: npt.NDArray[np.float64] = np.divide(
            num, den, out=np.full(len(num), np.nan), where=den > 0
        )
        return ratio

    def _paired(
        self,
        examples: Examples,
        a: pd.DataFrame,
        b: pd.DataFrame,
        top: str,
        bottom: str,
        at_rate: bool,
    ) -> npt.NDArray[np.float64]:
        rows_a, rows_b = self._rows(examples, a), self._rows(examples, b)
        if set(rows_a["user_id"]) != set(rows_b["user_id"]):
            raise ValueError("runs scored different users; they aren't comparable")
        weights = self._weights(rows_a)

        def users(rows: pd.DataFrame) -> pd.DataFrame:
            flagged = self._at_rate(rows)[0] if at_rate else rows["is_flagged"].to_numpy(dtype=bool)
            return self._per_user(rows, flagged)

        return self._ratio(weights, users(rows_a), top, bottom) - self._ratio(
            weights, users(rows_b), top, bottom
        )

    def selection_interval(self, examples: Examples, pooled: pd.DataFrame) -> tuple[float, float]:
        """95% user-bootstrap interval of validation recall at the common flag rate."""
        rows = self._rows(examples, pooled)
        return interval(
            self._ratio(
                self._weights(rows), self._per_user(rows, self._at_rate(rows)[0]), "tp", "positives"
            )
        )

    def difference_interval(
        self, examples: Examples, a: pd.DataFrame, b: pd.DataFrame
    ) -> tuple[float, float]:
        """95% paired user-bootstrap interval of recall at the common rate, run a minus run b."""
        return interval(self._paired(examples, a, b, "tp", "positives", at_rate=True))

    def tied(self, examples: Examples, leader: pd.DataFrame, other: pd.DataFrame) -> bool:
        lo, hi = self.difference_interval(examples, leader, other)
        return lo <= 0 <= hi

    def tiebreak_tied(self, examples: Examples, leader: pd.DataFrame, other: pd.DataFrame) -> bool:
        """Precision at each run's own cutoff, paired user bootstrap: tied if the interval of the
        difference holds 0 (§4, the first tie-break)."""
        lo, hi = interval(self._paired(examples, leader, other, "tp", "flags", at_rate=False))
        return lo <= 0 <= hi

    def diagnostics(self, examples: Examples, pooled: pd.DataFrame) -> list[str]:
        """Where a run's validation errors go, at its own cutoff (§6)."""
        rows = self._rows(examples, pooled)
        flagged = rows["is_flagged"].to_numpy(dtype=bool)
        fp = rows[flagged & (rows["label"] == NORMAL).to_numpy()]
        spikes = rows[rows["label"] == SPIKE]
        caught = spikes["is_flagged"].astype(bool)

        def counts(title: str, values: pd.Series) -> list[str]:
            head = [f"| {title} | False positives |", "| --- | --- |"]
            return head + [f"| {k} | {n} |" for k, n in values.value_counts().sort_index().items()]

        def banded(values: pd.Series, bands: tuple[tuple[float, float, str], ...]) -> pd.Series:
            out = pd.Series("", index=values.index)
            for lo, hi, name in bands:
                out[values.between(lo, hi)] = name
            return out

        def recall(title: str, groups: pd.Series) -> list[str]:
            head = [f"| Recall by {title} | Recall | Labels |", "| --- | --- | --- |"]
            body = [
                f"| {k} | {caught[groups == k].mean():.2f} | {int((groups == k).sum())} |"
                for k in sorted(groups.dropna().unique())
            ]
            return head + body

        return [
            *counts("False positives by category", fp["category"]),
            "",
            *counts("By month of year", fp["period_start"].str[5:7]),
            "",
            *counts("By usual purchases a month", banded(fp["usual_count"], COUNT_BANDS)),
            "",
            *recall("tier", spikes["tier"]),
            "",
            *recall("category", spikes["category"]),
            "",
            *recall("usual purchases a month", banded(spikes["usual_count"], COUNT_BANDS)),
            "",
            *recall("history length", banded(spikes["history_months"], HISTORY_BANDS)),
            "",
            *recall("persona (reading only, never gated)", spikes["persona"]),
        ]

    # --- Selection and gates -----------------------------------------------------------------

    def eligible(
        self, metrics: Mapping[str, float], baselines: Mapping[str, Mapping[str, float]]
    ) -> bool:
        precision = metrics.get("val_precision", float("nan"))
        recall = metrics.get("val_recall_at_rate", float("nan"))
        floor = baselines.get(BASELINE, {}).get("val_recall_at_rate", float("inf"))
        return bool(precision >= PRECISION_GATE and recall > floor)

    def gates(
        self, metrics: Mapping[str, float], baselines: Mapping[str, Mapping[str, float]]
    ) -> list[Gate]:
        """§5: precision at least 0.70 on test users; recall above the mean ± k·std baseline's
        at the same flag rate; and every flag carries its evidence (reason and size)."""
        precision = metrics.get("test_precision", float("nan"))
        recall = metrics.get("test_recall_at_rate", float("nan"))
        theirs = baselines.get(BASELINE, {}).get("test_recall_at_rate")
        unexplained = metrics.get("test_flags_without_evidence", float("nan"))
        interval_text = (
            f" ({metrics.get('test_precision_lo', float('nan')):.3f}-"
            f"{metrics.get('test_precision_hi', float('nan')):.3f})"
        )
        return [
            Gate(
                "precision",
                precision >= PRECISION_GATE,
                f"{precision:.3f}{interval_text} vs {PRECISION_GATE}",
            ),
            Gate(
                "beats_baseline_at_rate",
                theirs is not None and recall > theirs,
                f"{recall:.3f} vs {theirs:.3f}" if theirs is not None else "no baseline run",
            ),
            Gate("every_flag_has_evidence", unexplained == 0, f"{unexplained:.0f} without it"),
        ]

    def serving_files(
        self,
        examples: Examples,
        pooled: pd.DataFrame,
        model_version: str,
        source: Mapping[str, str],
    ) -> dict[str, str]:
        return {}  # the cutoff travels inside the model; season profiles are built nightly

    def reproduction_configs(self) -> set[str]:
        return set()


register_task("spending_spikes", SpendingSpikesTask)
