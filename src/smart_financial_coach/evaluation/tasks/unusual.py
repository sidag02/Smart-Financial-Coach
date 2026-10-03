"""The unusual-transactions task (FR-7 §4-§6): examples, splits, metrics, ties and gates.

Examples are every outflow of every user. Labels come from FR-2's label contract (`data.labels`):
`anomaly` for a planted unusual charge, `ignored` for what the contract ignores (the warm-up and
duplicate originals), `normal` otherwise. Models get them only through `fit`, where scorers never
see them and `Thresholded` uses them to place its cutoff.

Splits:
- `train`: train users' outflows; validation is 5 folds of whole train users, by persona.
- `test`: test users' outflows, scored once by `finalize`.

Merchant profiles come from two pools (§2, from review): train users' rows carry profiles built
from train users only, so test users never shape a validation score or model selection; test
users' rows carry profiles built from everyone, as serving does.

Runs are ranked at a common operating point (§4, from review): out-of-fold recall when each
held-out fold flags its top `FLAG_RATE` charges per user-month. Each run's precision at its own
tuned cutoff is reported beside it, and gates eligibility and promotion.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.data.features.history import history_features
from smart_financial_coach.data.features.merchant_profiles import profile_features
from smart_financial_coach.data.labels import Truth, load_truth
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
from smart_financial_coach.intelligence.anomaly.contract import INPUT_COLUMNS, scoring_rows
from smart_financial_coach.intelligence.anomaly.threshold import (
    ANOMALY,
    IGNORED,
    NORMAL,
    UNSCORED,
    WARMUP,
    top_k,
    user_months,
)
from smart_financial_coach.intelligence.models.contract import Checked

TEST = "test"
# The common operating point (FR-7 §4): flags per user-month, fixed before the round at the
# POC's 0.80-precision point
FLAG_RATE = 0.11
PRECISION_GATE = 0.70  # the PRD's target, on test users (§5)
BASELINE = "user_zscore"  # the run name every candidate must beat at FLAG_RATE
BOOTSTRAP_REPS = 1000
KINDS = ("duplicate", "amount_outlier", "new_merchant_large")
DEFAULTS: dict[str, Any] = {"k": 5}
# A merchant gets a profile from 3 other users (`profile_features`' min_users)
PROFILE_BANDS = ((0, 2, "none (under 3 users)"), (3, 5, "3-5 users"), (6, 10**9, "6+ users"))
HISTORY_BANDS = ((0, 29, "< 30 earlier"), (30, 199, "30-199"), (200, 10**9, "200+"))


def labels_for(truth: Truth, transaction_ids: pd.Series) -> pd.Series:
    """`anomaly`, `ignored` or `normal` per transaction, by the label contract."""
    kind = transaction_ids.map(truth.transactions.set_index("transaction_id")["anomaly_kind"])
    ignored = transaction_ids.isin(truth.ignored_transaction_ids())
    warmup = transaction_ids.map(truth.transactions.set_index("transaction_id")["ts"]) < (
        truth.warmup_end_day
    )
    values = np.where(
        kind.notna(), ANOMALY, np.where(warmup, WARMUP, np.where(ignored, IGNORED, NORMAL))
    )
    return pd.Series(values, index=transaction_ids.index, name="label")


class UnusualTransactionsTask:
    """Metric choices and the alternatives considered: FR-7 design, "Metrics and why"."""

    name = "unusual_transactions"
    selection_metric = "recall_at_rate"
    tuning_metric = "recall_at_rate"  # grid points are chosen at the same operating point
    # Among tied runs (lower is better): wrong reasons, then the own cutoff's shortfall from
    # precision 1 (the likelier a run is to hold the gate on test users), then batch cost
    tiebreak_metrics: tuple[str, ...] = (
        "val_reason_error",
        "val_cutoff_shortfall",
        "latency_batch_ms",
    )
    shipping_params: tuple[str, ...] = ()  # scorers don't train on labels: no shipping twins
    serving_files_required: tuple[str, ...] = ()  # no review policy or other serving files
    report_metrics: tuple[str, ...] = (
        "val_precision_at_rate",
        "val_precision",
        "val_recall",
        "val_flag_rate",
        "val_reason_accuracy",
        "val_recall_clear",
        "val_recall.duplicate",
        "val_recall.amount_outlier",
        "val_recall.new_merchant_large",
        "val_average_precision",
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
        "test_reason_accuracy",
        "test_recall_clear",
    )
    required_baselines: tuple[str, ...] = (BASELINE,)
    bootstrap_unit = "user"  # the report's wording: intervals resample users
    diagnostics_scope = "at its own cutoff"

    def __init__(self, reps: int = BOOTSTRAP_REPS) -> None:
        self.reps = reps
        self.truth: Truth | None = None

    # --- Data and splits -------------------------------------------------------------------

    def load(self, data: Path) -> Examples:
        meta = load_meta(data)
        truth = load_truth(data)
        self.truth = truth
        users = load_users(data)[["user_id", "split", "persona"]]
        txns = load_transactions(data)
        split = txns["user_id"].map(users.set_index("user_id")["split"])
        train = txns[split == "train"]
        parts = [
            scoring_rows(train, pool=train).assign(profile_pool="train"),
            scoring_rows(txns[split == "test"], pool=txns).assign(profile_pool="all"),
        ]
        frame = pd.concat(parts, ignore_index=True).merge(users, on="user_id", how="left")
        history = history_features(frame)[["prior_charges", "merchant_key"]]
        frame = pd.concat([frame, history], axis=1)
        frame["label"] = labels_for(truth, frame["transaction_id"])
        kinds = truth.transactions.set_index("transaction_id")[["anomaly_kind", "tier"]]
        frame = frame.join(kinds, on="transaction_id")
        return Examples(
            frame=frame,
            labels=frame["label"],
            id_column="transaction_id",
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
            id_column="transaction_id",
            stratify="persona",
        )
        return Splits(
            sets={
                TRAIN: ids(train["transaction_id"]),
                TEST: ids(f.loc[f["split"] == "test", "transaction_id"]),
            },
            folds=folds,
        )

    def leak_errors(self, examples: Examples, splits: Splits) -> list[str]:
        f = examples.frame.set_index("transaction_id")
        errors = leak_errors(splits, examples.frame, id_column="transaction_id", group="user_id")
        for name, split, pool in ((TRAIN, "train", "train"), (TEST, "test", "all")):
            rows = f.loc[splits.sets[name].tolist()]
            if (rows["split"] != split).any():
                errors.append(
                    f"{name} has outflows of {'test' if split == 'train' else 'train'} users"
                )
            if (rows["profile_pool"] != pool).any():
                errors.append(f"{name} rows carry profiles from the wrong pool (want {pool})")
        if shared := set(f.loc[splits.sets[TRAIN].tolist(), "user_id"]) & set(
            f.loc[splits.sets[TEST].tolist(), "user_id"]
        ):
            errors.append(f"users in both train and test: {sorted(shared)[:5]}")
        errors += self._validation_profile_errors(examples, splits)
        return errors

    @staticmethod
    def _validation_profile_errors(examples: Examples, splits: Splits) -> list[str]:
        """Checked from the data, not the `profile_pool` tag: rebuilding the train rows' profiles
        from train users' outflows alone must give exactly the loaded values, so no test user
        shaped a validation score (review on #34). Leaving the user out and the as-of-month rule
        are covered by `profile_features`' brute-force tests."""
        f = examples.frame
        train = f[f["transaction_id"].isin(set(splits.sets[TRAIN]))].reset_index(drop=True)
        rebuilt = profile_features(train, train)
        for column in ("profile_users", "profile_typical", "profile_spread"):
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
        return examples.rows(which), examples.labels_for(which)

    # --- Metrics ---------------------------------------------------------------------------

    def _rows(self, examples: Examples, predictions: pd.DataFrame) -> pd.DataFrame:
        """Predictions with the evaluation columns they're scored on."""
        cols = ["user_id", "ts", "label", "anomaly_kind", "tier", "merchant_key"]
        cols += ["profile_users", "prior_charges"]
        f = examples.frame.set_index("transaction_id")
        joined = f.loc[predictions["transaction_id"].tolist(), cols].reset_index(drop=True)
        fold = (
            predictions["fold"] if "fold" in predictions else pd.Series(0, index=predictions.index)
        )
        return pd.concat(
            [
                predictions[["transaction_id", "score", "is_flagged", "reason_code"]].reset_index(
                    drop=True
                ),
                joined,
                fold.reset_index(drop=True).rename("fold"),
            ],
            axis=1,
        )

    def _at_rate(self, rows: pd.DataFrame) -> npt.NDArray[np.bool_]:
        """Each fold flags its top `FLAG_RATE` charges per user-month. Rows the label contract
        ignores don't use up the budget: they'd be neither right nor wrong."""
        flagged = np.zeros(len(rows), dtype=bool)
        for _, idx in rows.groupby("fold").indices.items():
            part = rows.iloc[idx]
            # The budget excludes the warm-up by date only; duplicate originals (from truth) stay
            # in it, as a deployed cutoff can't know them (review on #34)
            k = round(FLAG_RATE * user_months(part[part["label"] != WARMUP]))
            score = np.where(part["label"] == WARMUP, -np.inf, part["score"].to_numpy(dtype=float))
            flagged[idx] = top_k(score, part["transaction_id"].to_numpy(), k)
        return flagged

    def _outcomes(self, rows: pd.DataFrame, flagged: npt.NDArray[np.bool_]) -> pd.DataFrame:
        """The label contract's outcome per flag and per missed label, for these rows' users."""
        assert self.truth is not None, "load() first"
        flags = rows.loc[flagged, ["transaction_id", "reason_code"]]
        outcomes = self.truth.score_transactions(flags)
        return outcomes[outcomes["transaction_id"].isin(set(rows["transaction_id"]))]

    def _per_user(self, rows: pd.DataFrame, flagged: npt.NDArray[np.bool_]) -> pd.DataFrame:
        """Per user: flags, true positives, labels, and reasons right among true positives."""
        scored = ~rows["label"].isin(UNSCORED)
        hit = flagged & scored.to_numpy() & (rows["label"] == ANOMALY).to_numpy()
        want = rows["anomaly_kind"].map(
            {
                "duplicate": "duplicate",
                "amount_outlier": "amount_unusual",
                "new_merchant_large": "new_merchant",
            }
        )
        right = hit & (rows["reason_code"] == want).to_numpy()
        frame = pd.DataFrame(
            {
                "user_id": rows["user_id"].to_numpy(),
                "flags": flagged & scored.to_numpy(),
                "tp": hit,
                "positives": (rows["label"] == ANOMALY).to_numpy(),
                "reasons_right": right,
            }
        )
        return frame.groupby("user_id").sum()

    def _metrics(self, rows: pd.DataFrame) -> dict[str, float]:
        flagged = rows["is_flagged"].to_numpy(dtype=bool)
        outcomes = self._outcomes(rows, flagged)
        out = outcome_metrics(outcomes)
        missed = outcomes["outcome"] == "fn"
        found = outcomes["outcome"] == "tp"
        for kind in KINDS:
            of_kind = outcomes["anomaly_kind"] == kind
            total = int((of_kind & (missed | found)).sum())
            out[f"recall.{kind}"] = (
                float((of_kind & found).sum() / total) if total else float("nan")
            )
        months = user_months(rows[rows["label"] != WARMUP])  # after the warm-up (§6)
        out["flag_rate"] = float((flagged & (rows["label"] != WARMUP).to_numpy()).sum() / months)
        out["flags_without_reason"] = float((flagged & rows["reason_code"].isna().to_numpy()).sum())
        out["cutoff_shortfall"] = 1.0 - out["precision"]
        out["reason_error"] = 1.0 - out.get("reason_accuracy", float("nan"))

        at_rate = outcome_metrics(self._outcomes(rows, self._at_rate(rows)))
        out["precision_at_rate"], out["recall_at_rate"] = at_rate["precision"], at_rate["recall"]

        scored = rows[~rows["label"].isin(UNSCORED)]
        order = np.argsort(-scored["score"].to_numpy(dtype=float), kind="mergesort")
        hits = (scored["label"] == ANOMALY).to_numpy()[order]
        if hits.any():
            precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
            out["average_precision"] = float(precision[hits].mean())
        return out

    def validation_metrics(self, examples: Examples, pooled: pd.DataFrame) -> dict[str, float]:
        return self._metrics(self._rows(examples, pooled))

    def test_metrics(self, examples: Examples, splits: Splits, model: Checked) -> dict[str, float]:
        predictions = model.predict(examples.rows(splits.sets[TEST]))
        rows = self._rows(examples, predictions)
        out = self._metrics(rows)
        weights = self._weights(rows)
        own = self._per_user(rows, rows["is_flagged"].to_numpy(dtype=bool))
        at_rate = self._per_user(rows, self._at_rate(rows))
        out["precision_lo"], out["precision_hi"] = interval(
            self._ratio(weights, own, "tp", "flags")
        )
        out["recall_at_rate_lo"], out["recall_at_rate_hi"] = interval(
            self._ratio(weights, at_rate, "tp", "positives")
        )
        return out

    # --- User bootstrap ----------------------------------------------------------------------

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
            flagged = self._at_rate(rows) if at_rate else rows["is_flagged"].to_numpy(dtype=bool)
            return self._per_user(rows, flagged)

        return self._ratio(weights, users(rows_a), top, bottom) - self._ratio(
            weights, users(rows_b), top, bottom
        )

    def selection_interval(self, examples: Examples, pooled: pd.DataFrame) -> tuple[float, float]:
        """95% user-bootstrap interval of validation recall at the common flag rate."""
        rows = self._rows(examples, pooled)
        return interval(
            self._ratio(
                self._weights(rows), self._per_user(rows, self._at_rate(rows)), "tp", "positives"
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
        """Reason accuracy at each run's own cutoff, paired user bootstrap: tied if the interval
        of the difference holds 0 (FR-7 §4, the first tie-break)."""
        lo, hi = interval(
            self._paired(examples, leader, other, "reasons_right", "tp", at_rate=False)
        )
        return lo <= 0 <= hi

    def diagnostics(self, examples: Examples, pooled: pd.DataFrame) -> list[str]:
        """Where a run's validation errors go, at its own cutoff (FR-7 §6): false positives by
        merchant, and misses by kind against profile size and history length."""
        rows = self._rows(examples, pooled)
        flagged = rows["is_flagged"].to_numpy(dtype=bool)
        fp = rows[flagged & (rows["label"] == NORMAL).to_numpy()]
        lines = ["| Most frequent false-positive merchants | Flags |", "| --- | --- |"]
        lines += [
            f"| {key} | {n} |" for key, n in fp["merchant_key"].value_counts().head(8).items()
        ]

        positives = rows[rows["label"] == ANOMALY]
        caught = positives["is_flagged"].astype(bool)

        def table(column: str, bands: tuple[tuple[int, int, str], ...], title: str) -> list[str]:
            head = [f"| Recall by kind and {title} |", "| --- |"]
            head[0] += "".join(f" {name} |" for *_, name in bands)
            head[1] += " --- |" * len(bands)
            body = []
            for kind in KINDS:
                cells = []
                for lo, hi, _ in bands:
                    band = (positives["anomaly_kind"] == kind) & positives[column].between(lo, hi)
                    n = int(band.sum())
                    cells.append(f"{caught[band].mean():.2f} of {n}" if n else "-")
                body.append(f"| {kind} | " + " | ".join(cells) + " |")
            return head + body

        return [
            *lines,
            "",
            *table("profile_users", PROFILE_BANDS, "merchant-profile size"),
            "",
            *table("prior_charges", HISTORY_BANDS, "the user's history length"),
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
        """FR-7 §5: precision at least 0.70 on test users; recall above the per-user z baseline's
        at the same flag rate; and every flag has a reason."""
        precision = metrics.get("test_precision", float("nan"))
        recall = metrics.get("test_recall_at_rate", float("nan"))
        theirs = baselines.get(BASELINE, {}).get("test_recall_at_rate")
        unexplained = metrics.get("test_flags_without_reason", float("nan"))
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
            Gate("every_flag_has_a_reason", unexplained == 0, f"{unexplained:.0f} without one"),
        ]

    def serving_files(
        self,
        examples: Examples,
        pooled: pd.DataFrame,
        model_version: str,
        source: Mapping[str, str],
    ) -> dict[str, str]:
        return {}  # the cutoff travels inside the model (`Thresholded`); nothing else to serve

    def reproduction_configs(self) -> set[str]:
        return set()  # no published POC configurations to reproduce through the runner


register_task("unusual_transactions", UnusualTransactionsTask)
