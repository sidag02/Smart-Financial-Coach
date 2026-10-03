"""The categorization task (FR-3): examples, splits, label noise, metrics, ties and gates.

Splits (Technical Design, FR-3 §5):
- `test_known`: a stratified 20% of train users' transactions; `train` is the other 80%.
- `test_all`: every test-user transaction; `test_unseen`: those at holdout merchants.
- Validation folds over `train`: K merchant-grouped folds that hold out only holdout-eligible
  merchants (the population `test_unseen` comes from), plus a 10% seen-merchant row sample.

Headline macro F1 is over the 12 spending categories; Income is reported on its own (FR-3 §4).
"""

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.data.features.merchant_text import normalize_merchant
from smart_financial_coach.data.generator.taxonomy import INCOME
from smart_financial_coach.data.store import load_meta, load_transactions, load_users
from smart_financial_coach.evaluation.experiment import load_experiment
from smart_financial_coach.evaluation.metrics.classification import (
    bootstrap_macro_f1,
    bootstrap_weights,
    calibration,
    group_confusions,
    interval,
    macro_f1,
    misallocated_spend,
    per_class_f1,
)
from smart_financial_coach.evaluation.splits import (
    SEEN,
    TRAIN,
    UNSEEN,
    Ids,
    Splits,
    group_kfold,
    ids,
    leak_errors,
    stratified_sample,
)
from smart_financial_coach.evaluation.tasks.base import Examples, Gate, register_task
from smart_financial_coach.intelligence.categorization.contract import INPUT_COLUMNS
from smart_financial_coach.intelligence.models.contract import Checked

MIN_SCHEMA = 3  # truth_merchants.holdout_eligible
POC_CONFIGS = PROJECT_ROOT / "configs" / "experiments" / "categorization" / "poc"
# The POC's published test numbers (known / all test users / unseen merchants), for the
# reproduction check (FR-3 §3)
POC_RESULTS = {
    "poc_ngrams": (0.986, 0.868, 0.504),
    "poc_embeddings": (0.973, 0.913, 0.643),
    "poc_both": (0.988, 0.929, 0.638),
}
KNOWN_GATE = 0.90
# FR-4's v1 gate. Accepted at 0.70 (Oct 2, 2026) when the POC's leader stood at 0.745; set to 0.66
# (owner, Oct 3, 2026) after the FR-4 round's validation results, before any test scoring: the
# round's leader scores 0.712, and 0.66 keeps the design's pass rate of about 85-90% for a model
# that good. 0.80 stays the goal for v1.1, through feedback (FR-5/FR-6).
# Scope: these gates promote FR-4's own candidates. Models retrained from feedback are promoted on
# FR-5/FR-6 §5's gates, scored against users' own view, with these truth-based numbers reported,
# not gated (owner, on PR #15); the retraining path chooses its gates when it's built
UNSEEN_GATE = 0.66
KEYWORD_BASELINE = "keyword"  # the run name the "beats the baseline" rule compares against
BOOTSTRAP_REPS = 1000
AMBIGUOUS_CLASSES = ("Groceries", "Shopping")
DEFAULTS: dict[str, Any] = {"test_frac": 0.2, "k": 5, "seen_frac": 0.1, "label_noise": 0.02}


def slug(label: str) -> str:
    """MLflow-safe metric suffix: "Insurance & Fees" -> "insurance_fees"."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _truth(data: Path) -> pd.DataFrame:
    sql = """
        SELECT t.transaction_id, t.category, t.merchant_id,
               m.holdout, m.holdout_eligible, m.is_ambiguous
        FROM truth_transactions t JOIN truth_merchants m USING (merchant_id)
    """
    conn = sqlite3.connect(f"file:{data}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def content_hash(frame: pd.DataFrame, meta: Mapping[str, str]) -> str:
    """A hash of exactly what was loaded (about a second on the default dataset).

    Not the spec hash and generator version alone: the version string doesn't change with every
    generator change, so different data could otherwise look identical.
    """
    rows = pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    header = f"{meta['spec_hash']}|{meta['schema_version']}|".encode()
    return hashlib.sha256(header + rows).hexdigest()


class CategorizationTask:
    """Metric choices and the alternatives considered: FR-3 design, "Metrics and why"."""

    name = "categorization"
    selection_metric = "unseen_macro_f1"  # ranks runs (the decision rule)
    tuning_metric = "tuning_macro_f1"  # picks C within a run: mean of known and unseen
    # Among tied runs: better confidence, then cheaper batch serving (ms for a fixed 10k-row batch)
    tiebreak_metrics: tuple[str, ...] = ("val_unseen_brier", "latency_batch_ms")
    # Promoted configs set these explicitly; the leaderboard ranks comparison runs under the
    # injected noise, and their `label_noise: 0` shipping twins are finalized (FR-4 §1)
    shipping_params: tuple[str, ...] = ("label_noise",)
    # Columns of the comparison report (`sfc-experiment report`), after the selection metric
    report_metrics: tuple[str, ...] = (
        "val_known_macro_f1",
        "val_unseen_brier",
        "val_unseen_ece",
        "val_unseen_acc_at_90",
        "val_unseen_coverage_at_90",
        "val_unseen_misallocated_spend",
        "latency_batch_ms",
        "latency_p95_ms",
    )
    # Test-set columns, for finalists and baselines once `finalize` has run
    test_report_metrics: tuple[str, ...] = (
        "test_known_macro_f1",
        "test_all_macro_f1",
        "test_unseen_macro_f1",
        "test_unseen_brier",
        "test_unseen_acc_at_90",
        "test_known_ece",
        "test_known_coverage_at_90",
        "test_unseen_misallocated_spend",
        "test_refund_accuracy",
    )
    required_baselines: tuple[str, ...] = (KEYWORD_BASELINE,)  # eligibility and gates need it

    def __init__(self, reps: int = BOOTSTRAP_REPS) -> None:
        self.reps = reps
        self.spending: tuple[str, ...] = ()

    # --- Data and splits -------------------------------------------------------------------

    def load(self, data: Path) -> Examples:
        meta = load_meta(data)
        if int(meta.get("schema_version", "0")) < MIN_SCHEMA:
            raise ValueError(f"{data} has schema {meta.get('schema_version')}; regenerate it")
        categories = [str(c) for c in json.loads(meta["categories"])]
        self.spending = tuple(c for c in categories if c != INCOME)
        frame = load_transactions(data).merge(
            load_users(data)[["user_id", "split"]], on="user_id", validate="many_to_one"
        )
        frame = frame.merge(_truth(data), on="transaction_id", validate="one_to_one")
        return Examples(
            frame=frame,
            labels=frame["category"],
            id_column="transaction_id",
            model_columns=INPUT_COLUMNS,
            data_hash=content_hash(frame, meta),
        )

    def split(self, examples: Examples, params: Mapping[str, Any], seed: int) -> Splits:
        p = DEFAULTS | dict(params)
        f = examples.frame
        train_users = f[f["split"] == "train"]
        test_known = stratified_sample(
            train_users, frac=p["test_frac"], by="category", seed=seed, id_column="transaction_id"
        )
        train = train_users[~train_users["transaction_id"].isin(test_known)]
        test_all = f[f["split"] == "test"]
        eligible = set(f.loc[f["holdout_eligible"] == 1, "merchant_id"])
        folds = group_kfold(
            train,
            group="merchant_id",
            k=int(p["k"]),
            seed=seed,
            id_column="transaction_id",
            stratify="category",
            eligible=eligible,
            seen_frac=float(p["seen_frac"]),
        )
        return Splits(
            sets={
                TRAIN: ids(train["transaction_id"]),
                "test_known": test_known,
                "test_all": ids(test_all["transaction_id"]),
                "test_unseen": ids(test_all.loc[test_all["holdout"] == 1, "transaction_id"]),
            },
            folds=folds,
        )

    def leak_errors(self, examples: Examples, splits: Splits) -> list[str]:
        f = examples.frame.set_index("transaction_id")
        protected = set(f.loc[f["holdout_eligible"] == 0, "merchant_id"])
        errors = leak_errors(
            splits,
            examples.frame,
            id_column="transaction_id",
            group="merchant_id",
            never_held_out=protected,
        )
        unseen = f.loc[splits.sets["test_unseen"].tolist()]
        if (unseen["holdout"] != 1).any():
            errors.append("test_unseen has transactions at non-holdout merchants")
        for name, split in (("test_known", "train"), ("test_all", "test"), (TRAIN, "train")):
            if (f.loc[splits.sets[name].tolist(), "split"] != split).any():
                errors.append(
                    f"{name} has transactions of {'test' if split == 'train' else 'train'} users"
                )
        return errors

    def training_rows(
        self, examples: Examples, which: Ids, params: Mapping[str, Any], seed: int
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Label noise (Technical Design control), applied after splitting to training rows only.

        A share of spending labels flips uniformly to another spending category. Income is never
        flipped: a payroll labeled Dining isn't a realistic correction error.
        """
        x, y = examples.rows(which), examples.labels_for(which)
        assert y is not None
        rate = float((DEFAULTS | dict(params))["label_noise"])
        spending = np.flatnonzero((y != INCOME).to_numpy())
        n = round(rate * len(spending))
        if n:
            rng = np.random.default_rng(seed)
            flip = rng.choice(spending, size=n, replace=False)
            others = np.asarray(self.spending)
            labels = y.to_numpy().copy()
            for i in flip:
                choices = others[others != labels[i]]
                labels[i] = choices[rng.integers(len(choices))]
            y = pd.Series(labels, name=y.name)
        return x, y

    # --- Metrics ---------------------------------------------------------------------------

    def _join(self, examples: Examples, predictions: pd.DataFrame) -> pd.DataFrame:
        """Predictions with truth columns; rows can repeat (seen in more than one fold)."""
        truth = examples.frame.set_index("transaction_id")
        cols = ["category", "merchant_id", "amount", "merchant_raw", "is_ambiguous"]
        joined = truth.loc[predictions["transaction_id"].tolist(), cols].reset_index(drop=True)
        return pd.concat(
            [
                predictions[["category", "confidence"]].rename(columns={"category": "predicted"}),
                joined,
            ],
            axis=1,
        )

    def _scores(self, rows: pd.DataFrame, prefix: str) -> dict[str, float]:
        spending = rows[rows["category"] != INCOME]
        out = {
            f"{prefix}_macro_f1": macro_f1(
                spending["category"], spending["predicted"], self.spending
            ),
            f"{prefix}_misallocated_spend": misallocated_spend(
                spending["category"], spending["predicted"], spending["amount"]
            ),
        }
        cal = calibration(rows["category"], rows["predicted"], rows["confidence"])
        return out | {f"{prefix}_{k}": v for k, v in cal.items()}

    def validation_metrics(self, examples: Examples, pooled: pd.DataFrame) -> dict[str, float]:
        rows = self._join(examples, pooled.reset_index(drop=True))
        held = pooled["held_out"].to_numpy()
        out = self._scores(rows[held == SEEN], "known") | self._scores(
            rows[held == UNSEEN], "unseen"
        )
        out["tuning_macro_f1"] = (out["known_macro_f1"] + out["unseen_macro_f1"]) / 2
        return out

    def test_metrics(self, examples: Examples, splits: Splits, model: Checked) -> dict[str, float]:
        known = self._join(examples, model.predict(examples.rows(splits.sets["test_known"])))
        everyone = self._join(examples, model.predict(examples.rows(splits.sets["test_all"])))
        unseen = everyone[everyone["merchant_id"].isin(self._holdout(examples))]
        out = (
            self._scores(known, "known")
            | self._scores(everyone, "all")
            | self._scores(unseen, "unseen")
        )

        # Confidence by familiarity, as the shipped model sees it. Calibration is fitted per
        # familiarity group, so it's checked per group
        familiar = self._familiar(model, examples, splits)
        for name, mask in (("familiar", familiar), ("unfamiliar", ~familiar)):
            rows = everyone[mask]
            cal = calibration(rows["category"], rows["predicted"], rows["confidence"])
            out |= {f"all_{name}_{k}": v for k, v in cal.items()}
        out["all_unfamiliar_share"] = float((~familiar).mean())

        spending = known[known["category"] != INCOME]
        for label, f1 in per_class_f1(
            spending["category"], spending["predicted"], self.spending
        ).items():
            out[f"known_f1.{slug(label)}"] = f1
        every_label = [*self.spending, INCOME]  # Income's F1 counts spending rows called Income
        income_f1 = per_class_f1(known["category"], known["predicted"], every_label)
        out["known_income_f1"] = income_f1.get(INCOME, float("nan"))

        rng = np.random.default_rng(0)
        u = unseen[unseen["category"] != INCOME].reset_index(drop=True)
        _, stack, majority = group_confusions(
            u["category"], u["predicted"], u["merchant_id"], self.spending
        )
        samples = bootstrap_macro_f1(
            stack, bootstrap_weights(majority, self.reps, rng), len(self.spending)
        )
        out["unseen_macro_f1_lo"], out["unseen_macro_f1_hi"] = interval(samples)

        train_raw = set(
            examples.frame.set_index("transaction_id").loc[
                splits.sets[TRAIN].tolist(), "merchant_raw"
            ]
        )
        new = spending[~spending["merchant_raw"].isin(train_raw)]
        out["known_new_string_macro_f1"] = macro_f1(
            new["category"], new["predicted"], self.spending
        )
        out["known_new_string_share"] = float(len(new) / max(len(spending), 1))

        both = pd.concat([known, everyone], ignore_index=True)
        refunds = both[(both["amount"] > 0) & (both["category"] != INCOME)]
        out["refund_accuracy"] = float((refunds["predicted"] == refunds["category"]).mean())

        # Ambiguity ceiling: each merchant's majority training category (a perfect memorizer)
        train = examples.frame.set_index("transaction_id").loc[splits.sets[TRAIN].tolist()]
        by_merchant = train.groupby("merchant_id")["category"]
        oracle = by_merchant.agg(lambda s: s.value_counts().index[0])
        guess = spending["merchant_id"].map(oracle).fillna("")
        ceiling = per_class_f1(spending["category"], guess, self.spending)
        for label in AMBIGUOUS_CLASSES:
            if label in ceiling:
                out[f"known_oracle_f1.{slug(label)}"] = ceiling[label]
        return out

    @staticmethod
    def _familiar(model: Checked, examples: Examples, splits: Splits) -> npt.NDArray[np.bool_]:
        """Familiarity of each test-user row: the model's own flag when it has one (so it uses
        the model's normalizer), else whether the normalized string occurs in training rows."""
        rows = examples.rows(splits.sets["test_all"])
        scorer = getattr(model.model, "base", model.model)
        if callable(scores := getattr(scorer, "scores", None)):
            return np.asarray(scores(rows)[1], dtype=bool)
        train = examples.frame.set_index("transaction_id").loc[splits.sets[TRAIN].tolist()]
        vocabulary = set(train["merchant_raw"].map(normalize_merchant))
        return rows["merchant_raw"].map(normalize_merchant).isin(vocabulary).to_numpy()

    @staticmethod
    def _holdout(examples: Examples) -> set[str]:
        f = examples.frame
        return set(f.loc[f["holdout"] == 1, "merchant_id"])

    # --- Selection -------------------------------------------------------------------------

    def eligible(
        self, metrics: Mapping[str, float], baselines: Mapping[str, Mapping[str, float]]
    ) -> bool:
        known = metrics.get("val_known_macro_f1", float("nan"))
        floor = baselines.get(KEYWORD_BASELINE, {}).get("val_known_macro_f1", float("inf"))
        return bool(known >= KNOWN_GATE and known > floor)

    def _unseen_stack(
        self, examples: Examples, pooled: pd.DataFrame
    ) -> tuple[list[str], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Per-merchant confusion matrices of the pooled unseen-merchant (spending) rows."""
        rows = self._join(examples, pooled[pooled["held_out"] == UNSEEN].reset_index(drop=True))
        rows = rows[rows["category"] != INCOME]
        return group_confusions(
            rows["category"], rows["predicted"], rows["merchant_id"], self.spending
        )

    def selection_interval(self, examples: Examples, pooled: pd.DataFrame) -> tuple[float, float]:
        """95% merchant-bootstrap interval of validation unseen-merchant macro F1."""
        _, stack, majority = self._unseen_stack(examples, pooled)
        weights = bootstrap_weights(majority, self.reps, np.random.default_rng(0))
        return interval(bootstrap_macro_f1(stack, weights, len(self.spending)))

    def difference_interval(
        self, examples: Examples, a: pd.DataFrame, b: pd.DataFrame
    ) -> tuple[float, float]:
        """95% paired merchant-bootstrap interval of unseen macro F1, run a minus run b."""
        names_a, stack_a, majority = self._unseen_stack(examples, a)
        names_b, stack_b, _ = self._unseen_stack(examples, b)
        if names_a != names_b:
            raise ValueError("runs held out different merchants; they aren't comparable")
        weights = bootstrap_weights(majority, self.reps, np.random.default_rng(0))
        n = len(self.spending)
        return interval(
            bootstrap_macro_f1(stack_a, weights, n) - bootstrap_macro_f1(stack_b, weights, n)
        )

    def tied(self, examples: Examples, leader: pd.DataFrame, other: pd.DataFrame) -> bool:
        """Paired merchant bootstrap of the unseen macro-F1 difference: tied if its CI holds 0."""
        lo, hi = self.difference_interval(examples, leader, other)
        return lo <= 0 <= hi

    def _unseen_brier_stack(
        self, examples: Examples, pooled: pd.DataFrame
    ) -> tuple[list[str], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Per merchant: squared error of the top-class confidence (summed) and row count, over
        the pooled unseen-merchant rows, as `val_unseen_brier` scores them."""
        rows = self._join(examples, pooled[pooled["held_out"] == UNSEEN].reset_index(drop=True))
        correct = (rows["category"] == rows["predicted"]).to_numpy(dtype=float)
        rows = rows.assign(sse=(rows["confidence"].to_numpy(dtype=float) - correct) ** 2, n=1.0)
        by = rows.groupby("merchant_id")
        sums = by[["sse", "n"]].sum()
        majority = by["category"].agg(lambda c: c.mode().iloc[0])
        codes = {c: i for i, c in enumerate(sorted(set(majority)))}
        return (
            [str(m) for m in sums.index],
            sums.to_numpy(dtype=float),
            majority.map(codes).to_numpy(dtype=np.int64),
        )

    def diagnostics(self, examples: Examples, pooled: pd.DataFrame) -> list[str]:
        """Per spending category, on validation unseen-merchant rows: its F1, where its errors go
        (the categories its true rows are wrongly given), and what lands in it (the true
        categories of rows wrongly given it). Diagnosis for later fixes, not a gate (FR-4,
        "Metrics and why", point 3)."""
        rows = self._join(examples, pooled[pooled["held_out"] == UNSEEN].reset_index(drop=True))
        rows = rows[rows["category"] != INCOME]
        if rows.empty:
            return []
        f1 = per_class_f1(rows["category"], rows["predicted"], self.spending)
        merchants = (
            rows.groupby("merchant_id")["category"].agg(lambda c: c.mode().iloc[0]).value_counts()
        )
        wrong = rows[rows["category"] != rows["predicted"]]

        def top(counts: pd.Series, total: int) -> str:
            shares = [
                f"{name} {n / total:.0%}" if n / total >= 0.005 else f"{name} <1%"
                for name, n in counts.head(2).items()
                if n
            ]
            return ", ".join(shares) or "\u2013"

        lines = [
            "| Category | Merchants | F1 | True rows | Predicted rows | Its errors go to "
            "(share of its rows) | Wrongly given it (share of rows given it) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for label in sorted(self.spending, key=lambda c: f1.get(c, 0.0)):
            true_rows = int((rows["category"] == label).sum())
            given = int((rows["predicted"] == label).sum())
            out = wrong.loc[wrong["category"] == label, "predicted"].value_counts()
            into = wrong.loc[wrong["predicted"] == label, "category"].value_counts()
            lines.append(
                f"| {label} | {int(merchants.get(label, 0))} | {f1.get(label, 0.0):.3f} | "
                f"{true_rows:,} | {given:,} | {top(out, max(true_rows, 1))} | "
                f"{top(into, max(given, 1))} |"
            )
        return lines

    def tiebreak_tied(self, examples: Examples, leader: pd.DataFrame, other: pd.DataFrame) -> bool:
        """Paired merchant bootstrap of the unseen Brier difference: tied if its CI holds 0."""
        names_a, a, majority = self._unseen_brier_stack(examples, leader)
        names_b, b, _ = self._unseen_brier_stack(examples, other)
        if names_a != names_b:
            raise ValueError("runs held out different merchants; they aren't comparable")
        weights = bootstrap_weights(majority, self.reps, np.random.default_rng(0))
        brier_a = (weights @ a[:, 0]) / (weights @ a[:, 1])
        brier_b = (weights @ b[:, 0]) / (weights @ b[:, 1])
        lo, hi = interval(brier_a - brier_b)
        return lo <= 0 <= hi

    def gates(
        self, metrics: Mapping[str, float], baselines: Mapping[str, Mapping[str, float]]
    ) -> list[Gate]:
        """Quality gates only. Latency is reported, not gated: categorization runs in batches on
        ingestion, so serving time is a cluster-sizing cost, not a property to reject a model on
        (decision Oct 2, 2026; FR-3 "Metrics and why", point 7).

        FR-3: known-merchant macro F1 at least 0.90 and above the keyword baseline's. FR-4: unseen-
        merchant macro F1 at least `UNSEEN_GATE` (0.66) and above the keyword baseline's unseen
        score.
        """
        keyword = baselines.get(KEYWORD_BASELINE, {})

        def above_keyword(name: str, metric: str) -> Gate:
            mine, theirs = metrics.get(metric, float("nan")), keyword.get(metric)
            return Gate(
                name,
                theirs is not None and mine > theirs,
                f"{mine:.3f} vs {theirs:.3f}" if theirs is not None else "no keyword baseline run",
            )

        known = metrics.get("test_known_macro_f1", float("nan"))
        unseen = metrics.get("test_unseen_macro_f1", float("nan"))
        return [
            Gate("known_macro_f1", known >= KNOWN_GATE, f"{known:.3f} vs {KNOWN_GATE}"),
            above_keyword("beats_keyword_baseline", "test_known_macro_f1"),
            Gate("unseen_macro_f1", unseen >= UNSEEN_GATE, f"{unseen:.3f} vs {UNSEEN_GATE}"),
            above_keyword("beats_keyword_unseen", "test_unseen_macro_f1"),
        ]

    def reproduction_configs(self) -> set[str]:
        """The POC's three configurations (feasibility at commit 55d4197), by config hash."""
        return {load_experiment(p).config_hash() for p in sorted(POC_CONFIGS.glob("*.yaml"))}


register_task("categorization", CategorizationTask)
