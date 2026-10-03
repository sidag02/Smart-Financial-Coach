from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.data.features.merchant_profiles import profile_features
from smart_financial_coach.data.store import load_transactions, load_users
from smart_financial_coach.evaluation.experiment import load_experiment
from smart_financial_coach.evaluation.promote import leaderboard
from smart_financial_coach.evaluation.runner import run_experiment
from smart_financial_coach.evaluation.splits import TRAIN, UNSEEN
from smart_financial_coach.evaluation.tasks.base import Examples, get_task
from smart_financial_coach.evaluation.tasks.unusual import (
    FLAG_RATE,
    TEST,
    UnusualTransactionsTask,
)
from smart_financial_coach.evaluation.tracking import Tracker
from smart_financial_coach.intelligence.anomaly.threshold import user_months

CONFIGS = PROJECT_ROOT / "configs" / "experiments" / "unusual_transactions"


@pytest.fixture(scope="module")
def task() -> UnusualTransactionsTask:
    return UnusualTransactionsTask(reps=200)


@pytest.fixture(scope="module")
def examples(task: UnusualTransactionsTask, small_sqlite: Path) -> Examples:
    return task.load(small_sqlite)


def predictions(examples: Examples, which: np.ndarray, flag: str = "anomaly") -> pd.DataFrame:
    """Hand-made predictions over `which`: a perfect score, flags on rows labeled `flag`, and the
    right reason for every planted kind."""
    f = examples.frame.set_index("transaction_id").loc[which.tolist()]
    reasons = f["anomaly_kind"].map(
        {
            "duplicate": "duplicate",
            "amount_outlier": "amount_unusual",
            "new_merchant_large": "new_merchant",
        }
    )
    flagged = (f["label"] == flag).to_numpy()
    return pd.DataFrame(
        {
            "transaction_id": f.index.to_numpy(),
            "score": (f["label"] == "anomaly").to_numpy(dtype=float),
            "is_flagged": flagged,
            "reason_code": np.where(flagged, reasons.fillna("amount_unusual"), None),
            "fold": 0,
        }
    )


def test_the_task_is_registered() -> None:
    assert isinstance(get_task("unusual_transactions"), UnusualTransactionsTask)


def test_examples_are_outflows_with_labels(examples: Examples) -> None:
    f = examples.frame

    assert (f["amount"] < 0).all()
    assert set(f["label"]) == {"anomaly", "normal", "ignored", "warmup"}
    assert (f.loc[f["label"] == "anomaly", "anomaly_kind"].notna()).all()
    assert f["anomaly_kind"].notna().sum() == (f["label"] == "anomaly").sum()


def test_validation_profiles_never_see_test_users(examples: Examples, small_sqlite: Path) -> None:
    txns = load_transactions(small_sqlite)
    users = load_users(small_sqlite)
    train = txns[txns["user_id"].isin(users.loc[users["split"] == "train", "user_id"])]
    f = examples.frame[examples.frame["split"] == "train"].reset_index(drop=True)

    expected = profile_features(train, train)
    assert f["profile_users"].tolist() == expected["profile_users"].tolist()
    assert (f["profile_pool"] == "train").all()
    test_rows = examples.frame[examples.frame["split"] == "test"]
    assert (test_rows["profile_pool"] == "all").all()


def test_splits_hold_out_whole_users(task: UnusualTransactionsTask, examples: Examples) -> None:
    splits = task.split(examples, {}, seed=0)
    f = examples.frame.set_index("transaction_id")

    assert task.leak_errors(examples, splits) == []
    assert len(splits.folds) == 5
    for fold in splits.folds:
        held = set(f.loc[fold.held_out[UNSEEN].tolist(), "user_id"])
        assert not held & set(f.loc[fold.train.tolist(), "user_id"])
    assert set(f.loc[splits.sets[TEST].tolist(), "split"]) == {"test"}


def test_leak_check_catches_test_users_in_validation_profiles(
    task: UnusualTransactionsTask, examples: Examples
) -> None:
    frame = examples.frame.copy()
    train = frame["split"] == "train"
    frame.loc[train, "profile_users"] = frame.loc[train, "profile_users"] + 1  # as if one more
    leaky = Examples(frame, examples.labels, "transaction_id", examples.model_columns, "x")

    errors = task.leak_errors(leaky, task.split(leaky, {}, seed=0))
    assert any("isn't built from train users alone" in e for e in errors)


def test_leak_check_catches_profiles_from_the_wrong_pool(
    task: UnusualTransactionsTask, examples: Examples
) -> None:
    frame = examples.frame.copy()
    frame.loc[frame["split"] == "train", "profile_pool"] = "all"
    leaky = Examples(frame, examples.labels, "transaction_id", examples.model_columns, "x")

    errors = task.leak_errors(leaky, task.split(leaky, {}, seed=0))
    assert any("wrong pool" in e for e in errors)


def test_perfect_predictions_score_perfectly(
    task: UnusualTransactionsTask, examples: Examples
) -> None:
    train = task.split(examples, {}, seed=0).sets[TRAIN]
    m = task.validation_metrics(examples, predictions(examples, train))

    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["reason_accuracy"] == 1.0
    assert m["average_precision"] == 1.0
    assert m["flags_without_reason"] == 0
    for kind in ("duplicate", "amount_outlier", "new_merchant_large"):
        assert m[f"recall.{kind}"] == 1.0
    # At the common rate, a perfect scorer finds what fits in the budget
    rows = examples.frame.set_index("transaction_id").loc[train.tolist()]
    budget = round(FLAG_RATE * user_months(rows[rows["label"] != "warmup"].reset_index()))
    positives = int((rows["label"] == "anomaly").sum())
    assert m["recall_at_rate"] == pytest.approx(min(budget, positives) / positives)


def test_flags_on_normal_charges_are_false_positives(
    task: UnusualTransactionsTask, examples: Examples
) -> None:
    train = task.split(examples, {}, seed=0).sets[TRAIN]
    m = task.validation_metrics(examples, predictions(examples, train, flag="normal"))

    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["fp"] > 0


def test_ties_and_intervals(task: UnusualTransactionsTask, examples: Examples) -> None:
    train = task.split(examples, {}, seed=0).sets[TRAIN]
    good = predictions(examples, train)
    noise = good.assign(score=np.random.default_rng(0).random(len(good)))

    assert task.tied(examples, good, good)
    assert not task.tied(examples, good, noise)
    lo, hi = task.selection_interval(examples, good)
    assert lo <= hi <= 1.0
    assert task.tiebreak_tied(examples, good, good)


def test_eligibility_and_gates(task: UnusualTransactionsTask) -> None:
    baselines = {"user_zscore": {"val_recall_at_rate": 0.1, "test_recall_at_rate": 0.1}}

    assert task.eligible({"val_precision": 0.75, "val_recall_at_rate": 0.6}, baselines)
    assert not task.eligible({"val_precision": 0.65, "val_recall_at_rate": 0.6}, baselines)
    assert not task.eligible({"val_precision": 0.75, "val_recall_at_rate": 0.05}, baselines)
    assert not task.eligible({"val_precision": 0.75, "val_recall_at_rate": 0.6}, {})

    passing = {
        "test_precision": 0.72,
        "test_recall_at_rate": 0.6,
        "test_flags_without_reason": 0.0,
    }
    assert all(g.passed for g in task.gates(passing, baselines))
    failing = task.gates({**passing, "test_precision": 0.69}, baselines)
    assert [g.name for g in failing if not g.passed] == ["precision"]


def test_baseline_runs_through_the_framework(small_sqlite: Path, tmp_path: Path) -> None:
    tracker = Tracker(f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=tmp_path / "art")
    config = load_experiment(CONFIGS / "00_user_zscore.yaml")
    result = run_experiment(config, small_sqlite, tracker)

    assert 0 <= result.metrics["val_recall_at_rate"] <= 1
    assert result.metrics["val_flags_without_reason"] == 0
    assert leaderboard("unusual_transactions", small_sqlite, tracker) == []  # baselines only


def test_the_budget_leaves_out_only_the_warm_up(
    task: UnusualTransactionsTask, examples: Examples
) -> None:
    f = examples.frame
    assert (f.loc[f["label"] == "warmup", "ts"] < f.loc[f["label"] == "ignored", "ts"].min()).all()
    assert (f["label"] == "ignored").any()  # duplicate originals stay in the budget
