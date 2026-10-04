from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.evaluation.experiment import load_experiment
from smart_financial_coach.evaluation.promote import leaderboard
from smart_financial_coach.evaluation.runner import run_experiment
from smart_financial_coach.evaluation.splits import TRAIN, UNSEEN
from smart_financial_coach.evaluation.tasks.base import Examples, get_task
from smart_financial_coach.evaluation.tasks.spikes import (
    BASKET_RANGE,
    FLAG_RATE,
    TEST,
    SpendingSpikesTask,
)
from smart_financial_coach.evaluation.tracking import Tracker
from smart_financial_coach.intelligence.spikes.contract import INPUT_COLUMNS, SpikeModel
from smart_financial_coach.intelligence.spikes.threshold import user_months

CONFIGS = PROJECT_ROOT / "configs" / "experiments" / "spending_spikes"


@pytest.fixture(scope="module")
def task() -> SpendingSpikesTask:
    return SpendingSpikesTask(reps=200)


@pytest.fixture(scope="module")
def examples(task: SpendingSpikesTask, small_sqlite: Path) -> Examples:
    return task.load(small_sqlite)


def predictions(examples: Examples, which: np.ndarray, flag: str = "spike") -> pd.DataFrame:
    """Hand-made predictions over `which`: a perfect score, flags on eligible rows labelled
    `flag`, with evidence wherever a row is flagged."""
    f = examples.frame.set_index("period_id").loc[which.tolist()]
    eligible = SpikeModel.eligible(f.reset_index())
    flagged = (f["label"] == flag).to_numpy() & eligible
    return pd.DataFrame(
        {
            "period_id": f.index.to_numpy(),
            "score": (f["label"] == "spike").to_numpy(dtype=float),
            "is_flagged": flagged,
            "evidence": ["{}" if f else None for f in flagged],
            "fold": 0,
        }
    )


def test_the_task_is_registered() -> None:
    assert isinstance(get_task("spending_spikes"), SpendingSpikesTask)


def test_examples_are_periods_after_the_warm_up(
    task: SpendingSpikesTask, examples: Examples
) -> None:
    f = examples.frame
    assert task.truth is not None
    real = f[f["label"] != "basket"]

    assert set(f["label"]) == {"spike", "normal", "ignored", "basket"}
    assert (real["period_start"] >= task.truth.warmup_end_month).all()
    assert real["period_id"].is_unique
    assert "Income" not in set(f["category"])
    assert examples.model_columns == INPUT_COLUMNS
    # Every planted monthly spike after the warm-up is an example
    planted = task.truth.spikes("month")
    keys = set(zip(real["user_id"], real["category"], real["period_start"], strict=True))
    assert all(
        k in keys
        for k in zip(planted["user_id"], planted["category"], planted["period_start"], strict=True)
    )


def test_basket_rows_change_spend_only(examples: Examples) -> None:
    f = examples.frame.set_index("period_id")
    baskets = f[f["label"] == "basket"]
    originals = f.loc[[i.removesuffix("|basket") for i in baskets.index]]

    assert len(baskets) > 0
    ratio = baskets["spend"].to_numpy() / originals["spend"].to_numpy()
    assert ((ratio >= BASKET_RANGE[0]) & (ratio <= BASKET_RANGE[1])).all()
    same = [c for c in INPUT_COLUMNS if c not in ("period_id", "spend")]
    pd.testing.assert_frame_equal(
        baskets[same].reset_index(drop=True), originals[same].reset_index(drop=True)
    )
    assert (originals["label"] == "normal").all()


def test_splits_hold_out_whole_users(task: SpendingSpikesTask, examples: Examples) -> None:
    splits = task.split(examples, {}, seed=0)
    f = examples.frame.set_index("period_id")

    assert task.leak_errors(examples, splits) == []
    assert len(splits.folds) == 5
    for fold in splits.folds:
        held = set(f.loc[fold.held_out[UNSEEN].tolist(), "user_id"])
        assert not held & set(f.loc[fold.train.tolist(), "user_id"])
    assert set(f.loc[splits.sets[TEST].tolist(), "split"]) == {"test"}


def test_basket_rows_are_never_trained_on(task: SpendingSpikesTask, examples: Examples) -> None:
    splits = task.split(examples, {}, seed=0)
    x, y = task.training_rows(examples, splits.sets[TRAIN], {}, seed=0)
    assert y is not None
    assert "basket" not in set(y)
    assert not x["period_id"].str.endswith("|basket").any()


def test_leak_check_catches_test_users_in_validation_profiles(
    task: SpendingSpikesTask, examples: Examples
) -> None:
    frame = examples.frame.copy()
    train = frame["split"] == "train"
    frame.loc[train, "profile_users"] = frame.loc[train, "profile_users"] + 1  # as if one more
    leaky = Examples(frame, examples.labels, "period_id", examples.model_columns, "x")

    errors = task.leak_errors(leaky, task.split(leaky, {}, seed=0))
    assert any("isn't built from train users alone" in e for e in errors)


def test_leak_check_catches_profiles_from_the_wrong_pool(
    task: SpendingSpikesTask, examples: Examples
) -> None:
    frame = examples.frame.copy()
    frame.loc[frame["split"] == "train", "season_pool"] = "all"
    leaky = Examples(frame, examples.labels, "period_id", examples.model_columns, "x")

    errors = task.leak_errors(leaky, task.split(leaky, {}, seed=0))
    assert any("wrong pool" in e for e in errors)


def test_perfect_predictions_score_perfectly(task: SpendingSpikesTask, examples: Examples) -> None:
    train = task.split(examples, {}, seed=0).sets[TRAIN]
    m = task.validation_metrics(examples, predictions(examples, train))
    rows = examples.frame.set_index("period_id").loc[train.tolist()].reset_index()
    spikes = rows[rows["label"] == "spike"]
    allowed = SpikeModel.eligible(spikes)

    assert m["precision"] == 1.0
    assert m["recall"] == pytest.approx(allowed.mean())  # the product rules exclude some labels
    assert m["flags_without_evidence"] == 0
    assert 0 < m["excess_coverage"] <= 1
    assert m["usual_error"] >= 0
    # At the common rate, a perfect scorer finds what fits in the budget, among allowed periods
    budget = round(FLAG_RATE * user_months(rows[rows["label"] != "basket"]))
    assert m["recall_at_rate"] == pytest.approx(min(budget, allowed.sum()) / len(spikes))
    assert m["basket_recall"] == 0.0


def test_flags_on_normal_months_are_false_positives(
    task: SpendingSpikesTask, examples: Examples
) -> None:
    train = task.split(examples, {}, seed=0).sets[TRAIN]
    m = task.validation_metrics(examples, predictions(examples, train, flag="normal"))

    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["fp"] > 0


def test_ties_and_intervals(task: SpendingSpikesTask, examples: Examples) -> None:
    train = task.split(examples, {}, seed=0).sets[TRAIN]
    good = predictions(examples, train)
    noise = good.assign(score=np.random.default_rng(0).random(len(good)))

    assert task.tied(examples, good, good)
    assert not task.tied(examples, good, noise)
    lo, hi = task.selection_interval(examples, good)
    assert lo <= hi <= 1.0
    assert task.tiebreak_tied(examples, good, good)


def test_eligibility_and_gates(task: SpendingSpikesTask) -> None:
    baselines = {"mean_k_std": {"val_recall_at_rate": 0.1, "test_recall_at_rate": 0.1}}

    assert task.eligible({"val_precision": 0.75, "val_recall_at_rate": 0.5}, baselines)
    assert not task.eligible({"val_precision": 0.65, "val_recall_at_rate": 0.5}, baselines)
    assert not task.eligible({"val_precision": 0.75, "val_recall_at_rate": 0.05}, baselines)
    assert not task.eligible({"val_precision": 0.75, "val_recall_at_rate": 0.5}, {})

    passing = {
        "test_precision": 0.72,
        "test_recall_at_rate": 0.5,
        "test_flags_without_evidence": 0.0,
    }
    assert all(g.passed for g in task.gates(passing, baselines))
    failing = task.gates({**passing, "test_precision": 0.69}, baselines)
    assert [g.name for g in failing if not g.passed] == ["precision"]


def test_the_tie_breaks_follow_decision_6(task: SpendingSpikesTask) -> None:
    assert task.tiebreak_metrics == (
        "val_cutoff_shortfall",
        "fit.assumes_poisson",
        "latency_batch_ms",
    )
    assert task.shipping_params == ()


def test_baseline_runs_through_the_framework(small_sqlite: Path, tmp_path: Path) -> None:
    tracker = Tracker(f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=tmp_path / "art")
    config = load_experiment(CONFIGS / "00_mean_k_std.yaml")
    result = run_experiment(config, small_sqlite, tracker)

    assert 0 <= result.metrics["val_recall_at_rate"] <= 1
    assert result.metrics["val_flags_without_evidence"] == 0
    assert result.metrics["fit.assumes_poisson"] == 0.0
    assert leaderboard("spending_spikes", small_sqlite, tracker) == []  # baselines only
