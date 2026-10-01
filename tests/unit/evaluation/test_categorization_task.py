"""The categorization task on the small dataset: splits, label noise, metrics, ties, gates."""

from pathlib import Path

import numpy as np
import pytest

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.evaluation.experiment import experiment_files, load_experiment
from smart_financial_coach.evaluation.promote import leaderboard
from smart_financial_coach.evaluation.runner import cross_fit, run_experiment
from smart_financial_coach.evaluation.splits import TRAIN, UNSEEN
from smart_financial_coach.evaluation.tasks.base import Examples
from smart_financial_coach.evaluation.tasks.categorization import CategorizationTask
from smart_financial_coach.evaluation.tracking import Tracker
from smart_financial_coach.intelligence.categorization.contract import CONTRACT
from smart_financial_coach.intelligence.models import Checked, build

BASELINES = PROJECT_ROOT / "configs" / "experiments" / "categorization"


@pytest.fixture(scope="module")
def task() -> CategorizationTask:
    return CategorizationTask(reps=200)


@pytest.fixture(scope="module")
def examples(task: CategorizationTask, small_sqlite: Path) -> Examples:
    return task.load(small_sqlite)


def test_splits(task: CategorizationTask, examples: Examples) -> None:
    splits = task.split(examples, {}, 0)
    f = examples.frame.set_index("transaction_id")
    train_users = (f["split"] == "train").sum()
    eligible = set(f.loc[f["holdout_eligible"] == 1, "merchant_id"])
    held = [set(f.loc[fold.held_out[UNSEEN].tolist(), "merchant_id"]) for fold in splits.folds]
    in_train = set(f.loc[splits.sets[TRAIN].tolist(), "merchant_id"])

    assert task.leak_errors(examples, splits) == []
    assert len(splits.sets["test_known"]) == pytest.approx(0.2 * train_users, rel=0.01)
    assert len(splits.sets["test_unseen"]) > 0
    assert len(splits.folds) == 5
    assert sorted(m for h in held for m in h) == sorted(in_train & eligible)  # each once
    assert splits.hash() == task.split(examples, {}, 0).hash()


def test_label_noise_flips_spending_labels_only(
    task: CategorizationTask, examples: Examples
) -> None:
    which = task.split(examples, {}, 0).sets[TRAIN]
    _, clean = task.training_rows(examples, which, {"label_noise": 0.0}, 0)
    _, noisy = task.training_rows(examples, which, {"label_noise": 0.05}, 0)
    _, again = task.training_rows(examples, which, {"label_noise": 0.05}, 0)
    assert clean is not None
    assert noisy is not None
    assert again is not None
    flipped = clean != noisy
    spending = clean != "Income"

    assert flipped.sum() == round(0.05 * spending.sum())
    assert not (flipped & ~spending).any()  # Income never flipped
    assert (noisy[flipped] != "Income").all()  # never flipped to Income
    assert noisy.equals(again)
    original = examples.labels_for(which)
    assert original is not None
    assert original.equals(clean)  # the examples' own labels are untouched


def test_test_metrics(task: CategorizationTask, examples: Examples) -> None:
    splits = task.split(examples, {}, 0)
    x, y = task.training_rows(examples, splits.sets[TRAIN], {}, 0)
    model = Checked(build({"type": "categorization/lookup"}).fit(x, y), CONTRACT)

    out = task.test_metrics(examples, splits, model)

    assert out["known_macro_f1"] > 0.9  # lookup memorizes known merchants
    assert out["unseen_macro_f1"] < 0.3  # and knows nothing about new ones
    assert out["unseen_macro_f1_lo"] <= out["unseen_macro_f1"] <= out["unseen_macro_f1_hi"]
    assert 0 <= out["known_ece"] <= 1
    assert {"known_f1.groceries", "known_oracle_f1.shopping", "refund_accuracy"} <= set(out)


def test_ties(task: CategorizationTask, examples: Examples) -> None:
    splits = task.split(examples, {}, 0)
    configs = {c.name: c for c in map(load_experiment, experiment_files([BASELINES]))}
    lookup = cross_fit(task, examples, splits, configs["lookup"], {})
    keyword = cross_fit(task, examples, splits, configs["keyword"], {})

    assert task.tied(examples, lookup, lookup)
    assert not task.tied(examples, keyword, lookup)


def test_gates_and_eligibility(task: CategorizationTask) -> None:
    baselines = {"keyword": {"val_known_macro_f1": 0.5, "test_known_macro_f1": 0.5}}
    good = {"test_known_macro_f1": 0.95, "latency_p95_ms": 1.0, "val_known_macro_f1": 0.95}

    assert all(g.passed for g in task.gates(good, baselines))
    assert task.eligible(good, baselines)
    assert not task.eligible({"val_known_macro_f1": 0.89}, baselines)
    slow = task.gates(good | {"latency_p95_ms": 9.0}, baselines)
    assert [g.name for g in slow if not g.passed] == ["latency_p95"]
    assert not task.gates(good, {})[1].passed  # no keyword baseline: the gate can't pass


def test_round_zero_runs_end_to_end(small_sqlite: Path, tmp_path: Path) -> None:
    tracker = Tracker(f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=tmp_path / "art")
    configs = [load_experiment(p) for p in experiment_files([BASELINES])]
    results = {c.name: run_experiment(c, small_sqlite, tracker) for c in configs}

    assert set(results) == {"majority", "keyword", "lookup"}
    assert results["lookup"].metrics["val_known_macro_f1"] > 0.9
    assert results["keyword"].metrics["val_known_macro_f1"] < 0.9
    assert np.isfinite(results["majority"].metrics["latency_p95_ms"])
    assert leaderboard("categorization", small_sqlite, tracker) == []  # baselines aren't candidates
