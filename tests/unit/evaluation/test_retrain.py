"""Retraining from feedback (#15 §5): supersede, contributors only, the leak check, clean labels."""

from pathlib import Path

import pandas as pd
import pytest

from smart_financial_coach.data.features.merchant_text import normalize_merchant
from smart_financial_coach.evaluation.experiment import ExperimentConfig
from smart_financial_coach.evaluation.retrain import RetrainError, leak_errors, retrain
from smart_financial_coach.evaluation.splits import TRAIN
from smart_financial_coach.evaluation.tasks.categorization import CategorizationTask

STUB = {
    "name": "retrain_stub",
    "task": "categorization",
    "model": {
        "type": "categorization/calibrated",
        "params": {
            "base": {
                "$model": {
                    "type": "categorization/linear_text",
                    "params": {"embeddings": "stub:16"},
                }
            }
        },
    },
    "task_params": {"k": 3, "label_noise": 0},
}
CUTOFF = pd.Timestamp("2025-06-01")


@pytest.fixture(scope="module")
def setup(small_sqlite: Path):  # type: ignore[no-untyped-def]
    task = CategorizationTask()
    examples = task.load(small_sqlite)
    config = ExperimentConfig.model_validate(STUB)
    splits = task.split(examples, config.task_params, config.seed)
    return task, examples, splits, config


def test_a_retraining_needs_clean_labels(setup) -> None:  # type: ignore[no-untyped-def]
    task, examples, splits, config = setup
    noisy = config.model_copy(update={"task_params": {"k": 3, "label_noise": 0.02}})
    with pytest.raises(RetrainError, match="clean labels"):
        retrain(task, examples, splits, noisy, {}, {}, CUTOFF, [], version="v")


def test_the_leak_check() -> None:
    rows = pd.DataFrame({"user_id": ["u_tr_1", "u_te_2"], "ts": ["2025-01-01", "2025-07-01"]})
    assert leak_errors(rows[:1], rows[:0], ["u_te_2"], CUTOFF) == []
    assert "evaluation user" in leak_errors(rows, rows[:0], ["u_te_2"], CUTOFF)[0]
    assert "after the cutoff" in leak_errors(rows[:1], rows[1:], [], CUTOFF)[0]


@pytest.mark.slow  # cross-fits a small model
def test_a_label_supersedes_training_rows_and_adds_only_contributors_rows(setup) -> None:  # type: ignore[no-untyped-def]
    task, examples, splits, config = setup
    f = examples.frame.assign(key=examples.frame["merchant_raw"].map(normalize_merchant))
    train = f[f["transaction_id"].isin(set(splits.sets[TRAIN].tolist()))]
    test = f[f["split"] == "test"]
    key = next(k for k in test["key"].value_counts().index if k in set(train["key"]))
    users = sorted(test.loc[test["key"] == key, "user_id"].unique())
    contributor, evaluation = users[0], users[1:]
    label = "Travel"

    done = retrain(
        task, examples, splits, config, {key: label}, {key: [contributor]}, CUTOFF, evaluation,
        version="v-feedback",
    )  # fmt: skip

    assert done.relabelled_rows == int((train["key"] == key).sum())
    mine = test[(test["key"] == key) & (test["user_id"] == contributor)]
    assert done.added_rows == int((pd.to_datetime(mine["ts"]) < CUTOFF).sum())
    assert done.training_rows == len(train) + done.added_rows
    assert done.model.version == "v-feedback"
    assert done.policy.model_version == "v-feedback"
    revoked = retrain(task, examples, splits, config, {}, {}, CUTOFF, evaluation, version="v2")
    assert (revoked.relabelled_rows, revoked.added_rows) == (0, 0)  # original labels return
    with pytest.raises(RetrainError, match="evaluation user"):
        retrain(
            task, examples, splits, config, {key: label}, {key: [contributor]}, CUTOFF,
            [contributor], version="v3",
        )  # fmt: skip
