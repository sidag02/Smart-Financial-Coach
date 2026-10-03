"""Retraining from feedback (FR-5 and FR-6 design, §5), through the FR-3 framework's own parts.

Training data for a retraining with cutoff `cutoff`:
- the shipped model's training rows (`TRAIN` of the task's splits), with **a global label
  superseding the original label at its string** (owner, Oct 2, 2026): otherwise a majority
  preference at a known merchant could never move the model;
- the **contributing subjects' own transactions** at globally labelled strings, from **before the
  cutoff**, with the agreed category.

Nobody else's rows enter training: evaluation scores users who supplied no labels, in later
months, and their rows in training would leak into it. `leak_errors` stops a retraining whose
added rows include an evaluation user's transaction or anything after the cutoff.

Like a run, the model is cross-fitted on merchant-grouped folds, calibrated on the pooled
held-out outputs, and gets its review policy from them (§1). Labels are clean (`label_noise: 0`,
decision Oct 2, 2026).

    model, policy = retrain(task, examples, splits, config, labels, contributors, cutoff, ...)
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import pandas as pd

from smart_financial_coach.data.features.merchant_text import normalize_merchant
from smart_financial_coach.evaluation.experiment import ExperimentConfig
from smart_financial_coach.evaluation.runner import cross_fit, ids_in_order
from smart_financial_coach.evaluation.splits import TRAIN, Splits, group_kfold, ids
from smart_financial_coach.evaluation.tasks.base import Examples
from smart_financial_coach.evaluation.tasks.categorization import DEFAULTS, CategorizationTask
from smart_financial_coach.intelligence.categorization.review import (
    POLICY_FILE,
    PolicyError,
    ReviewPolicy,
)
from smart_financial_coach.intelligence.models.base import HeldOutFit
from smart_financial_coach.intelligence.models.contract import Checked
from smart_financial_coach.intelligence.models.registry import build
from smart_financial_coach.intelligence.service import get_service


class RetrainError(ValueError):
    pass


@dataclass(frozen=True)
class Retrained:
    model: Checked
    policy: ReviewPolicy
    relabelled_rows: int  # original training rows whose label a global label replaced
    added_rows: int  # contributors' rows added with the agreed category
    training_rows: int


def leak_errors(
    added: pd.DataFrame, evaluation_users: Collection[str], cutoff: pd.Timestamp
) -> list[str]:
    errors = []
    if added["user_id"].isin(set(evaluation_users)).any():
        errors.append("added rows include an evaluation user's transactions")
    if (pd.to_datetime(added["ts"]) >= cutoff).any():
        errors.append(f"added rows include transactions on or after the cutoff {cutoff.date()}")
    return errors


def retrain(
    task: CategorizationTask,
    examples: Examples,
    splits: Splits,
    config: ExperimentConfig,
    labels: Mapping[str, str],
    contributors: Mapping[str, Collection[str]],
    cutoff: pd.Timestamp,
    evaluation_users: Collection[str],
    version: str,
) -> Retrained:
    """`labels`: merchant_key -> agreed category; `contributors`: merchant_key -> the subjects
    (here, user ids) whose votes agreed. Raises `RetrainError` on a leak or no policy."""
    if float((DEFAULTS | config.task_params)["label_noise"]) != 0:
        raise RetrainError("models retrained from feedback train on clean labels (label_noise 0)")
    frame = examples.frame
    key = frame["merchant_raw"].map(normalize_merchant)
    train = set(splits.sets[TRAIN].tolist())
    in_train = frame["transaction_id"].isin(train)
    agreed = key.map(labels)

    relabel = in_train & agreed.notna()
    voter_of = {(u, k) for k, users in contributors.items() for u in users}
    added = pd.Series(
        [(u, k) in voter_of for u, k in zip(frame["user_id"], key, strict=True)], index=frame.index
    )
    added &= ~in_train & agreed.notna() & (pd.to_datetime(frame["ts"]) < cutoff)
    if errors := leak_errors(frame[added], evaluation_users, cutoff):
        raise RetrainError("; ".join(errors))

    new_labels = examples.labels.where(~(relabel | added), agreed)  # type: ignore[union-attr]
    new_frame = frame.assign(category=new_labels)
    retrained = Examples(
        frame=new_frame,
        labels=new_labels,
        id_column=examples.id_column,
        model_columns=examples.model_columns,
        data_hash=examples.data_hash,
    )
    train_ids = ids(new_frame.loc[in_train | added, "transaction_id"])
    params = DEFAULTS | config.task_params
    rows = new_frame[new_frame["transaction_id"].isin(set(train_ids.tolist()))]
    folds = group_kfold(
        rows,
        group="merchant_id",
        k=int(params["k"]),
        seed=config.seed,
        id_column="transaction_id",
        stratify="category",
        eligible=set(frame.loc[frame["holdout_eligible"] == 1, "merchant_id"]),
        seen_frac=float(params["seen_frac"]),
    )
    new_splits = Splits(sets={TRAIN: train_ids}, folds=folds)

    pooled = cross_fit(task, retrained, new_splits, config, {})
    x, y = task.training_rows(retrained, train_ids, config.task_params, config.seed)
    model = build(config.model_spec({})).fit(x, y)
    model.version = version
    if isinstance(model, HeldOutFit) and pooled.outputs is not None:
        held = retrained.labels_for(ids_in_order(pooled.outputs))
        assert held is not None
        pooled.predictions["confidence"] = model.fit_held_out(
            pooled.outputs.drop(columns="fold"), held, pooled.outputs["fold"].to_numpy()
        )
    files = task.serving_files(
        retrained, pooled.predictions, version, {"retrained_from": f"feedback to {cutoff.date()}"}
    )
    if POLICY_FILE not in files:
        raise RetrainError("the retrained model's predictions can't meet the review rule")
    try:
        policy = ReviewPolicy.from_json(files[POLICY_FILE])
    except (KeyError, ValueError, PolicyError) as error:
        raise RetrainError(f"bad review policy: {error}") from error
    return Retrained(
        model=Checked(model, get_service(task.name).contract),
        policy=policy,
        relabelled_rows=int(relabel.sum()),
        added_rows=int(added.sum()),
        training_rows=len(train_ids),
    )
