"""Classification metrics against sklearn and hand-worked cases."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

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

LABELS = ["a", "b", "c"]


def test_macro_f1_matches_sklearn_with_fixed_labels() -> None:
    rng = np.random.default_rng(0)
    truth = pd.Series(rng.choice(LABELS, 500))
    pred = pd.Series(rng.choice([*LABELS, "Income"], 500))  # predictions outside the label set

    expected = f1_score(truth, pred, labels=LABELS, average="macro")

    assert macro_f1(truth, pred, LABELS) == pytest.approx(expected)
    assert per_class_f1(truth, pred, LABELS) == pytest.approx(
        dict(zip(LABELS, f1_score(truth, pred, labels=LABELS, average=None), strict=True))
    )


def test_macro_f1_ignores_rows_outside_the_label_set() -> None:
    truth = pd.Series(["a", "b", "Income"])
    pred = pd.Series(["a", "b", "a"])  # the Income row is not a spending row: not counted

    assert macro_f1(truth, pred, LABELS) == 1.0


def test_calibration() -> None:
    truth = pd.Series(["a", "a", "a", "a"])
    pred = pd.Series(["a", "a", "b", "b"])
    conf = pd.Series([0.95, 0.95, 0.95, 0.95])  # 50% accurate at 0.95: ECE 0.45

    out = calibration(truth, pred, conf)

    assert out["ece"] == pytest.approx(0.45)
    assert out["brier"] == pytest.approx((2 * 0.05**2 + 2 * 0.95**2) / 4)
    assert out["acc_at_90"] == 0.5
    assert out["coverage_at_90"] == 1.0


def test_misallocated_spend_weighs_dollars() -> None:
    truth = pd.Series(["Housing", "Dining", "Dining"])
    pred = pd.Series(["Housing", "Dining", "Shopping"])

    assert misallocated_spend(truth, pred, pd.Series([-2000.0, -4.0, -6.0])) == pytest.approx(
        6 / 2010
    )


def test_bootstrap_resamples_groups_within_their_label() -> None:
    truth = pd.Series(["a"] * 4 + ["b"] * 4)
    pred = pd.Series(["a", "a", "b", "b", "b", "b", "b", "b"])
    groups = pd.Series(["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"])
    names, stack, majority = group_confusions(truth, pred, groups, ["a", "b"])
    weights = bootstrap_weights(majority, 200, np.random.default_rng(0))

    samples = bootstrap_macro_f1(stack, weights, 2)
    lo, hi = interval(samples)

    assert names == ["g1", "g2", "g3", "g4"]
    assert (weights[:, :2].sum(axis=1) == 2).all()  # two "a" merchants drawn per resample
    assert samples.max() == 1.0  # g1 drawn twice: perfect
    assert lo < macro_f1(truth, pred, ["a", "b"]) < hi or lo == hi
