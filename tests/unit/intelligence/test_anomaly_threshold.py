import json
import math
from typing import Self

import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.intelligence.anomaly.baseline import UserZScore
from smart_financial_coach.intelligence.anomaly.contract import CONTRACT, AnomalyModel
from smart_financial_coach.intelligence.anomaly.threshold import (
    Thresholded,
    precision_cutoff,
    top_k,
    user_months,
)
from smart_financial_coach.intelligence.models.contract import Checked
from smart_financial_coach.intelligence.models.registry import build


def rows(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [f"t{i}" for i in range(n)],
            "user_id": "u1",
            "ts": [
                (pd.Timestamp("2025-01-01") + pd.Timedelta(days=30 * i)).strftime("%Y-%m-%d %H:%M")
                for i in range(n)
            ],
            "amount": -10.0,
        }
    )


class Fixed(AnomalyModel):
    """Scores the rows 0, 1, 2, ...; every row would be a duplicate. Records what fit saw."""

    name = "test/fixed_scorer"

    def __init__(self) -> None:
        super().__init__()
        self.saw_labels: object = "not fitted"

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self.saw_labels = y
        return self

    def scores(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "transaction_id": x["transaction_id"].to_numpy(),
                "score": np.arange(len(x), dtype=float),
                "reason_code": "duplicate",
                "original_transaction_id": "t0",
                "minutes_apart": np.float64(6.0),
                "amount": 10.0,
            }
        )

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        raise AssertionError("Thresholded uses scores(), not predict()")


def test_precision_cutoff_is_the_deepest_with_the_target() -> None:
    score = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    labels = np.array(["anomaly", "anomaly", "normal", "anomaly", "normal"])

    assert precision_cutoff(score, labels, 0.75) == 2.0  # 3 of the top 4
    assert precision_cutoff(score, labels, 1.0) == 4.0
    assert precision_cutoff(score, np.array(["normal"] * 5), 0.5) == math.inf


def test_precision_cutoff_skips_ignored_rows_and_tied_scores() -> None:
    score = np.array([5.0, 4.0, 4.0, 3.0])
    labels = np.array(["anomaly", "anomaly", "normal", "ignored"])

    # The two 4.0s are flagged together (2 of 3); the ignored 3.0 doesn't count
    assert precision_cutoff(score, labels, 0.6) == 4.0
    assert precision_cutoff(score, labels, 0.7) == 5.0


def test_top_k_breaks_ties_by_id() -> None:
    chosen = top_k(np.array([1.0, 2.0, 2.0, 0.0]), np.array(["d", "c", "b", "a"]), 2)

    assert chosen.tolist() == [False, True, True, False]
    assert not top_k(np.array([1.0]), np.array(["a"]), 0).any()


def test_user_months_sums_each_users_span() -> None:
    x = pd.concat([rows(4), rows(2).assign(user_id="u2")])

    assert user_months(x) == pytest.approx((90 + 30) / 30.4)


def test_thresholded_hides_labels_from_the_scorer() -> None:
    base = Fixed()
    labels = pd.Series(["normal", "normal", "normal", "anomaly", "anomaly", "anomaly"])
    model = Thresholded(base, precision=1.0).fit(rows(), labels)

    assert base.saw_labels is None
    assert model.cutoff == 3.0
    assert model.report["train_precision"] == 1.0

    out = Checked(model, CONTRACT).predict(rows())
    assert out["is_flagged"].tolist() == [False, False, False, True, True, True]
    assert json.loads(str(out.loc[5, "evidence"])) == {
        "original_transaction_id": "t0",
        "minutes_apart": 6.0,
        "amount": 10.0,
    }


def test_thresholded_at_a_rate_needs_no_labels() -> None:
    model = Thresholded(Fixed(), precision=None, rate=0.4).fit(rows())

    out = Checked(model, CONTRACT).predict(rows())  # 150 days, about 4.9 user-months: the top 2
    assert out["is_flagged"].tolist() == [False, False, False, False, True, True]


def test_thresholded_needs_labels_for_a_precision_cutoff() -> None:
    with pytest.raises(ValueError, match="needs labels"):
        Thresholded(Fixed(), precision=0.8).fit(rows())
    with pytest.raises(ValueError, match="exactly one"):
        Thresholded(Fixed(), precision=0.8, rate=0.1)


def test_baseline_is_point_in_time_and_builds_from_a_spec() -> None:
    amounts = [10, 12, 11, 9, 10, 11, 10, 12, 9, 10, 100, 10]
    x = rows(len(amounts)).assign(amount=[-float(a) for a in amounts])
    model = build(
        {
            "type": "unusual_transactions/thresholded",
            "params": {
                "precision": 0.5,
                "base": {"$model": {"type": "unusual_transactions/user_zscore"}},
            },
        }
    )
    labels = pd.Series(["normal"] * 10 + ["anomaly", "normal"])
    out = Checked(model.fit(x, labels), CONTRACT).predict(x)

    assert out["is_flagged"].tolist() == [False] * 10 + [True, False]
    evidence = json.loads(str(out.loc[10, "evidence"]))
    assert evidence["prior_charges"] == 10
    assert evidence["usual_amount"] == pytest.approx(10.4)

    early = UserZScore().scores(x.iloc[:11])
    assert early["score"].tolist() == UserZScore().scores(x)["score"].tolist()[:11]
