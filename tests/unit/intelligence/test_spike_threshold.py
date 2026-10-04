import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from smart_financial_coach.intelligence.models.contract import Checked
from smart_financial_coach.intelligence.models.registry import build
from smart_financial_coach.intelligence.spikes.baseline import MeanKStd
from smart_financial_coach.intelligence.spikes.contract import CONTRACT, INPUT_COLUMNS
from smart_financial_coach.intelligence.spikes.threshold import (
    SpikeScorerModel,
    SpikeThresholded,
    precision_cutoff,
    rate_cutoff,
    user_months,
)


def periods(n: int = 12) -> pd.DataFrame:
    """`n` periods of one user's Dining, one per month, all eligible but the last: its spend is
    under 1.3x usual."""
    rows = pd.DataFrame(dict.fromkeys(INPUT_COLUMNS, 0.0), index=range(n))
    rows["user_id"] = "u1"
    rows["category"] = "Dining"
    rows["month"] = 2025 * 12 + np.arange(n)
    rows["period_start"] = [f"2025-{m + 1:02d}-01" for m in range(n)]
    rows["period_id"] = "u1|Dining|" + rows["period_start"]
    rows["usual"] = 100.0
    rows["usual_months"] = 12
    rows["usual_count"] = 10.0
    rows["count"] = 25
    rows["spend"] = 250.0
    rows.loc[n - 1, "spend"] = 120.0
    rows["history_mean"] = 100.0
    rows["history_sd"] = 10.0
    return rows


class Ranked(SpikeScorerModel):
    """Scores the i-th month of 2025 as i (times `step`, a searchable param); records what each
    fit was given."""

    name = "test/ranked_spikes"
    assumes_poisson = 1.0

    def __init__(self, step: float = 1.0) -> None:
        super().__init__(step=step)
        self.step = step
        self.seen_labels: list[Any] = []

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "Ranked":
        self.seen_labels.append(y)
        return self

    def scores(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        return (x["month"].to_numpy(dtype=float) - 2025 * 12) * self.step


def test_the_cutoff_sees_labels_and_the_scorer_never_does() -> None:
    x = periods()
    labels = pd.Series(["normal"] * 8 + ["spike"] * 4)
    base = Ranked()
    model = SpikeThresholded(base, precision=0.8).fit(x, labels)

    assert base.seen_labels == [None]
    assert model.cutoff == 8.0  # periods 8-10 are spikes; 11 isn't eligible
    out = Checked(model, CONTRACT).predict(x)
    assert out["is_flagged"].tolist() == [False] * 8 + [True] * 3 + [False]
    assert model.report["assumes_poisson"] == 1.0


def test_unscored_labels_never_count_towards_precision() -> None:
    score = np.array([3.0, 2.0, 1.0])
    labels = np.array(["ignored", "spike", "normal"])
    assert precision_cutoff(score, labels, 1.0) == 2.0
    assert precision_cutoff(score, np.array(["normal"] * 3), 0.5) == math.inf


def test_a_rate_cutoff_needs_no_labels_and_is_fixed_after_fitting() -> None:
    x = periods()
    model = SpikeThresholded(Ranked(), precision=None, rate=2 / 12).fit(x)
    assert model.cutoff == 9.0  # the 2 highest eligible scores among 12 user-months: 10 and 9
    # One user's month alone is judged by the same cutoff, not re-ranked within the batch
    alone = Checked(model, CONTRACT).predict(x.iloc[[9]].reset_index(drop=True))
    assert alone["is_flagged"].tolist() == [True]
    assert rate_cutoff(np.array([-np.inf]), np.array(["a"]), 1) == math.inf


def test_the_search_fits_each_point_without_labels() -> None:
    x = periods()
    labels = pd.Series(["normal"] * 8 + ["spike"] * 4)
    model = SpikeThresholded(
        Ranked(), precision=0.8, search={"step": [1.0, -1.0]}, search_rate=4 / 12
    ).fit(x, labels)

    assert model.chosen == {"step": 1.0}
    assert isinstance(model.base, Ranked)
    assert model.base.seen_labels == [None]
    assert model.report["search_recall.step_1.0"] > model.report["search_recall.step_-1.0"]


def test_user_months_count_distinct_months() -> None:
    x = pd.concat([periods(), periods().assign(category="Shopping")])
    assert user_months(x) == 12


def test_mean_k_std_scores_the_z_and_skips_months_without_a_spread() -> None:
    x = periods(3)
    x.loc[2, "history_sd"] = 0.0
    assert MeanKStd().scores(x).tolist() == [15.0, 15.0, -np.inf]


def test_the_baseline_builds_from_its_config() -> None:
    spec = {
        "type": "spending_spikes/thresholded",
        "params": {"precision": 0.8, "base": {"$model": {"type": "spending_spikes/mean_k_std"}}},
    }
    model = build(spec)
    assert isinstance(model, SpikeThresholded)
    assert isinstance(model.base, MeanKStd)
    with pytest.raises(ValueError, match="exactly one"):
        SpikeThresholded(MeanKStd(), precision=0.8, rate=0.1)
