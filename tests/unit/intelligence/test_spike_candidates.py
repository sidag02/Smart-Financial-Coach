from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.evaluation.experiment import load_experiment
from smart_financial_coach.intelligence.models.registry import build
from smart_financial_coach.intelligence.spikes.contract import INPUT_COLUMNS
from smart_financial_coach.intelligence.spikes.count import CountScorer
from smart_financial_coach.intelligence.spikes.threshold import SpikeThresholded

CONFIGS = PROJECT_ROOT / "configs" / "experiments" / "spending_spikes"


def periods(count: list[int], usual: float = 10.0, **columns: Any) -> pd.DataFrame:
    rows = pd.DataFrame(dict.fromkeys(INPUT_COLUMNS, 0.0), index=range(len(count)))
    rows["period_id"] = [f"u{i}|Dining|2025-06-01" for i in range(len(count))]
    rows["count"] = count
    rows["usual_count"] = usual
    rows["usual_months"] = 12
    rows["count_var"] = usual  # Poisson-like: variance equals the mean
    rows["income_ratio"] = 1.0
    for name, value in columns.items():
        rows[name] = value
    return rows


def test_the_score_is_the_poisson_tail_of_the_count() -> None:
    x = periods([10, 20, 30])
    score = CountScorer(season=False, income=False).fit(x).scores(x)
    np.testing.assert_allclose(score, -poisson.logsf(np.array([10, 20, 30]) - 1, 10.0))
    assert score[0] < score[1] < score[2]


def test_the_season_shrinks_own_toward_the_profile() -> None:
    profile = np.log(2.0)
    x = periods([20, 20], profile_season=profile, own_season=[np.nan, 0.0], own_years=[0, 1])
    score = CountScorer(income=False, kappa=1.0).scores(x)
    # No own season: the profile's 2x alone, so 20 is the expected count. One year of an
    # ordinary month (log 0) halves the profile: the season is sqrt(2)
    np.testing.assert_allclose(score[0], -poisson.logsf(19, 20.0))
    np.testing.assert_allclose(score[1], -poisson.logsf(19, 10.0 * np.sqrt(2.0)))


def test_beta_is_fitted_without_labels_and_moves_the_expectation() -> None:
    rng = np.random.default_rng(0)
    ratio = rng.uniform(0.6, 1.5, 4000)
    count = rng.poisson(10.0 * ratio**0.5)
    x = periods(list(count), income_ratio=ratio)
    model = CountScorer(season=False).fit(x, y=pd.Series(["spike"] * len(x)))

    assert model.beta == pytest.approx(0.5, abs=0.1)
    assert model.report["beta"] == model.beta
    flat = CountScorer(season=False).fit(periods(list(rng.poisson(10.0, 4000)), income_ratio=ratio))
    assert flat.beta == pytest.approx(0.0, abs=0.1)


def test_the_negative_binomial_is_more_forgiving_of_a_variable_user() -> None:
    x = periods([25], count_var=60.0)  # variance six times the mean
    poisson_score = CountScorer(season=False, income=False).fit(x).scores(x)
    negbin = CountScorer(dispersion="negbin", season=False, income=False).fit(x)
    assert negbin.pooled_dispersion == pytest.approx((60 - 10) / 100)
    assert negbin.scores(x)[0] < poisson_score[0]
    # Without over-dispersion it is the Poisson
    even = periods([25])
    np.testing.assert_allclose(
        CountScorer(dispersion="negbin", season=False, income=False).fit(even).scores(even),
        CountScorer(season=False, income=False).scores(even),
        rtol=1e-4,
    )


def test_assumes_poisson_marks_the_tie_break() -> None:
    assert CountScorer().report["assumes_poisson"] == 1.0
    assert CountScorer(dispersion="negbin").report["assumes_poisson"] == 0.0
    with pytest.raises(ValueError, match="dispersion"):
        CountScorer(dispersion="gamma")


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_round_config_builds(path: object) -> None:
    config = load_experiment(path)  # type: ignore[arg-type]
    model = build(config.model_spec({}))
    assert isinstance(model, SpikeThresholded)
    if config.name == "simple_count":  # decision 12's fallback: a rate cutoff, no labels
        assert (model.precision, model.rate) == (None, 0.035)
    else:
        assert model.precision == 0.80
    if config.tags.get("report_only") == "true":
        assert not config.baseline
