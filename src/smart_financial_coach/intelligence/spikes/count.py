"""Count candidates for spending spikes (FR-8 §3): how improbable is this month's purchase count?

A spike multiplies how often a user buys, not what they pay (FR-8 Feasibility), so the score is
the upper tail of the month's count against an expected count:

    expected = usual_count x season x income_ratio^β

- `usual_count`: the user's average month in the category over the previous 12 months.
- `season`: exp(w · own + (1 - w) · profile), w = own_years / (own_years + κ): the user's own
  season a year earlier, shrunk toward the category season profile (§2). κ is tuned within folds.
- `income_ratio^β`: spending that follows the previous two months' income. β is pooled over the
  training rows by least squares, without labels and without a persona.

The tail is Poisson, or negative binomial with over-dispersion from the user's trailing count
variance shrunk toward a pooled value (fitted without labels). The score is -log P(count ≥ k),
so one cutoff is one knob for FR-9's sensitivity later.

`season=False` and `income=False` give the ablations; both off is the simple count rule
(decision 5, reported, never gated).
"""

from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import nbinom, poisson

from smart_financial_coach.intelligence.models.registry import register
from smart_financial_coach.intelligence.spikes.threshold import SpikeScorerModel

MIN_RATE = 0.3  # an expected count below this is treated as 0.3 (the POC's floor)
INCOME_CLIP = (0.5, 1.6)  # income ratios outside this are clipped: one bonus isn't a new level
BETA_MIN_COUNT = 3.0  # β is fitted on periods whose usual month has at least this many purchases
DISPERSION_PRIOR_MONTHS = 12.0  # a user's own variance counts as much as the pool after a year
DISPERSIONS = ("poisson", "negbin")


@register("spending_spikes/count")
class CountScorer(SpikeScorerModel):
    def __init__(
        self,
        dispersion: str = "poisson",
        season: bool = True,
        income: bool = True,
        kappa: float = 1.0,
    ) -> None:
        if dispersion not in DISPERSIONS:
            raise ValueError(f"dispersion must be one of {DISPERSIONS}, got {dispersion!r}")
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        super().__init__(dispersion=dispersion, season=season, income=income, kappa=kappa)
        self.dispersion = dispersion
        self.use_season = season
        self.use_income = income
        self.kappa = kappa
        self.beta = 0.0
        self.pooled_dispersion = 0.0  # 1/size of the negative binomial; 0 is Poisson
        self.report = {"assumes_poisson": 1.0 if dispersion == "poisson" else 0.0}

    def _season(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        if not self.use_season:
            return np.ones(len(x))
        years = x["own_years"].to_numpy(dtype=float)
        w = years / (years + self.kappa)
        own = np.nan_to_num(x["own_season"].to_numpy(dtype=float), nan=0.0)
        log_season = w * own + (1 - w) * x["profile_season"].to_numpy(dtype=float)
        season: npt.NDArray[np.float64] = np.exp(log_season)
        return season

    def _income(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        ratio = np.nan_to_num(x["income_ratio"].to_numpy(dtype=float), nan=1.0)
        clipped: npt.NDArray[np.float64] = np.clip(ratio, *INCOME_CLIP)
        return clipped

    def _expected(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        rate = np.clip(x["usual_count"].to_numpy(dtype=float) * self._season(x), MIN_RATE, None)
        if self.use_income:
            rate = rate * self._income(x) ** self.beta
        clipped: npt.NDArray[np.float64] = np.clip(rate, MIN_RATE, None)
        return clipped

    def _own_dispersion(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Each row's own estimate of 1/size: (variance - mean) / mean², at least 0."""
        mean = x["usual_count"].to_numpy(dtype=float)
        var = x["count_var"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            own = (var - mean) / mean**2
        clipped: npt.NDArray[np.float64] = np.clip(np.nan_to_num(own, nan=0.0, posinf=0.0), 0, None)
        return clipped

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """β and the pooled dispersion from the training rows; never the labels."""
        if self.use_income:
            usual = x["usual_count"].to_numpy(dtype=float)
            ratio = x["income_ratio"].to_numpy(dtype=float)
            ok = (usual >= BETA_MIN_COUNT) & (ratio >= INCOME_CLIP[0]) & (ratio <= INCOME_CLIP[1])
            if ok.sum() > 1:
                target = (x["count"].to_numpy(dtype=float)[ok] + 0.5) / (
                    usual[ok] * self._season(x)[ok]
                )
                self.beta = float(np.polyfit(np.log(ratio[ok]), np.log(target), 1)[0])
        if self.dispersion == "negbin":
            enough = x["usual_count"].to_numpy(dtype=float) >= BETA_MIN_COUNT
            own = self._own_dispersion(x)[enough]
            self.pooled_dispersion = float(np.median(own)) if len(own) else 0.0
        self.report |= {"beta": self.beta, "pooled_dispersion": self.pooled_dispersion}
        return self

    def scores(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        expected = self._expected(x)
        k = x["count"].to_numpy(dtype=float)
        if self.dispersion == "poisson":
            tail = poisson.logsf(k - 1, expected)
        else:
            months = np.clip(x["usual_months"].to_numpy(dtype=float) - 1, 0, None)
            w = months / (months + DISPERSION_PRIOR_MONTHS)
            alpha = w * self._own_dispersion(x) + (1 - w) * self.pooled_dispersion
            alpha = np.clip(alpha, 1e-9, None)  # 1/size; tiny is Poisson
            size = 1.0 / alpha
            tail = nbinom.logsf(k - 1, size, size / (size + expected))
        # A tail of 0 (log -inf) is certain: +inf. A score that can't be computed can't flag
        score: npt.NDArray[np.float64] = -np.asarray(tail, dtype=np.float64)
        return np.nan_to_num(score, nan=-np.inf, posinf=np.inf)
