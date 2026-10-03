"""The Probabilistic candidate (FR-7 §3): one predictive distribution per (user, merchant).

A Student-t on log amount. Its center and spread start at the merchant profile (`prior_weight`
pseudo-charges) and move towards the user's own history there as it grows:

    center = (k0 * typical + n * user_median) / (k0 + n)
    spread = sqrt((k0 * profile_spread^2 + n * user_spread^2) / (k0 + n)), at least `min_spread`
    df     = k0 + n, at most `max_df`

with k0 = `prior_weight` and n the user's earlier charges there. Without a profile, the prior is
the user's own history alone (no first-visit score). The score is the upper-tail surprise,
-log10 P(T > t), so every kind shares one scale and one cutoff (the FR-9 sensitivity is one
knob). An exact repeat within the window scores +inf. The reason is `new_merchant` on a first
visit of at least `min_amount`, `amount_unusual` otherwise.
"""

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from smart_financial_coach.intelligence.anomaly.features import (
    DUPLICATE_MINUTES,
    MIN_SPREAD,
    scores_frame,
)
from smart_financial_coach.intelligence.anomaly.threshold import ScorerModel
from smart_financial_coach.intelligence.models.registry import register

DEFAULT_PROFILE_SPREAD = 0.5  # a profile with a typical price but no spread yet


@register("unusual_transactions/probabilistic")
class Probabilistic(ScorerModel):
    def __init__(
        self,
        prior_weight: float = 3.0,
        max_df: float = 30.0,
        min_amount: float = 0.0,
        duplicate_minutes: float = DUPLICATE_MINUTES,
        min_spread: float = MIN_SPREAD,
    ) -> None:
        if prior_weight <= 0 or max_df < 1:
            raise ValueError("prior_weight must be positive and max_df at least 1")
        super().__init__(
            prior_weight=prior_weight,
            max_df=max_df,
            min_amount=min_amount,
            duplicate_minutes=duplicate_minutes,
            min_spread=min_spread,
        )
        self.prior_weight = prior_weight
        self.max_df = max_df
        self.min_amount = min_amount
        self.duplicate_minutes = duplicate_minutes
        self.min_spread = min_spread

    def scores_from(self, f: pd.DataFrame) -> pd.DataFrame:
        n = f["key_prior"].to_numpy(dtype=float)
        user_median = f["key_median"].to_numpy(dtype=float)
        user_spread = f["key_spread"].fillna(0.0).to_numpy(dtype=float)
        typical = f["profile_typical"].to_numpy(dtype=float)
        has_profile = ~np.isnan(typical)
        profile_spread = f["profile_spread"].fillna(DEFAULT_PROFILE_SPREAD).to_numpy(dtype=float)

        k0 = np.where(has_profile, self.prior_weight, 0.0)
        weight = k0 + n
        center = (np.nan_to_num(k0 * typical) + np.nan_to_num(n * user_median)) / np.where(
            weight > 0, weight, 1.0
        )
        variance = (k0 * profile_spread**2 + n * user_spread**2) / np.where(weight > 0, weight, 1.0)
        spread = np.maximum(np.sqrt(variance), self.min_spread)
        df = np.clip(weight, 1.0, self.max_df)
        t = (f["log_amount"].to_numpy() - center) / spread
        surprise = -student_t.logsf(t, df) / np.log(10)
        judged = weight > 0
        score = np.where(judged, surprise, np.finfo(float).min)

        first = f["first_visit"].to_numpy()
        new = first & (f["amount"].to_numpy() >= self.min_amount)
        reason = np.where(new, "new_merchant", "amount_unusual").astype(object)
        reason[~judged | (first & ~new)] = None
        score = np.where(first & ~new, np.finfo(float).min, score)
        repeat = f["repeat"].to_numpy()
        score = np.where(repeat, np.inf, score)
        reason[repeat] = "duplicate"
        return scores_frame(f, score, reason)
