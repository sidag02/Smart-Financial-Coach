"""The Rules candidate (FR-7 §3): one rule per kind of unusual charge, on one score scale.

- **Duplicate:** an exact repeat within the window scores +inf, so it's flagged at any cutoff.
- **New merchant:** a first visit of at least `min_amount`, scored by how far its log amount is
  above the merchant profile's typical one, in units of log(`ratio_scale`).
- **Unusual amount:** a charge at a merchant the user has used before, scored by its z against the
  user's history there (spread borrowed from the profile), in units of `z_scale`.

The score is the largest component, and the reason is that component's. Only the scales' ratio
and `min_amount` matter once `Thresholded` places the cutoff; they're chosen per fold through
`Thresholded`'s search, never from labels here.
"""

import numpy as np
import pandas as pd

from smart_financial_coach.intelligence.anomaly.features import (
    DUPLICATE_MINUTES,
    MIN_SPREAD,
    scores_frame,
)
from smart_financial_coach.intelligence.anomaly.threshold import ScorerModel
from smart_financial_coach.intelligence.models.registry import register


@register("unusual_transactions/rules")
class Rules(ScorerModel):
    def __init__(
        self,
        z_scale: float = 5.0,
        ratio_scale: float = 5.0,
        min_amount: float = 0.0,
        duplicate_minutes: float = DUPLICATE_MINUTES,
        min_spread: float = MIN_SPREAD,
    ) -> None:
        if z_scale <= 0 or ratio_scale <= 1:
            raise ValueError("z_scale must be positive and ratio_scale above 1")
        super().__init__(
            z_scale=z_scale,
            ratio_scale=ratio_scale,
            min_amount=min_amount,
            duplicate_minutes=duplicate_minutes,
            min_spread=min_spread,
        )
        self.z_scale = z_scale
        self.ratio_scale = ratio_scale
        self.min_amount = min_amount
        self.duplicate_minutes = duplicate_minutes
        self.min_spread = min_spread

    def scores_from(self, f: pd.DataFrame) -> pd.DataFrame:
        new = f["first_visit"] & (f["amount"] >= self.min_amount)
        new_score = np.where(new, f["profile_log_ratio"] / np.log(self.ratio_scale), np.nan)
        amount_score = f["z_merchant"].to_numpy() / self.z_scale
        new_score = np.nan_to_num(new_score, nan=-np.inf)
        amount_score = np.nan_to_num(amount_score, nan=-np.inf)
        score = np.maximum(new_score, amount_score)
        reason = np.where(new_score >= amount_score, "new_merchant", "amount_unusual").astype(
            object
        )
        reason[~np.isfinite(score)] = None
        repeat = f["repeat"].to_numpy()
        score = np.where(repeat, np.inf, score)
        reason[repeat] = "duplicate"
        # A row nothing can judge (a first visit without a profile) scores lowest, not NaN
        score = np.where(np.isneginf(score), np.finfo(float).min, score)
        return scores_frame(f, score, reason)
