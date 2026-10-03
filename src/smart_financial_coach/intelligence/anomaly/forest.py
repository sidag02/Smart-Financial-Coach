"""The Isolation forest candidate (FR-7 §3): one forest across users, on relative features.

Features are one-sided, so a charge far *below* usual isn't unusual: the z at the merchant and
the profile ratio are clipped at 0, a first visit and an exact repeat are 0/1, and the rank in the
user's history is used as is. Missing values are 0 (nothing unusual known). The forest is fitted
without labels on a sample of the training rows. The score is the negated `score_samples`, and an
exact repeat scores +inf, as in the other candidates.

The reason is the kind of row, not an attribution: a repeat is a `duplicate`, a first visit is a
`new_merchant`, anything else is `amount_unusual`. That makes reasons weaker than the rules' and is
why the forest has to win on recall to be chosen (§3).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from smart_financial_coach.intelligence.anomaly.features import (
    DUPLICATE_MINUTES,
    MIN_SPREAD,
    scores_frame,
)
from smart_financial_coach.intelligence.anomaly.threshold import ScorerModel
from smart_financial_coach.intelligence.models.registry import register

FEATURES = ("z_merchant", "profile_log_ratio", "first_visit", "repeat", "rank_in_history")


def _matrix(f: pd.DataFrame) -> np.ndarray:
    columns = [
        np.clip(np.nan_to_num(f["z_merchant"].to_numpy(dtype=float)), 0, None),
        np.clip(np.nan_to_num(f["profile_log_ratio"].to_numpy(dtype=float)), 0, None),
        f["first_visit"].to_numpy(dtype=float),
        f["repeat"].to_numpy(dtype=float),
        np.nan_to_num(f["rank_in_history"].to_numpy(dtype=float), nan=0.5),
    ]
    return np.column_stack(columns)


@register("unusual_transactions/isolation_forest")
class Forest(ScorerModel):
    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: int = 4096,
        seed: int = 0,
        duplicate_minutes: float = DUPLICATE_MINUTES,
        min_spread: float = MIN_SPREAD,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_samples=max_samples,
            seed=seed,
            duplicate_minutes=duplicate_minutes,
            min_spread=min_spread,
        )
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seed = seed
        self.duplicate_minutes = duplicate_minutes
        self.min_spread = min_spread
        self.forest: IsolationForest | None = None

    def fit_features(self, f: pd.DataFrame) -> None:
        self.forest = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, len(f)),
            random_state=self.seed,
        ).fit(_matrix(f))

    def scores_from(self, f: pd.DataFrame) -> pd.DataFrame:
        if self.forest is None:
            raise ValueError("fit the forest first")
        score = -self.forest.score_samples(_matrix(f))
        repeat = f["repeat"].to_numpy()
        first = f["first_visit"].to_numpy()
        reason = np.where(first, "new_merchant", "amount_unusual").astype(object)
        reason[repeat] = "duplicate"
        return scores_frame(f, np.where(repeat, np.inf, score), reason)
