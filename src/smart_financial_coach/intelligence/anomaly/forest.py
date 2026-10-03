"""The Isolation forest candidate (FR-7 §3): one forest across users, on relative features.

Features: the z at the merchant and the profile ratio, both clipped at 0 so a charge below usual
isn't unusual; a first visit and an exact repeat, as 0/1; and the rank in the user's history.
Missing values are 0 (nothing unusual known). The forest is fitted without labels on a sample of
the training rows. The score is the negated `score_samples`; an exact repeat scores +inf, as in
the other candidates.

- **`one_sided_rank`** makes the rank one-sided (a charge below the user's median counts as the
  median), as §3's one-sided features specify. Off by default, as in the first promoted model,
  where an unusually *cheap* first visit looks anomalous and reads as a large new-merchant charge.
  The owner chose the fix on #39.
- **Departures from §3** (recorded in the design's Implementation notes, review on #37):
  - the reason is the kind of row, not an attribution: a repeat is a `duplicate`, a first visit
    a `new_merchant`, anything else `amount_unusual`;
  - there's no amount-ratio feature at the merchant (the z carries it), and a 0/1 repeat flag
    replaces minutes since the repeat.

  So reason accuracy is about the same for any candidate that flags the same rows; it isn't
  evidence that the forest explains itself well.
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


def _matrix(f: pd.DataFrame, one_sided_rank: bool = False) -> np.ndarray:
    rank = np.nan_to_num(f["rank_in_history"].to_numpy(dtype=float), nan=0.5)
    columns = [
        np.clip(np.nan_to_num(f["z_merchant"].to_numpy(dtype=float)), 0, None),
        np.clip(np.nan_to_num(f["profile_log_ratio"].to_numpy(dtype=float)), 0, None),
        f["first_visit"].to_numpy(dtype=float),
        f["repeat"].to_numpy(dtype=float),
        np.maximum(rank, 0.5) if one_sided_rank else rank,
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
        one_sided_rank: bool = False,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_samples=max_samples,
            seed=seed,
            duplicate_minutes=duplicate_minutes,
            min_spread=min_spread,
            one_sided_rank=one_sided_rank,
        )
        self.one_sided_rank = one_sided_rank
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
        ).fit(_matrix(f, getattr(self, "one_sided_rank", False)))

    def scores_from(self, f: pd.DataFrame) -> pd.DataFrame:
        if self.forest is None:
            raise ValueError("fit the forest first")
        score = -self.forest.score_samples(_matrix(f, getattr(self, "one_sided_rank", False)))
        repeat = f["repeat"].to_numpy()
        first = f["first_visit"].to_numpy()
        reason = np.where(first, "new_merchant", "amount_unusual").astype(object)
        reason[repeat] = "duplicate"
        return scores_frame(f, np.where(repeat, np.inf, score), reason)
