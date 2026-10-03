"""The Technical Design's baseline for unusual transactions: a per-user z-score on amount.

Each outflow is compared with the mean and standard deviation of the same user's earlier
outflows, ignoring merchants. It is the PRD's "simple rule-based alternative" that a candidate
must beat at the same flag rate (FR-7 §5). Charges with fewer than `min_history` earlier outflows
score 0.

Its reasons say `amount_unusual` with the user's mean as the usual amount. That reads "here" for a
whole history, which is one reason the baseline is never shipped.

With `repeats`, an exact repeat (same raw text and amount within `duplicate_minutes`) scores +inf
as a `duplicate`: the "baseline plus the duplicate rule", reported alongside, never gated (owner
decision on #30).
"""

from typing import Any, Self

import numpy as np
import pandas as pd

from smart_financial_coach.data.features.history import history_features
from smart_financial_coach.intelligence.anomaly.contract import AnomalyModel
from smart_financial_coach.intelligence.models.registry import register

EVIDENCE_COLUMNS = ("usual_amount", "ratio", "prior_charges")


@register("unusual_transactions/user_zscore")
class UserZScore(AnomalyModel):
    def __init__(
        self, min_history: int = 10, repeats: bool = False, duplicate_minutes: float = 90.0
    ) -> None:
        super().__init__(
            min_history=min_history, repeats=repeats, duplicate_minutes=duplicate_minutes
        )
        self.min_history = min_history
        self.repeats = repeats
        self.duplicate_minutes = duplicate_minutes

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self  # nothing to learn: every statistic is the user's own history

    def scores(self, x: pd.DataFrame) -> pd.DataFrame:
        order = x.sort_values(["user_id", "ts", "transaction_id"], kind="mergesort").index
        s = x.loc[order]
        amount = -s["amount"].astype(float)
        by_user = s["user_id"].to_numpy()
        n = amount.groupby(by_user).cumcount()
        total = amount.groupby(by_user).cumsum() - amount
        squares = (amount**2).groupby(by_user).cumsum() - amount**2
        mean = total / n.clip(lower=1)
        sd = ((squares / n.clip(lower=1) - mean**2).clip(lower=0)) ** 0.5
        enough = (n >= self.min_history) & (sd > 0)
        z = ((amount - mean) / sd.where(sd > 0, 1.0)).where(enough, 0.0)
        out = pd.DataFrame(
            {
                "transaction_id": s["transaction_id"].to_numpy(),
                "score": z.to_numpy(dtype=float),
                "reason_code": pd.Series("amount_unusual", index=s.index).where(enough, None),
                "usual_amount": mean.round(2).to_numpy(),
                "ratio": (amount / mean.where(mean > 0, np.nan)).round(2).to_numpy(),
                "prior_charges": n.to_numpy(),
            },
            index=order,
        )
        out = out.loc[x.index].reset_index(drop=True)
        if self.repeats:
            h = history_features(x)  # one row per outflow, in input order
            minutes = h["minutes_since_repeat"].to_numpy(dtype=float)
            repeat = np.nan_to_num(minutes, nan=np.inf) <= self.duplicate_minutes
            out.loc[repeat, "score"] = np.inf
            out.loc[repeat, "reason_code"] = "duplicate"
            out["original_transaction_id"] = h["repeat_of"].where(pd.Series(repeat), None)
            out["minutes_apart"] = np.where(repeat, minutes, np.nan)
            out["amount"] = (-x["amount"].to_numpy(dtype=float)).round(2)
        return out

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """Scores only; flags come from a cutoff (`Thresholded`)."""
        scored = self.scores(x)
        none: list[Any] = [None] * len(x)
        return self._output(x, scored["score"], np.zeros(len(x), dtype=bool), none, none)
