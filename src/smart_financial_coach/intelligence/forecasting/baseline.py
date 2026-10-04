"""The goal-forecasting baselines (FR-11 and FR-12 design, §4): the floors every candidate beats.

- `naive_pace`: the PRD's "naive" on-track call. The balance so far over the months since the
  goal was created, extended to the target date. A 0/1 call with no range; a goal created this
  month counts one month, so a new goal's pace is whatever was entered.
- `flat`: the same probability for every goal (0.5 by default), the honest floor a calibrated
  forecast must beat. Its projection is the balance as it stands.

Neither learns anything, and neither has a share.
"""

from typing import Any, Self

import pandas as pd

from smart_financial_coach.intelligence.forecasting.contract import (
    months_left,
    output_frame,
    round_up_dollars,
    status_for,
    to_date,
)
from smart_financial_coach.intelligence.models.base import BaseModel
from smart_financial_coach.intelligence.models.registry import register


def _months_since(created: object, saved_as_of: object) -> int:
    """Months from the goal's creation to its saved amount's date, both counted (at least 1)."""
    a, b = pd.Period(to_date(created), freq="M"), pd.Period(to_date(saved_as_of), freq="M")
    return max(1, (b - a).n + 1)


@register("goal_forecasting/naive_pace")
class NaivePace(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for r in x.to_dict("records"):
            saved, target = float(r["saved"]), float(r["target_amount"])
            left = months_left(to_date(r["as_of_date"]), to_date(r["target_date"]))
            pace = saved / _months_since(r["created_date"], r["saved_as_of"])
            projected = saved + pace * left
            reached = saved >= target
            p = 1.0 if projected >= target else 0.0
            short = max(0.0, target - projected)
            rows.append(
                {
                    "example_id": r["example_id"],
                    "status": status_for(p, reached),
                    "p_goal_met": p,
                    "projected_balance": projected,
                    "range_lo": projected,
                    "range_hi": projected,
                    "gap": short,
                    "extra_per_month": (
                        round_up_dollars(short / left)
                        if short > 0 and left and not reached
                        else None
                    ),
                    "share": None,
                    "share_source": None,
                    "may_draw_down": False if reached else None,  # no paths: nothing to fall
                }
            )
        return output_frame(rows, self.version)


@register("goal_forecasting/flat")
class Flat(BaseModel):
    def __init__(self, p: float = 0.5) -> None:
        if not 0 <= p <= 1:
            raise ValueError(f"p must be 0-1, got {p}")
        super().__init__(p=p)
        self.p = p

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for r in x.to_dict("records"):
            saved, target = float(r["saved"]), float(r["target_amount"])
            reached = saved >= target
            rows.append(
                {
                    "example_id": r["example_id"],
                    "status": status_for(self.p, reached),
                    "p_goal_met": self.p,
                    "projected_balance": saved,
                    "range_lo": saved,
                    "range_hi": saved,
                    "gap": max(0.0, target - saved),
                    "extra_per_month": None,
                    "share": None,
                    "share_source": None,
                    "may_draw_down": False if reached else None,  # no paths: nothing to fall
                }
            )
        return output_frame(rows, self.version)
