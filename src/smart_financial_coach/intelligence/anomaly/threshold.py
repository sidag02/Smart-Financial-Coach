"""Scorers and the cutoff that turns their scores into flags (FR-7 §5).

A *scorer* is unsupervised: `fit` takes no labels, and `scores(x)` returns, for every input row,
a score (higher is more unusual), the reason code it would carry if flagged, and one column per
evidence key its reasons use (`EVIDENCE`). Scorers flag nothing on their own.

`Thresholded(base, precision=0.80)` places the cutoff. Its `fit` passes no labels to the base
scorer and uses them only to find the deepest cutoff with that precision on the training users.
`Thresholded(base, rate=0.11)` needs no labels: it flags the top `rate` charges per user-month,
which is how a cutoff carries to real data without labels (v2).

Labels, when given, are `anomaly`, `normal` or `ignored` per row (the label contract's ignore
rules: warm-up and duplicate originals); ignored rows don't count towards precision.
"""

import math
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.intelligence.anomaly.contract import EVIDENCE, AnomalyModel
from smart_financial_coach.intelligence.models.base import Model
from smart_financial_coach.intelligence.models.registry import register

ANOMALY, NORMAL, IGNORED = "anomaly", "normal", "ignored"
DAYS_PER_MONTH = 30.4
SCORE_COLUMNS = ("transaction_id", "score", "reason_code")


@runtime_checkable
class Scorer(Protocol):
    def scores(self, x: pd.DataFrame) -> pd.DataFrame: ...


def user_months(x: pd.DataFrame) -> float:
    """The span of each user's rows in months, summed over users: the denominator of flag rates.

    Flag rates are per user-month after the warm-up (FR-7 §6), so callers with labels pass only
    the rows the label contract scores."""
    ts = pd.to_datetime(x["ts"])
    span = ts.groupby(x["user_id"].to_numpy()).agg(lambda t: (t.max() - t.min()).days)
    return float(span.sum() / DAYS_PER_MONTH)


def top_k(score: npt.NDArray[np.float64], ids: npt.NDArray[Any], k: int) -> npt.NDArray[np.bool_]:
    """The `k` highest scores, ties broken by ID so the choice is deterministic."""
    order = np.lexsort((ids, -score))
    chosen = np.zeros(len(score), dtype=bool)
    chosen[order[: max(k, 0)]] = True
    return chosen


def precision_cutoff(
    score: npt.NDArray[np.float64], labels: npt.NDArray[Any], target: float
) -> float:
    """The lowest score whose flags (score >= it) have at least `target` precision, ignored rows
    left out; +inf when no cutoff reaches it, so only +inf scores (exact repeats) are flagged."""
    keep = labels != IGNORED
    s, hit = score[keep], labels[keep] == ANOMALY
    order = np.argsort(-s, kind="mergesort")
    s, hit = s[order], hit[order]
    # Precision is read at the last row of each run of equal scores: they're flagged together
    last = np.r_[s[1:] != s[:-1], True]
    precision = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    ok = np.flatnonzero(last & (precision >= target) & np.isfinite(s))
    return float(s[ok[-1]]) if len(ok) else math.inf


def _plain(value: Any) -> Any:
    """A JSON-ready evidence value: None for missing, Python numbers and strings."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return _plain(value.item())
    return value


def evidence_dicts(scored: pd.DataFrame, flagged: npt.NDArray[np.bool_]) -> list[Any]:
    """Each flagged row's evidence, from the scorer's evidence columns; None elsewhere."""
    out: list[Any] = [None] * len(scored)
    codes = scored["reason_code"].to_numpy()
    for i in np.flatnonzero(flagged):
        keys = EVIDENCE[str(codes[i])]
        out[i] = {k: _plain(scored[k].iat[int(i)]) for k in keys}
    return out


@register("unusual_transactions/thresholded")
class Thresholded(AnomalyModel):
    """Flags the base scorer's charges at or above a cutoff, fitted to a precision or a rate."""

    def __init__(
        self, base: Model, precision: float | None = 0.80, rate: float | None = None
    ) -> None:
        if (precision is None) == (rate is None):
            raise ValueError("give exactly one of precision and rate")
        if not isinstance(base, Scorer):
            raise TypeError(f"{base.name} has no scores(); it can't be thresholded")
        super().__init__(base=base, precision=precision, rate=rate)
        self.base = base
        self.precision = precision
        self.rate = rate
        self.cutoff = math.inf
        self.report: dict[str, float] = {}

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self.base.fit(x)  # never the labels: scorers are unsupervised
        if self.rate is not None:  # the rate is applied per scored batch, in predict
            return self
        if y is None:
            raise ValueError("a precision cutoff needs labels; use rate= without them")
        scored = self.base.scores(x)
        score = scored["score"].to_numpy(dtype=np.float64)
        labels = y.to_numpy()
        self.cutoff = precision_cutoff(score, labels, self.precision or 0.0)
        flagged = (score >= self.cutoff) & scored["reason_code"].notna().to_numpy()
        kept = labels != IGNORED
        hits = flagged & kept & (labels == ANOMALY)
        self.report = {
            "cutoff": self.cutoff,
            "train_precision": float(hits.sum() / max((flagged & kept).sum(), 1)),
            "train_recall": float(hits.sum() / max((kept & (labels == ANOMALY)).sum(), 1)),
            # Per post-warm-up user-month: the rows the label contract scores
            "train_flag_rate": float((flagged & kept).sum() / max(user_months(x[kept]), 1e-9)),
        }
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        scored = self.base.scores(x)
        score = scored["score"].to_numpy(dtype=np.float64)
        has_reason = scored["reason_code"].notna().to_numpy()
        if self.rate is not None:
            k = round(self.rate * user_months(x)) if len(x) else 0
            flagged = top_k(np.where(has_reason, score, -np.inf), x["transaction_id"].to_numpy(), k)
            flagged &= has_reason
        else:
            flagged = (score >= self.cutoff) & has_reason
        codes = scored["reason_code"].where(pd.Series(flagged, index=scored.index), None)
        return self._output(x, score, flagged, codes, evidence_dicts(scored, flagged))
