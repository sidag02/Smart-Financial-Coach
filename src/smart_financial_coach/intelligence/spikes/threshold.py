"""Spike scorers and the cutoff that turns their scores into flags (FR-8 §5).

A *scorer* is unsupervised: `fit` takes no labels, and `scores(x)` returns a score per period
(higher is more unusual; -inf where it can't judge). Scorers flag nothing on their own.

`SpikeThresholded(base, precision=0.80)` places the cutoff, as FR-7's `Thresholded` does for
charges, on period keys. Its `fit` passes no labels to the base scorer and uses them only to find
the deepest cutoff with that precision among the periods the product rules allow
(`SpikeModel.eligible`). `SpikeThresholded(base, rate=0.035)` needs no labels: the cutoff is the
score that flags `rate` periods per user-month on the rows it was fitted on, a fixed number from
then on, so a batch of one user's months is judged the same as a nightly batch of everyone's
(the simple-rule fallback, decision 12; the v2 path to real data).

Labels, when given, are `spike`, `normal`, `ignored` or `basket` per period. `ignored` is the label
contract's ignore set (months with a planted weekly spike or unusual charge in the category).
`basket` marks the task's simulated basket-size spikes (decision 1), which are scored for one
diagnostic only. Neither counts towards precision, and `basket` rows never count towards a rate.
"""

import itertools
import math
from typing import Any, ClassVar, Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.intelligence.anomaly.threshold import top_k
from smart_financial_coach.intelligence.models.base import Model
from smart_financial_coach.intelligence.models.registry import register
from smart_financial_coach.intelligence.spikes.contract import SpikeModel

SPIKE, NORMAL, IGNORED, BASKET = "spike", "normal", "ignored", "basket"
UNSCORED = (IGNORED, BASKET)  # neither right nor wrong when flagged
SEARCH_RATE = 0.035  # the common operating point (FR-8 §4)


def user_months(x: pd.DataFrame) -> int:
    """Distinct (user, month) pairs among the periods: the denominator of flag rates."""
    return int(x[["user_id", "period_start"]].drop_duplicates().shape[0])


@runtime_checkable
class Scorer(Protocol):
    def scores(self, x: pd.DataFrame) -> npt.NDArray[np.float64]: ...


class SpikeScorerModel(SpikeModel):
    """Base for scorers: unsupervised `fit`, `scores`, and a `predict` that flags nothing.

    `report` holds what the fit learned and anything selection reads from it, e.g.
    `assumes_poisson` (decision 6's tie-break: the negative binomial before the Poisson).
    """

    assumes_poisson: ClassVar[float] = 0.0

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.report: dict[str, float] = {"assumes_poisson": self.assumes_poisson}

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def scores(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """Scores only; flags come from a cutoff (`SpikeThresholded`)."""
        return self._output(x, self.scores(x), np.zeros(len(x), dtype=bool))


def precision_cutoff(
    score: npt.NDArray[np.float64], labels: npt.NDArray[Any], target: float
) -> float:
    """The lowest score whose flags (score >= it) have at least `target` precision, unscored rows
    left out; +inf when no cutoff reaches it, so nothing is flagged."""
    keep = ~np.isin(labels, UNSCORED) & np.isfinite(score)
    s, hit = score[keep], labels[keep] == SPIKE
    order = np.argsort(-s, kind="mergesort")
    s, hit = s[order], hit[order]
    # Precision is read at the last row of each run of equal scores: they're flagged together
    last = np.r_[s[1:] != s[:-1], True]
    precision = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    ok = np.flatnonzero(last & (precision >= target))
    return float(s[ok[-1]]) if len(ok) else math.inf


def rate_cutoff(score: npt.NDArray[np.float64], ids: npt.NDArray[Any], k: int) -> float:
    """The score of the `k`-th highest finite score: flagging at or above it flags about `k`."""
    finite = np.isfinite(score)
    if k <= 0 or not finite.any():
        return math.inf
    chosen = top_k(np.where(finite, score, -np.inf), ids, k) & finite
    return float(score[chosen].min())


@register("spending_spikes/thresholded")
class SpikeThresholded(SpikeModel):
    """Flags the base scorer's eligible periods at or above a cutoff, fitted to a precision or a
    rate.

    With `search`, the base scorer's params are chosen per fit, before the cutoff: each point is a
    fresh scorer fitted without labels, and the point with the best recall at `search_rate` flags
    per user-month on the training rows wins (label-tuned parameters are fitted within the folds,
    FR-8 §5). The first point wins ties.
    """

    def __init__(
        self,
        base: Model,
        precision: float | None = 0.80,
        rate: float | None = None,
        search: dict[str, list[Any]] | None = None,
        search_rate: float = SEARCH_RATE,
    ) -> None:
        if (precision is None) == (rate is None):
            raise ValueError("give exactly one of precision and rate")
        if not isinstance(base, Scorer):
            raise TypeError(f"{base.name} has no scores(); it can't be thresholded")
        if search and not isinstance(base, SpikeScorerModel):
            raise TypeError(f"{base.name} can't be searched: it isn't a SpikeScorerModel")
        super().__init__(
            base=base, precision=precision, rate=rate, search=search, search_rate=search_rate
        )
        self.base: Model = base
        self.precision = precision
        self.rate = rate
        self.search = dict(search or {})
        self.search_rate = search_rate
        self.cutoff = math.inf
        self.report: dict[str, float] = {}
        self.chosen: dict[str, Any] = {}

    def _scores(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        """The base scorer's scores, -inf where the product rules forbid a flag."""
        base = self.base
        assert isinstance(base, Scorer)
        score = np.nan_to_num(np.asarray(base.scores(x), dtype=np.float64), nan=-np.inf)
        return np.where(self.eligible(x), score, -np.inf)

    def _points(self) -> list[dict[str, Any]]:
        keys = sorted(self.search)
        return [
            dict(zip(keys, values, strict=True))
            for values in itertools.product(*(self.search[k] for k in keys))
        ]

    def _search(self, x: pd.DataFrame, labels: npt.NDArray[Any]) -> None:
        """Choose the base scorer's params on these rows, by recall at `search_rate`."""
        base = self.base
        assert isinstance(base, SpikeScorerModel)
        counted = labels != BASKET
        positives = max(int((labels == SPIKE).sum()), 1)
        k = round(self.search_rate * user_months(x[counted]))
        ids = x["period_id"].to_numpy()
        best: tuple[float, SpikeScorerModel, dict[str, Any]] | None = None
        for point in self._points():
            candidate = type(base)(**{**base.params, **point})
            candidate.fit(x)  # never the labels
            score = np.where(self.eligible(x) & counted, np.asarray(candidate.scores(x)), -np.inf)
            chosen = top_k(score, ids, k) & np.isfinite(score)
            recall = float((chosen & (labels == SPIKE)).sum() / positives)
            key = ".".join(f"{name}_{value}" for name, value in sorted(point.items()))
            self.report[f"search_recall.{key}"] = recall
            if best is None or recall > best[0]:
                best = (recall, candidate, point)
        assert best is not None
        _, self.base, self.chosen = best
        # The params now describe the scorer in use, so the manifest records what it runs with
        self._params["base"] = self.base
        self.report |= {f"chosen.{name}": float(value) for name, value in self.chosen.items()}

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        if self.rate is not None and self.search:
            raise ValueError("a search needs labels; it can't run with a rate cutoff")
        labels = None if y is None else y.to_numpy()
        if self.rate is not None:
            self.base.fit(x)  # never the labels: scorers are unsupervised
            counted = np.ones(len(x), dtype=bool) if labels is None else labels != BASKET
            score = np.where(counted, self._scores(x), -np.inf)
            k = round(self.rate * user_months(x[counted]))
            self.cutoff = rate_cutoff(score, x["period_id"].to_numpy(), k)
        else:
            if labels is None:
                raise ValueError("a precision cutoff needs labels; use rate= without them")
            if self.search:
                self._search(x, labels)
            else:
                self.base.fit(x)
            self.cutoff = precision_cutoff(self._scores(x), labels, self.precision or 0.0)
        self.report |= {"cutoff": self.cutoff, **getattr(self.base, "report", {})}
        if labels is not None:
            flagged = self._scores(x) >= self.cutoff
            kept = ~np.isin(labels, UNSCORED)
            hits = flagged & kept & (labels == SPIKE)
            self.report |= {
                "train_precision": float(hits.sum() / max((flagged & kept).sum(), 1)),
                "train_recall": float(hits.sum() / max((labels == SPIKE).sum(), 1)),
                "train_flag_rate": float(
                    (flagged & kept).sum() / max(user_months(x[labels != BASKET]), 1)
                ),
            }
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        base = self.base
        assert isinstance(base, Scorer)
        raw = np.nan_to_num(np.asarray(base.scores(x), dtype=np.float64), nan=-np.inf)
        return self._output(x, raw, self.eligible(x) & (raw >= self.cutoff))
