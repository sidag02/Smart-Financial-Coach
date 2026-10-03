"""Scorers and the cutoff that turns their scores into flags (FR-7 §5).

A *scorer* is unsupervised: `fit` takes no labels, and `scores(x)` returns, for every input row,
a score (higher is more unusual), the reason code it would carry if flagged, and one column per
evidence key its reasons use (`EVIDENCE`). Scorers flag nothing on their own.

`Thresholded(base, precision=0.80)` places the cutoff. Its `fit` passes no labels to the base
scorer and uses them only to find the deepest cutoff with that precision on the training users.
`Thresholded(base, rate=0.11)` needs no labels: it flags the top `rate` charges per user-month,
which is how a cutoff carries to real data without labels (v2).

Labels, when given, are `anomaly`, `normal`, `warmup` or `ignored` per row. The last two are the
label contract's ignore set (the warm-up, and duplicate originals) and don't count towards
precision; flag-rate budgets and month counts leave out only the warm-up, a date rule.
"""

import itertools
import math
from typing import Any, ClassVar, Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.intelligence.anomaly.contract import EVIDENCE, AnomalyModel
from smart_financial_coach.intelligence.anomaly.features import (
    DUPLICATE_MINUTES,
    MIN_SPREAD,
    relative_features,
)
from smart_financial_coach.intelligence.models.base import Model
from smart_financial_coach.intelligence.models.registry import register

ANOMALY, NORMAL, IGNORED, WARMUP = "anomaly", "normal", "ignored", "warmup"
# Neither right nor wrong when flagged (the label contract's ignore set): the warm-up, a date rule,
# and duplicate originals, known only from truth. Flag-rate budgets exclude only the warm-up
UNSCORED = (IGNORED, WARMUP)
DAYS_PER_MONTH = 30.4
SCORE_COLUMNS = ("transaction_id", "score", "reason_code")


@runtime_checkable
class Scorer(Protocol):
    def scores(self, x: pd.DataFrame) -> pd.DataFrame: ...


class ScorerModel(AnomalyModel):
    """Base for scorers: unsupervised `fit`, `scores`, and a `predict` that flags nothing.

    Subclasses implement `scores_from` (and `fit_features` if they learn anything) on the frame
    `features` builds, so `Thresholded`'s search builds it once for every point it tries.
    """

    duplicate_minutes: float = DUPLICATE_MINUTES
    min_spread: float = MIN_SPREAD
    FEATURE_PARAMS: ClassVar[frozenset[str]] = frozenset({"duplicate_minutes", "min_spread"})

    def features(self, x: pd.DataFrame) -> pd.DataFrame:
        return relative_features(x, self.duplicate_minutes, self.min_spread)

    def fit_features(self, f: pd.DataFrame) -> None:
        """Learn from the training rows' features; nothing by default."""

    def scores_from(self, f: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self.fit_features(self.features(x))
        return self

    def scores(self, x: pd.DataFrame) -> pd.DataFrame:
        return self.scores_from(self.features(x))

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """Scores only; flags come from a cutoff (`Thresholded`)."""
        none: list[Any] = [None] * len(x)
        return self._output(x, self.scores(x)["score"], np.zeros(len(x), dtype=bool), none, none)


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
    keep = ~np.isin(labels, UNSCORED)
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
    """Flags the base scorer's charges at or above a cutoff, fitted to a precision or a rate.

    With `search`, the base scorer's params are chosen per fit, before the cutoff: each point is
    a fresh scorer fitted without labels, and the point with the best recall at `search_rate`
    flags per user-month on the training rows wins (FR-7 §5: label-tuned parameters are fitted
    within the folds). The first point wins ties.
    """

    def __init__(
        self,
        base: Model,
        precision: float | None = 0.80,
        rate: float | None = None,
        search: dict[str, list[Any]] | None = None,
        search_rate: float = 0.11,
    ) -> None:
        if (precision is None) == (rate is None):
            raise ValueError("give exactly one of precision and rate")
        if not isinstance(base, Scorer):
            raise TypeError(f"{base.name} has no scores(); it can't be thresholded")
        if search and not isinstance(base, ScorerModel):
            raise TypeError(f"{base.name} can't be searched: it isn't a ScorerModel")
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

    def _scores(self, x: pd.DataFrame) -> pd.DataFrame:
        base = self.base
        assert isinstance(base, Scorer)
        return base.scores(x)

    def _points(self) -> list[dict[str, Any]]:
        keys = sorted(self.search)
        return [
            dict(zip(keys, values, strict=True))
            for values in itertools.product(*(self.search[k] for k in keys))
        ]

    def _search(self, x: pd.DataFrame, labels: npt.NDArray[Any]) -> None:
        """Choose the base scorer's params on these rows, by recall at `search_rate`."""
        base = self.base
        assert isinstance(base, ScorerModel)
        shared = not set(self.search) & base.FEATURE_PARAMS
        features = base.features(x) if shared else None
        # The budget leaves out only the warm-up (a date rule), as the task's common rate does
        budget = labels != WARMUP
        positives = max(int((labels == ANOMALY).sum()), 1)
        k = round(self.search_rate * user_months(x[budget]))  # per post-warm-up user-month
        ids = x["transaction_id"].to_numpy()
        best: tuple[float, ScorerModel, dict[str, Any]] | None = None
        for point in self._points():
            candidate = type(base)(**{**base.params, **point})
            f = features if features is not None else candidate.features(x)
            candidate.fit_features(f)  # never the labels
            score = candidate.scores_from(f)["score"].to_numpy(dtype=np.float64)
            chosen = top_k(np.where(budget, score, -np.inf), ids, k)
            recall = float((chosen & (labels == ANOMALY)).sum() / positives)
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
        if self.rate is not None:  # the rate is applied per scored batch, in predict
            self.base.fit(x)  # never the labels: scorers are unsupervised
            return self
        if y is None:
            raise ValueError("a precision cutoff needs labels; use rate= without them")
        labels = y.to_numpy()
        if self.search:
            self._search(x, labels)
        else:
            self.base.fit(x)
        scored = self._scores(x)
        score = scored["score"].to_numpy(dtype=np.float64)
        self.cutoff = precision_cutoff(score, labels, self.precision or 0.0)
        flagged = (score >= self.cutoff) & scored["reason_code"].notna().to_numpy()
        kept = ~np.isin(labels, UNSCORED)
        hits = flagged & kept & (labels == ANOMALY)
        self.report |= {
            "cutoff": self.cutoff,
            "train_precision": float(hits.sum() / max((flagged & kept).sum(), 1)),
            "train_recall": float(hits.sum() / max((kept & (labels == ANOMALY)).sum(), 1)),
            # Per post-warm-up user-month: the rows the label contract scores
            "train_flag_rate": float(
                (flagged & kept).sum() / max(user_months(x[labels != WARMUP]), 1e-9)
            ),
        }
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        scored = self._scores(x)
        score = scored["score"].to_numpy(dtype=np.float64)
        has_reason = scored["reason_code"].notna().to_numpy()
        if self.rate is not None:
            # Over the batch's whole span: a nightly job that scores a window of new charges must
            # pass rows whose span matches the rate it was given (per post-warm-up user-month)
            k = round(self.rate * user_months(x)) if len(x) else 0
            flagged = top_k(np.where(has_reason, score, -np.inf), x["transaction_id"].to_numpy(), k)
            flagged &= has_reason
        else:
            flagged = (score >= self.cutoff) & has_reason
        codes = scored["reason_code"].where(pd.Series(flagged, index=scored.index), None)
        return self._output(x, score, flagged, codes, evidence_dicts(scored, flagged))
