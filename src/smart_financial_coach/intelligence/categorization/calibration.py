"""Calibrated confidence: one calibrator per familiarity group, fitted on held-out predictions.

    {"type": "categorization/calibrated",
     "params": {"base": {"$model": {"type": "categorization/linear_text"}}, "method": "auto"}}

Only the predicted class's confidence is mapped, so the predicted category never changes.

- Groups: "familiar" (the normalized string occurs in the base model's training rows) and
  "unfamiliar". Familiarity is model-visible, so production computes it the same way.
- Methods: "none", "temperature" (one scale on the log-probabilities, fitted by negative log
  likelihood of the true class) and "isotonic" (a monotone map of the top-class confidence).
- "auto" picks per group by **cross-validated Brier score** of the top-class confidence: each
  fold's calibrator is fitted on the other folds' predictions, so a flexible method can't win by
  fitting the rows it's scored on. Brier, not expected calibration error, because ECE isn't a
  proper score: a constant confidence equal to the accuracy has ECE 0 and separates nothing.

The runner calls `held_out_outputs` on each fold model and `fit_held_out` on the final model with
the pooled outputs (FR-3 §5): the calibrators learn from merchants each fold model never saw.
"""

from typing import Any, Protocol, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from smart_financial_coach.intelligence.categorization.contract import CategorizerModel
from smart_financial_coach.intelligence.models.registry import register

METHODS = ("none", "temperature", "isotonic")
GROUPS = {"familiar": True, "unfamiliar": False}
MIN_ROWS = 200  # below this a group keeps the uncalibrated confidence
PROB = "p::"  # held-out output columns holding class probabilities
EPS = 1e-12

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.integer[Any]]


class Scorer(Protocol):
    categories: tuple[str, ...]

    def scores(self, x: pd.DataFrame) -> tuple[Floats, npt.NDArray[np.bool_]]: ...


def _top(proba: Floats) -> tuple[Ints, Floats]:
    top = proba.argmax(axis=1)
    return top, proba[np.arange(len(top)), top]


def _tempered_top(proba: Floats, temperature: float) -> Floats:
    logits = np.log(np.clip(proba, EPS, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    tempered = np.exp(logits)
    return np.asarray(tempered.max(axis=1) / tempered.sum(axis=1))


class Calibrator:
    def __init__(self, method: str) -> None:
        self.method = method
        self.temperature = 1.0
        self.isotonic: IsotonicRegression | None = None

    def fit(self, proba: Floats, truth: Ints) -> Self:
        known = truth >= 0  # labels the base model never saw can't be scored
        proba, truth = proba[known], truth[known]
        if self.method == "temperature":
            log_p = np.log(np.clip(proba, EPS, 1))

            def nll(log_t: float) -> float:
                z = log_p / np.exp(log_t)
                z -= z.max(axis=1, keepdims=True)
                log_softmax = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
                return float(-log_softmax[np.arange(len(truth)), truth].mean())

            self.temperature = float(np.exp(minimize_scalar(nll, bounds=(-3, 3)).x))
        elif self.method == "isotonic":
            top, confidence = _top(proba)
            self.isotonic = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
                confidence, (top == truth).astype(float)
            )
        return self

    def apply(self, proba: Floats) -> Floats:
        if self.method == "temperature":
            return _tempered_top(proba, self.temperature)
        _, confidence = _top(proba)
        if self.method == "isotonic" and self.isotonic is not None:
            return np.asarray(self.isotonic.predict(confidence))
        return confidence


def brier(confidence: Floats, correct: npt.NDArray[np.bool_]) -> float:
    return float(np.mean((confidence - correct) ** 2))


@register("categorization/calibrated")
class Calibrated(CategorizerModel):
    def __init__(self, base: Any, method: str = "auto", by: str = "familiarity") -> None:
        super().__init__(base=base, method=method, by=by)
        if method not in (*METHODS, "auto") or by not in ("familiarity", "none"):
            raise ValueError(f"unknown calibration method {method!r} or grouping {by!r}")
        self.base: Scorer = base
        self.method = method
        self.by = by
        self.calibrators: dict[str, Calibrator] = {}
        self.report: dict[str, float] = {}

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self.base.fit(x, y)  # type: ignore[attr-defined]
        self.categories = self.base.categories
        self.calibrators = {}  # uncalibrated until `fit_held_out`
        return self

    def _groups(self, familiar: npt.NDArray[np.bool_]) -> dict[str, npt.NDArray[np.bool_]]:
        if self.by == "none":
            return {"all": np.ones(len(familiar), dtype=bool)}
        return {name: familiar == flag for name, flag in GROUPS.items()}

    def _confidence(self, proba: Floats, familiar: npt.NDArray[np.bool_]) -> Floats:
        _, confidence = _top(proba)
        out = confidence.copy()
        for name, mask in self._groups(familiar).items():
            if name in self.calibrators and mask.any():
                out[mask] = self.calibrators[name].apply(proba[mask])
        return out

    def reset_caches(self) -> None:
        if callable(reset := getattr(self.base, "reset_caches", None)):
            reset()

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        proba, familiar = self.base.scores(x)
        top, _ = _top(proba)
        category = np.asarray(self.categories)[top]
        return self._output(x, category, self._confidence(proba, familiar), familiar)

    # --- Held-out fitting (called by the runner) -------------------------------------------

    def held_out_outputs(self, x: pd.DataFrame) -> pd.DataFrame:
        proba, familiar = self.base.scores(x)
        columns = {f"{PROB}{c}": proba[:, i] for i, c in enumerate(self.categories)}
        return pd.DataFrame(
            {"transaction_id": x["transaction_id"].to_numpy(), "familiar": familiar, **columns}
        )

    def fit_held_out(self, outputs: pd.DataFrame, y: pd.Series, folds: Ints) -> Floats:
        """Choose and fit each group's calibrator; return out-of-fold calibrated confidences."""
        index = {c: i for i, c in enumerate(self.categories)}
        proba = outputs[[f"{PROB}{c}" for c in self.categories]].to_numpy(dtype=float)
        truth = y.map(index).fillna(-1).to_numpy(dtype=np.int64)
        correct = proba.argmax(axis=1) == truth
        oof = _top(proba)[1].copy()
        self.calibrators, self.report = {}, {}
        for name, mask in self._groups(outputs["familiar"].to_numpy(dtype=bool)).items():
            if mask.sum() < MIN_ROWS:
                continue
            candidates = METHODS if self.method == "auto" else (self.method,)
            scored = {m: self._cross_validated(m, proba, truth, folds, mask) for m in candidates}
            for m, conf in scored.items():
                self.report[f"{name}.{m}_brier"] = brier(conf, correct[mask])
            best = min(candidates, key=lambda m: self.report[f"{name}.{m}_brier"])
            self.report[f"{name}.chosen_{best}"] = 1.0
            self.calibrators[name] = Calibrator(best).fit(proba[mask], truth[mask])
            oof[mask] = scored[best]
        return oof

    @staticmethod
    def _cross_validated(
        method: str,
        proba: Floats,
        truth: Ints,
        folds: Ints,
        mask: npt.NDArray[np.bool_],
    ) -> Floats:
        group_proba, group_truth, group_folds = proba[mask], truth[mask], folds[mask]
        out = _top(group_proba)[1].copy()
        for fold in np.unique(group_folds):
            test = group_folds == fold
            if test.all():  # one fold only: nothing to fit on
                continue
            calibrator = Calibrator(method).fit(group_proba[~test], group_truth[~test])
            out[test] = calibrator.apply(group_proba[test])
        return out
