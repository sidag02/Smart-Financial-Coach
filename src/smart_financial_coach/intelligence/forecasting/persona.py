"""How much a user's history looks like each persona: the weights of the persona mixture.

A real user has no persona label (FR-1's personas are how the data was generated), so a forecast
that serves real users can't be keyed by one. The mixture instead weighs every persona's seasonal
prior by how much the user's own monthly net savings look like that persona's. The weights come
from a small classifier over features of the history alone, trained on train users, whose persona
is known as part of the training data; predicting never reads a persona.

    weights = PersonaWeights().fit(histories, personas)
    weights.predict(history)  # {"family_budgeter": 0.1, "freelancer": 0.8, ...}
"""

from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

Floats = npt.NDArray[np.float64]

SEASON_FROM = 13  # months before a month-of-year profile says anything


def history_features(history: pd.Series) -> Floats:
    """Scale-free features of a user's monthly net savings: how volatile, lopsided and
    persistent it is, how it drifts, and its month-of-year pattern once there's a year of it."""
    y = history.to_numpy(dtype=float)
    n = len(y)
    scale = float(np.mean(np.abs(y))) if n else 0.0
    out = np.zeros(20)
    if n == 0 or scale <= 0:
        return out
    z = (y - y.mean()) / scale
    out[0] = np.log(n)
    out[1] = y.std() / scale
    out[2] = float(np.mean(y < 0))
    out[3] = float(np.median(np.abs(y - np.median(y))) / scale)
    out[4] = float(np.mean(z**3) / max(np.mean(z**2) ** 1.5, 1e-9))
    out[5] = float(np.corrcoef(z[:-1], z[1:])[0, 1]) if n >= 4 and z.std() > 0 else 0.0
    half = min(12, n // 2)
    out[6] = (y[-half:].mean() - y[-2 * half : -half].mean()) / scale if half >= 3 else 0.0
    if n >= SEASON_FROM:
        start = history.index[-1] - (n - 1)
        total, count = np.zeros(12), np.zeros(12)
        for t in range(12, n):
            m = (start + t).month - 1
            total[m] += y[t] - y[t - 12 : t].mean()
            count[m] += 1
        out[7] = 1.0
        out[8:20] = np.divide(total, count, out=np.zeros(12), where=count > 0) / scale
    return out


class PersonaWeights:
    """A multinomial logistic regression from history features to persona probabilities."""

    def __init__(self, c: float = 1.0) -> None:
        self.c = c
        self.personas: tuple[str, ...] = ()
        self._model: Pipeline | None = None

    def fit(self, histories: list[pd.Series], personas: list[str]) -> Self:
        x = np.stack([history_features(h) for h in histories])
        self._model = make_pipeline(StandardScaler(), LogisticRegression(C=self.c, max_iter=2000))
        self._model.fit(x, personas)
        self.personas = tuple(str(p) for p in self._model.classes_)
        return self

    def predict(self, history: pd.Series) -> dict[str, float]:
        if self._model is None:
            raise ValueError("PersonaWeights isn't fitted")
        p = self._model.predict_proba(history_features(history)[None, :])[0]
        return {persona: float(w) for persona, w in zip(self.personas, p, strict=True)}
