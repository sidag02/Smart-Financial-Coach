"""The interface every model implements, whatever its service or algorithm.

    model = build({"type": "categorization/linear_text", "params": {"C": 3.0}})
    model.fit(x, y)
    model.predict(x)  # one row per input row, in input order

Labels arrive only as `fit`'s argument, so a model never knows where they came from.
"""

from collections.abc import Mapping
from typing import Any, ClassVar, Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

UNVERSIONED = "unversioned"


@runtime_checkable
class Model(Protocol):
    name: ClassVar[str]  # registry key, e.g. "categorization/linear_text"
    version: str

    @property
    def params(self) -> Mapping[str, Any]: ...

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self: ...

    def predict(self, x: pd.DataFrame) -> pd.DataFrame: ...


@runtime_checkable
class HeldOutFit(Protocol):
    """A model with a stage fitted on held-out predictions (e.g. calibration).

    The runner collects `held_out_outputs` from every validation-fold model, pools them, and
    passes them to the final model's `fit_held_out` with the true labels and each row's fold.
    `fit_held_out` returns out-of-fold confidences, so validation metrics stay honest.
    """

    def held_out_outputs(self, x: pd.DataFrame) -> pd.DataFrame: ...

    def fit_held_out(
        self, outputs: pd.DataFrame, y: pd.Series, folds: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]: ...


class BaseModel:
    """Keeps the constructor's params and a version, so implementations only fit and predict.

    Subclasses pass their constructor arguments up: `super().__init__(C=C, features=features)`.
    `y` is optional because anomaly and spike models are unsupervised.
    """

    name: ClassVar[str] = ""
    version: str = UNVERSIONED

    def __init__(self, **params: Any) -> None:
        self._params = dict(params)

    @property
    def params(self) -> Mapping[str, Any]:
        return dict(self._params)

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        raise NotImplementedError

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


def describe(value: Any) -> Any:
    """JSON-ready params: nested models become `{"$model": {type, params}}`, as `build` takes."""
    if isinstance(value, Model):
        return {"$model": {"type": value.name, "params": describe(dict(value.params))}}
    if isinstance(value, Mapping):
        return {str(k): describe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [describe(v) for v in value]
    return value
