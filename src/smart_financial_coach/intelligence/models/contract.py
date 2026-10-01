"""Service contracts: each service's output schema, checked on every prediction.

A model that breaks its contract fails in validation, at promotion and in tests, not in the
dashboard.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd

from smart_financial_coach.intelligence.models.base import Model


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class Contract:
    service: str
    id_column: str
    columns: tuple[str, ...]  # output columns, in order, starting with `id_column`
    check: Callable[[pd.DataFrame], list[str]] | None = None  # service-specific rules

    def violations(self, x: pd.DataFrame, out: pd.DataFrame) -> list[str]:
        if list(out.columns) != list(self.columns):
            return [f"columns {list(out.columns)}, expected {list(self.columns)}"]
        if len(out) != len(x):
            return [f"{len(out)} rows for {len(x)} inputs"]
        errors = []
        returned = out[self.id_column].astype(str).to_numpy()
        if not np.array_equal(returned, x[self.id_column].astype(str).to_numpy()):
            errors.append(f"{self.id_column} doesn't match the input rows in order")
        if nulls := [c for c in self.columns if out[c].isna().any()]:
            errors.append(f"missing values in {nulls}")
        if self.check is not None and not errors:
            errors += self.check(out)
        return errors


class Checked:
    """A model whose every prediction is checked against its service's contract."""

    name: ClassVar[str] = "checked"

    def __init__(self, model: Model, contract: Contract) -> None:
        self.model = model
        self.contract = contract

    @property
    def version(self) -> str:
        return self.model.version

    @property
    def params(self) -> Mapping[str, Any]:
        return self.model.params

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self.model.fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        out = self.model.predict(x)
        if errors := self.contract.violations(x, out):
            raise ContractError(
                f"{self.model.name} ({self.model.version}) broke the {self.contract.service} "
                f"contract: {'; '.join(errors)}"
            )
        return out
