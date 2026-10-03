"""The categorization service contract, and what callers get from `load_service`.

in:  transaction_id, user_id, ts, amount, currency, merchant_raw, channel
out: transaction_id, category, confidence, model_version, familiar

`familiar` says whether the row's normalized merchant string occurs in the model's training rows.
It is model-visible, so production computes it exactly as validation does, and the review policy
(FR-5 §1) needs nothing the service doesn't already know.
"""

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.data.features.merchant_text import normalize_merchant
from smart_financial_coach.data.generator.taxonomy import DEFAULT_CATEGORIES
from smart_financial_coach.intelligence.models.base import BaseModel
from smart_financial_coach.intelligence.models.contract import Checked, Contract
from smart_financial_coach.intelligence.service import register_service

INPUT_COLUMNS = ("transaction_id", "user_id", "ts", "amount", "currency", "merchant_raw", "channel")
OUTPUT_COLUMNS = ("transaction_id", "category", "confidence", "model_version", "familiar")
# v1 assumes the default taxonomy (12 spending categories + Income). A spec with other
# categories needs this to come from the dataset's `meta.categories` instead.
KNOWN = frozenset(DEFAULT_CATEGORIES)


def _check(out: pd.DataFrame) -> list[str]:
    errors = []
    if unknown := sorted(set(out["category"]) - KNOWN):
        errors.append(f"unknown categories {unknown[:5]}")
    if not out["confidence"].between(0, 1).all():
        errors.append("confidence outside [0, 1]")
    if (out["model_version"].astype(str) == "").any():
        errors.append("empty model_version")
    if not pd.api.types.is_bool_dtype(out["familiar"]):
        errors.append(f"familiar is {out['familiar'].dtype}, not bool")
    return errors


class Categorizer(Checked):
    """The promoted categorizer, contract-checked. `categorize` is the service's verb."""

    def categorize(self, transactions: pd.DataFrame) -> pd.DataFrame:
        return self.predict(transactions)

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(getattr(self.model, "categories", ()))


CONTRACT = Contract("categorization", "transaction_id", OUTPUT_COLUMNS, _check)
register_service(CONTRACT, Categorizer)


class CategorizerModel(BaseModel):
    """Base for categorization models: records the categories seen in training."""

    categories: tuple[str, ...] = ()
    vocabulary: frozenset[str] = frozenset()  # normalized merchant strings in the training rows
    contract: ClassVar[Contract] = CONTRACT

    def _remember(self, y: pd.Series | None) -> pd.Series:
        if y is None:
            raise ValueError(f"{self.name} is supervised: fit needs labels")
        self.categories = tuple(sorted(set(y.astype(str))))
        return y.astype(str).reset_index(drop=True)

    def _learn_strings(self, x: pd.DataFrame) -> None:
        self.vocabulary = frozenset(x["merchant_raw"].map(normalize_merchant))

    def _familiar(self, x: pd.DataFrame) -> npt.NDArray[np.bool_]:
        text = x["merchant_raw"].map(normalize_merchant)
        # A Python set lookup: Series.isin rebuilds the set on every call
        return np.fromiter((s in self.vocabulary for s in text), dtype=bool, count=len(text))

    def _output(
        self, x: pd.DataFrame, category: Any, confidence: Any, familiar: Any
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "transaction_id": x["transaction_id"].to_numpy(),
                "category": category,
                "confidence": confidence,
                "model_version": self.version,
                "familiar": np.asarray(familiar, dtype=bool),
            }
        )
