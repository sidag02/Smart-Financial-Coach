"""The categorization service contract, and what callers get from `load_service`.

in:  transaction_id, user_id, ts, amount, currency, merchant_raw, channel
out: transaction_id, category, confidence, model_version
"""

from typing import Any, ClassVar

import pandas as pd

from smart_financial_coach.data.generator.taxonomy import DEFAULT_CATEGORIES
from smart_financial_coach.intelligence.models.base import BaseModel
from smart_financial_coach.intelligence.models.contract import Checked, Contract
from smart_financial_coach.intelligence.service import register_service

INPUT_COLUMNS = ("transaction_id", "user_id", "ts", "amount", "currency", "merchant_raw", "channel")
OUTPUT_COLUMNS = ("transaction_id", "category", "confidence", "model_version")
KNOWN = frozenset(DEFAULT_CATEGORIES)  # 12 spending categories + Income


def _check(out: pd.DataFrame) -> list[str]:
    errors = []
    if unknown := sorted(set(out["category"]) - KNOWN):
        errors.append(f"unknown categories {unknown[:5]}")
    if not out["confidence"].between(0, 1).all():
        errors.append("confidence outside [0, 1]")
    if (out["model_version"].astype(str) == "").any():
        errors.append("empty model_version")
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
    contract: ClassVar[Contract] = CONTRACT

    def _remember(self, y: pd.Series | None) -> pd.Series:
        if y is None:
            raise ValueError(f"{self.name} is supervised: fit needs labels")
        self.categories = tuple(sorted(set(y.astype(str))))
        return y.astype(str).reset_index(drop=True)

    def _output(self, x: pd.DataFrame, category: Any, confidence: Any) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "transaction_id": x["transaction_id"].to_numpy(),
                "category": category,
                "confidence": confidence,
                "model_version": self.version,
            }
        )
