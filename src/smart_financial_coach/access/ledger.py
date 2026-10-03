"""One user's categorized ledger: the only data the tools, the web app and the coach read.

A `Ledger` is built for a single `user_id` and every read under it goes through the data store's
user-scoped readers, so nothing built on it can reach another user's rows (Technical Design,
"Security and data isolation", step 3). Categories come from the predictions file of the promoted
model (FR-3); merchant display names from the shared normalizer, title-cased (Web App UI, gap 1).

    sources = DataSources.from_dir(settings.demo_dir)
    ledger = Ledger.load(sources, user_id)
    ledger.transactions  # ts, amount, merchant, merchant_raw, category, confidence, needs_review
"""

import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from smart_financial_coach.data import store
from smart_financial_coach.data.features.merchant_text import normalize_merchant
from smart_financial_coach.data.predictions import load_categories, load_prediction_meta

INCOME = "Income"

_COLUMNS = [
    "transaction_id",
    "ts",
    "amount",
    "currency",
    "merchant",
    "merchant_raw",
    "channel",
    "category",
    "confidence",
    "model_version",
    "familiar",
    "needs_review",  # by the model's review policy, computed in the batch (FR-5 §1)
    "review_reason",
]


@dataclass(frozen=True)
class DataSources:
    dataset: Path  # a generated dataset (model-visible tables are all that's read)
    predictions: Path  # the promoted categorizer's output for it

    @classmethod
    def from_dir(cls, root: Path) -> "DataSources":
        return cls(root / "dataset.sqlite", root / "predictions.sqlite")

    def as_of(self) -> date:
        """The dataset's last day: the app's "today" (Web App UI, gap 8)."""
        return date.fromisoformat(store.load_meta(self.dataset)["calendar_end"])

    def review_thresholds(self) -> tuple[float, float]:
        """The review policy the predictions were flagged with: (familiar, unfamiliar); a
        category is flagged below its group's threshold."""
        meta = load_prediction_meta(self.predictions)
        return float(meta["review_familiar_below"]), float(meta["review_unfamiliar_below"])

    def categories(self) -> list[str]:
        """The dataset's taxonomy, Income included."""
        return list(json.loads(store.load_meta(self.dataset)["categories"]))


def display_name(merchant_raw: str) -> str:
    """A readable merchant name: "POS DEBIT SQ *BLUE BOTTLE #4321" -> "Blue Bottle".

    A stand-in for real merchant names (Web App UI, gap 1): the normalizer's matching key,
    capitalized, with a trailing two-letter state code kept upper-case.
    """
    words = normalize_merchant(merchant_raw).split()
    last = len(words) - 1
    return " ".join(
        w.upper() if i == last and i > 0 and len(w) == 2 else _capitalize(w)
        for i, w in enumerate(words)
    )


def _capitalize(word: str) -> str:
    # "cvs/pharmacy" -> "Cvs/Pharmacy", but "peet's" -> "Peet's", not str.title's "Peet'S"
    return word.capitalize() if "'" in word else word.title()


@dataclass(frozen=True)
class Ledger:
    user_id: str
    timezone: str
    transactions: pd.DataFrame  # one row per transaction, newest first
    goals: pd.DataFrame
    as_of: date

    @classmethod
    def load(cls, sources: DataSources, user_id: str) -> "Ledger":
        return _load(sources, user_id)

    def between(self, start: date, end: date) -> pd.DataFrame:
        """Transactions with `start <= day <= end`."""
        day = self.transactions["day"]
        return self.transactions[(day >= start) & (day <= end)]


@lru_cache(maxsize=64)  # demo data is read-only, so a user's ledger never goes stale
def _load(sources: DataSources, user_id: str) -> Ledger:
    users = store.load_users(sources.dataset, user_id=user_id)
    if users.empty:
        raise LookupError(f"no user {user_id!r} in {sources.dataset}")
    txns = store.load_transactions(sources.dataset, user_id=user_id)
    cats = load_categories(sources.predictions, user_id=user_id)
    merged = txns.merge(cats.drop(columns="user_id"), on="transaction_id", how="left")
    if merged["category"].isna().any():
        missing = int(merged["category"].isna().sum())
        raise ValueError(f"{missing} of {user_id}'s transactions have no predicted category")
    merged["merchant"] = merged["merchant_raw"].map(display_name)
    merged = merged[_COLUMNS].copy()
    merged["ts"] = pd.to_datetime(merged["ts"])
    merged["day"] = merged["ts"].dt.date
    merged = merged.sort_values(["ts", "transaction_id"], ascending=False, ignore_index=True)
    return Ledger(
        user_id=user_id,
        timezone=str(users["timezone"].iloc[0]),
        transactions=merged,
        goals=store.load_goals(sources.dataset, user_id=user_id),
        as_of=sources.as_of(),
    )
