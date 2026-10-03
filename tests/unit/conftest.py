from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.access.ledger import DataSources
from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.data import store
from smart_financial_coach.data.generator import generate, load_spec
from smart_financial_coach.data.predictions import CategoryWriter
from smart_financial_coach.intelligence.categorization.review import ReviewPolicy

# Flags the stub categories: every 5th row (0.4) in both groups, and unfamiliar rows at 0.9
STUB_POLICY = ReviewPolicy("stub", familiar_threshold=0.5, unfamiliar_threshold=0.95)
STUB_META = {
    "model_version": "stub",
    "review_familiar_below": str(STUB_POLICY.familiar_threshold),
    "review_unfamiliar_below": str(STUB_POLICY.unfamiliar_threshold),
}


@pytest.fixture(autouse=True)
def feedback_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test's app writes category feedback to its own file, never the repo's data/."""
    path = tmp_path / "feedback.sqlite"
    monkeypatch.setenv("SFC_FEEDBACK_DB", str(path))
    return path


@pytest.fixture(scope="session")
def small_sqlite(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The small spec (30 users), generated once per session and written to SQLite."""
    path = tmp_path_factory.mktemp("data") / "small.sqlite"
    generate(load_spec(PROJECT_ROOT / "configs" / "data" / "small.yaml")).to_sqlite(path)
    return path


def stub_categories(txns: pd.DataFrame) -> pd.DataFrame:
    """Deterministic stand-in for the categorizer and its review flags: Income for money in, a few
    spending categories, every fifth transaction at low confidence, every third unfamiliar."""
    spend = ["Dining", "Groceries", "Housing", "Shopping"]
    rows = range(len(txns))
    out = pd.DataFrame(
        {
            "transaction_id": txns["transaction_id"].to_numpy(),
            "category": [
                "Income" if a > 0 else spend[i % len(spend)]
                for i, a in zip(rows, txns["amount"], strict=True)
            ],
            "confidence": [0.4 if i % 5 == 0 else 0.9 for i in rows],
            "model_version": "stub",
            "familiar": [i % 3 != 0 for i in rows],
        }
    )
    needs_review, reason = STUB_POLICY.flag(out["confidence"], out["familiar"])
    spending = (out["category"] != "Income").to_numpy()  # as the batch flags them
    return out.assign(
        needs_review=needs_review & spending, review_reason=np.where(spending, reason, "")
    )


@pytest.fixture(scope="session")
def sources(small_sqlite: Path, tmp_path_factory: pytest.TempPathFactory) -> DataSources:
    """The small dataset with stub predictions, as the web app reads it."""
    predictions = tmp_path_factory.mktemp("predictions") / "predictions.sqlite"
    txns = store.load_transactions(small_sqlite)
    with CategoryWriter(predictions, STUB_META) as writer:
        writer.append(txns["user_id"], stub_categories(txns))
    return DataSources(small_sqlite, predictions)


@pytest.fixture(scope="session")
def two_users(small_sqlite: Path) -> tuple[str, str]:
    users = store.load_users(small_sqlite)
    test = users[users["split"] == "test"]["user_id"].tolist()
    return test[0], test[1]
