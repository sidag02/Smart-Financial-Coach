from pathlib import Path

import pandas as pd
import pytest

from smart_financial_coach.access.ledger import DataSources
from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.data import store
from smart_financial_coach.data.generator import generate, load_spec
from smart_financial_coach.data.predictions import CategoryWriter


@pytest.fixture(scope="session")
def small_sqlite(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The small spec (30 users), generated once per session and written to SQLite."""
    path = tmp_path_factory.mktemp("data") / "small.sqlite"
    generate(load_spec(PROJECT_ROOT / "configs" / "data" / "small.yaml")).to_sqlite(path)
    return path


def stub_categories(txns: pd.DataFrame) -> pd.DataFrame:
    """Deterministic stand-in for the categorizer: Income for money in, a few spending categories,
    and every fifth transaction below the review threshold."""
    spend = ["Dining", "Groceries", "Housing", "Shopping"]
    rows = range(len(txns))
    return pd.DataFrame(
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


@pytest.fixture(scope="session")
def sources(small_sqlite: Path, tmp_path_factory: pytest.TempPathFactory) -> DataSources:
    """The small dataset with stub predictions, as the web app reads it."""
    predictions = tmp_path_factory.mktemp("predictions") / "predictions.sqlite"
    txns = store.load_transactions(small_sqlite)
    with CategoryWriter(predictions, {"model_version": "stub"}) as writer:
        writer.append(txns["user_id"], stub_categories(txns))
    return DataSources(small_sqlite, predictions)


@pytest.fixture(scope="session")
def two_users(small_sqlite: Path) -> tuple[str, str]:
    users = store.load_users(small_sqlite)
    test = users[users["split"] == "test"]["user_id"].tolist()
    return test[0], test[1]
