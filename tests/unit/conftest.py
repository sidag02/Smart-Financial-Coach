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


@pytest.fixture(autouse=True)
def goals_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test's app writes goal changes to its own file, never the repo's data/."""
    path = tmp_path / "goals.sqlite"
    monkeypatch.setenv("SFC_GOALS_DB", str(path))
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


@pytest.fixture(scope="session")
def flag_artifacts(small_sqlite: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An artifacts folder with a promoted FR-7 model: Rules, cut at precision 0.8 on the small
    data's own labels (a test fixture, not a measurement)."""
    from smart_financial_coach.data.labels import load_truth
    from smart_financial_coach.evaluation.tasks.unusual import labels_for
    from smart_financial_coach.intelligence.anomaly.contract import scoring_rows
    from smart_financial_coach.intelligence.anomaly.rules import Rules
    from smart_financial_coach.intelligence.anomaly.threshold import Thresholded
    from smart_financial_coach.intelligence.models.artifact import record_promotion, save_artifact

    txns = store.load_transactions(small_sqlite)
    rows = scoring_rows(txns, pool=txns)
    model = Thresholded(Rules(), precision=0.8)
    model.fit(rows, labels_for(load_truth(small_sqlite), rows["transaction_id"]))
    model.version = "fr7-test"
    root = tmp_path_factory.mktemp("artifacts")
    save_artifact(model, root / "unusual_transactions" / model.version, {})
    record_promotion(root / "unusual_transactions", {"version": model.version})
    return root


@pytest.fixture(scope="session")
def flagged_sources(
    sources: DataSources, flag_artifacts: Path, tmp_path_factory: pytest.TempPathFactory
) -> DataSources:
    """`sources` with the promoted FR-7 model's flags for the small dataset."""
    from smart_financial_coach.intelligence.anomaly.batch import flag_dataset

    flags = tmp_path_factory.mktemp("flags") / "flags.sqlite"
    flag_dataset(sources.dataset, flags, artifacts_dir=flag_artifacts)
    return DataSources(sources.dataset, sources.predictions, flags)
