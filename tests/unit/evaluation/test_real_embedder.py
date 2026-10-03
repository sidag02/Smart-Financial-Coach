"""The real embedding model end to end (downloads ~67 MB once): slow, run by the slow CI job.

The full POC reproduction on the default dataset takes ~15 minutes, longer than the slow job's
budget; it is run by hand (`sfc-experiment run configs/experiments/categorization/poc/
--data data/synthetic/default.sqlite --reproduce-poc`) and recorded in the evaluation report.
"""

from pathlib import Path

import pandas as pd
import pytest

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.evaluation.experiment import ExperimentConfig, load_experiment
from smart_financial_coach.evaluation.promote import finalize, promote
from smart_financial_coach.evaluation.runner import run_experiment, run_with_twin
from smart_financial_coach.evaluation.tasks import categorization
from smart_financial_coach.evaluation.tracking import Tracker
from smart_financial_coach.intelligence.categorization.contract import Categorizer
from smart_financial_coach.intelligence.service import load_service

pytestmark = pytest.mark.slow


def test_real_embedder_trains_promotes_and_serves(
    small_sqlite: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The plumbing, not the quality: the small spec's 6 test users can't measure FR-4's unseen
    # gate (0.70), which the FR-4 round judges on the default dataset. The gate still runs
    monkeypatch.setattr(categorization, "UNSEEN_GATE", 0.0)
    tracker = Tracker(f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=tmp_path / "art")
    for name in ("keyword", "lookup"):  # the committed round 0 configs, on this test's 3 folds
        committed = load_experiment(
            PROJECT_ROOT / f"configs/experiments/categorization/00_{name}.yaml"
        )
        run_experiment(
            committed.model_copy(update={"task_params": {"k": 3}}), small_sqlite, tracker
        )
    config = ExperimentConfig.model_validate(
        {
            "name": "linear_both",
            "task": "categorization",
            "model": {
                "type": "categorization/calibrated",
                "params": {"base": {"$model": {"type": "categorization/linear_text"}}},
            },
            "task_params": {"k": 3},
            "ship": {"task_params": {"label_noise": 0.0}},
        }
    )
    _, twin = run_with_twin(config, small_sqlite, tracker)
    assert twin is not None
    run = twin
    finalize("categorization", small_sqlite, tracker)  # scores the shipping twin
    entry = promote(
        "categorization", twin.run_id, "linear weights", tracker, tmp_path / "artifacts"
    )
    assert entry["task_params"] == {"label_noise": 0.0}
    assert "unseen_macro_f1" in {g["name"] for g in entry["gates"]}

    categorizer = load_service("categorization", tmp_path / "artifacts")  # verifies the file
    assert isinstance(categorizer, Categorizer)
    out = categorizer.categorize(transactions(["SQ *TACO BELL 1234", "DELTA AIR LINES"]))

    assert run.metrics["latency_batch_ms"] > 0  # reported (batch cost), not gated
    assert out["category"].tolist() == ["Dining", "Travel"]


def transactions(strings: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [str(i) for i in range(len(strings))],
            "user_id": "u",
            "ts": "2026-01-01 12:00",
            "amount": -20.0,
            "currency": "USD",
            "merchant_raw": strings,
            "channel": "card_present",
        }
    )
