"""The feedback replay end to end on the small dataset, with a stub embedder (#15 §7)."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from smart_financial_coach.data.store import load_transactions
from smart_financial_coach.evaluation.replay import (
    FEEDBACK_KIND,
    Behavior,
    ReplayConfig,
    retrain_step,
    run_replay,
)
from smart_financial_coach.evaluation.tracking import KIND_TAG, MODEL_PATH, Tracker
from smart_financial_coach.intelligence.categorization.agreement import AgreementRule
from smart_financial_coach.intelligence.categorization.baseline import Majority
from smart_financial_coach.intelligence.categorization.review import (
    ReviewPolicy,
    read_review_policy,
    save_review_policy,
)
from smart_financial_coach.intelligence.models.artifact import (
    POINTER_FILE,
    read_manifest,
    record_promotion,
    save_artifact,
)

STUB = {
    "name": "replay_stub",
    "task": "categorization",
    "model": {
        "type": "categorization/calibrated",
        "params": {
            "base": {
                "$model": {
                    "type": "categorization/linear_text",
                    "params": {"embeddings": "stub:16"},
                }
            }
        },
    },
    "task_params": {"k": 3},
    "ship": {"task_params": {"label_noise": 0}},
}


@pytest.fixture
def setup(small_sqlite: Path, tmp_path: Path) -> tuple[Path, Path]:
    """A promoted model that calls everything Dining at 0.4 (so much is flagged), and the
    shipped config with a stub embedder."""
    t = load_transactions(small_sqlite)
    labels = pd.Series(["Dining", "Shopping"] * (len(t) // 2 + 1))[: len(t)]
    model = Majority().fit(t, labels)
    root = tmp_path / "artifacts" / "categorization"
    save_artifact(model, root / model.version, {})
    save_review_policy(ReviewPolicy(model.version, 0.95, 0.95), root / model.version)
    record_promotion(root, {"version": model.version})
    experiment = tmp_path / "replay_stub.yaml"
    experiment.write_text(yaml.safe_dump(STUB))
    return tmp_path / "artifacts", experiment


@pytest.mark.slow  # about a minute: three cross-fitted retrainings
def test_the_replay_runs_the_loop_and_reports_it(
    small_sqlite: Path, setup: tuple[Path, Path]
) -> None:
    artifacts, experiment = setup
    config = ReplayConfig(
        behavior=Behavior(engagement=1.0, adversarial=0.0),
        rule=AgreementRule(n=1),  # a handful of feedback users in the small data
        first_retrain=4,
        every=6,
    )

    result = run_replay(small_sqlite, config, experiment=experiment, artifacts_dir=artifacts)

    assert result.users["feedback"] + result.users["evaluation"] == 6  # the test users
    assert min(result.users["feedback"], result.users["evaluation"]) >= 1
    assert result.users["adversarial"] == 0
    assert len(result.months) == 24
    assert result.summary["votes"] > 0
    assert result.summary["global_labels"] > 0
    assert result.retrainings, "a retraining was considered"
    trained = [s for s in result.retrainings if "gates" in s]
    assert trained, result.retrainings
    for step in trained:
        assert step["decision"] in ("promoted", "rejected")
        assert step["added_rows"] >= 0
        assert {g["name"] for g in step["gates"]} >= {"macro_f1_view", "unfamiliar_brier"}
    data = json.loads(result.to_json())  # the Learning page reads it
    for remap in data["remaps"]:  # category -> count, not mangled by dataclasses.asdict
        assert all(isinstance(n, int) and "/" not in c for c, n in remap["labels_now"].items())
    assert all("evaluation" in m for m in result.months)

    # A recorded retraining can be rebuilt as a tracked run, without promoting anything
    step = trained[0]
    assert set(step["inputs"]) == {"cutoff", "agreed", "contributors"}
    tracker = Tracker(
        f"sqlite:///{artifacts.parent / 'mlflow.db'}", artifact_root=artifacts.parent / "art"
    )
    pointer = (artifacts / "categorization" / POINTER_FILE).read_text()

    run_id = retrain_step(small_sqlite, data, step["month"], tracker, experiment=experiment)

    run = tracker.get(run_id)
    assert run.tags[KIND_TAG] == FEEDBACK_KIND  # not an experiment: never on the leaderboard
    assert run.params["global_labels"] == str(len(step["inputs"]["agreed"]))
    folder = tracker.download(run_id, MODEL_PATH, artifacts.parent / "dl")
    assert read_manifest(folder)["replay_month"] == step["month"]
    assert read_review_policy(folder).model_version == run.tags["sfc.version"]
    assert (artifacts / "categorization" / POINTER_FILE).read_text() == pointer  # not promoted
