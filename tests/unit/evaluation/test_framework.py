"""End to end on the toy task: run -> leaderboard -> finalize -> promote -> load_service."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from smart_financial_coach.evaluation.experiment import ExperimentConfig
from smart_financial_coach.evaluation.promote import (
    FINALIST_TAG,
    OVERRIDE_TAG,
    RANK_TAG,
    SelectionError,
    finalize,
    leaderboard,
    promote,
)
from smart_financial_coach.evaluation.runner import run_experiment
from smart_financial_coach.evaluation.tracking import Tracker
from smart_financial_coach.intelligence.models import ContractError
from smart_financial_coach.intelligence.models.artifact import (
    POINTER_FILE,
    promotion_errors,
    promotions,
)
from smart_financial_coach.intelligence.service import load_service

MakeConfig = Callable[..., ExperimentConfig]
NOTE = "group lookup with a hint fallback; no external dependencies"


@pytest.fixture
def runs(toy_data: Path, tracker: Tracker, make_config: MakeConfig) -> dict[str, str]:
    """A baseline and three candidates on the same data and splits."""
    configs = [
        make_config("majority", "toy/majority", baseline=True),
        make_config("memory_hint", grid={"confidence": [0.7, 0.9]}),
        make_config(
            "memory_no_hint", model={"type": "toy/memory", "params": {"trust_hint": False}}
        ),
        make_config("memory_hint_complex", complexity=3),
    ]
    return {c.name: run_experiment(c, toy_data, tracker).run_id for c in configs}


def test_run_logs_validation_only(toy_data: Path, tracker: Tracker, runs: dict[str, str]) -> None:
    run = tracker.get(runs["memory_hint"])

    assert run.metrics["val_unseen_accuracy"] > 0.7
    assert run.metrics["val_seen_accuracy"] == 1.0
    assert not [k for k in run.metrics if k.startswith("test_")]
    for tag in ("sfc.config_hash", "sfc.data_hash", "sfc.split_hash", "sfc.git_commit"):
        assert run.tags[tag]
    assert run.params["chosen.confidence"] in {"0.7", "0.9"}
    assert len({tracker.get(r).tags["sfc.split_hash"] for r in runs.values()}) == 1


def test_identical_run_is_skipped_unless_forced(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig, runs: dict[str, str]
) -> None:
    again = run_experiment(make_config("memory_hint_complex", complexity=3), toy_data, tracker)
    forced = run_experiment(
        make_config("memory_hint_complex", complexity=3), toy_data, tracker, force=True
    )

    assert again.skipped
    assert again.run_id == runs["memory_hint_complex"]
    assert not forced.skipped
    assert forced.run_id != again.run_id


def test_rerun_gives_identical_metrics(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig, runs: dict[str, str]
) -> None:
    rerun = run_experiment(
        make_config("memory_hint_complex", complexity=3), toy_data, tracker, force=True
    )
    first = tracker.get(runs["memory_hint_complex"]).metrics

    assert {k: v for k, v in rerun.metrics.items() if k.startswith("val_")} == {
        k: v for k, v in first.items() if k.startswith("val_")
    }


def test_leaderboard_orders_by_rule(toy_data: Path, tracker: Tracker, runs: dict[str, str]) -> None:
    standings = leaderboard("toy", toy_data, tracker)
    names = [s.name for s in standings]

    # Hint models tie on unseen accuracy; the simpler one wins the tie. No baselines listed.
    assert names[:2] == ["memory_hint", "memory_hint_complex"]
    assert standings[1].tied_with_leader
    assert names[-1] == "memory_no_hint"
    assert "majority" not in names


def test_finalize_takes_the_rules_top_three(
    toy_data: Path, tracker: Tracker, runs: dict[str, str]
) -> None:
    scored = finalize("toy", toy_data, tracker)
    ranks = {name: tracker.get(runs[name]).tags.get(RANK_TAG) for name in runs}

    assert set(scored) == set(runs.values())  # three candidates + the baseline
    assert ranks == {
        "majority": None,
        "memory_hint": "1",
        "memory_hint_complex": "2",
        "memory_no_hint": "3",
    }
    assert scored[runs["memory_hint"]]["test_known_accuracy"] == 1.0
    assert FINALIST_TAG not in tracker.get(runs["majority"]).tags


def test_test_sets_are_used_once_per_split(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig, runs: dict[str, str]
) -> None:
    """A second call can't test-score more runs, even new ones, without a recorded reason."""
    finalize("toy", toy_data, tracker)
    late = run_experiment(make_config("late", "toy/memory_alt"), toy_data, tracker).run_id

    with pytest.raises(SelectionError, match="already used"):
        finalize("toy", toy_data, tracker)
    with pytest.raises(SelectionError, match="override reason"):
        finalize("toy", toy_data, tracker, run_ids=[late])
    with pytest.raises(SelectionError, match="already scored"):
        finalize("toy", toy_data, tracker, run_ids=[runs["memory_hint"]], override="recheck")

    finalize("toy", toy_data, tracker, run_ids=[late], override="new model type added late")

    tags = tracker.get(late).tags
    assert tags[OVERRIDE_TAG] == "new model type added late"
    assert tags[RANK_TAG] == "1"  # its place in the rule's order: it ties the leader, simpler


def test_finalize_refusals(toy_data: Path, tracker: Tracker, runs: dict[str, str]) -> None:
    with pytest.raises(SelectionError, match="1 to 3"):
        finalize("toy", toy_data, tracker, run_ids=list(runs.values()), override="all")
    with pytest.raises(SelectionError, match="baseline"):
        finalize("toy", toy_data, tracker, run_ids=[runs["majority"]], override="x")


def test_finalize_needs_baselines(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig
) -> None:
    run_experiment(make_config("alone"), toy_data, tracker)

    with pytest.raises(SelectionError, match="no baseline"):
        finalize("toy", toy_data, tracker)


def test_finalize_refuses_other_data(
    toy_data: Path, tracker: Tracker, runs: dict[str, str], tmp_path: Path
) -> None:
    other = tmp_path / "other.csv"
    pd.read_csv(toy_data).iloc[::-1].to_csv(other, index=False)

    with pytest.raises(SelectionError, match="not the dataset"):
        finalize("toy", other, tracker)


def test_promote_then_load_service(
    toy_data: Path, tracker: Tracker, runs: dict[str, str], tmp_path: Path
) -> None:
    artifacts = tmp_path / "artifacts"
    finalize("toy", toy_data, tracker)
    entry = promote("toy", runs["memory_hint"], NOTE, tracker, artifacts)

    served = load_service("toy", artifacts)
    x = pd.DataFrame({"id": ["n1"], "group": ["g05"], "hint": ["c"]})

    assert served.predict(x)["label"].tolist() == ["c"]  # g05 is a "c" group
    assert served.version == entry["version"]
    assert "override" not in entry
    assert (artifacts / "toy" / POINTER_FILE).read_text().strip() == entry["version"]
    assert promotions(artifacts / "toy")[-1]["mlflow_run_id"] == runs["memory_hint"]
    assert promotion_errors(artifacts / "toy") == []
    assert tracker.champion_run("toy") == runs["memory_hint"]


def test_switching_models_needs_no_code_change(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig, runs: dict[str, str], tmp_path: Path
) -> None:
    """Promote one model type, then another: the caller's code is the same two lines.

    Promoting a runner-up departs from the decision rule, so it carries a recorded override.
    """
    artifacts = tmp_path / "artifacts"
    alt = run_experiment(make_config("alt", "toy/memory_alt", complexity=1), toy_data, tracker)
    finalize("toy", toy_data, tracker)
    x = pd.DataFrame({"id": ["n1"], "group": ["g05"], "hint": ["c"]})

    promote("toy", runs["memory_hint"], NOTE, tracker, artifacts)
    first = load_service("toy", artifacts)
    first_out = first.predict(x)
    promote("toy", alt.run_id, NOTE, tracker, artifacts, override="demonstrate a model switch")
    second = load_service("toy", artifacts)
    second_out = second.predict(x)

    assert (first.model.name, second.model.name) == ("toy/memory", "toy/memory_alt")
    assert first_out["confidence"].tolist() != second_out["confidence"].tolist()
    log = promotions(artifacts / "toy")
    assert [e.get("override") for e in log] == [None, "demonstrate a model switch"]
    assert promotion_errors(artifacts / "toy") == []
    assert tracker.champion_run("toy") == alt.run_id


def test_promote_refusals(
    toy_data: Path, tracker: Tracker, runs: dict[str, str], tmp_path: Path
) -> None:
    artifacts = tmp_path / "artifacts"
    with pytest.raises(SelectionError, match="not a finalist"):
        promote("toy", runs["memory_hint"], NOTE, tracker, artifacts)

    finalize("toy", toy_data, tracker)
    with pytest.raises(SelectionError, match="finalist #2, not the decision rule's winner"):
        promote("toy", runs["memory_hint_complex"], NOTE, tracker, artifacts)
    with pytest.raises(SelectionError, match="note"):
        promote("toy", runs["memory_hint"], "  ", tracker, artifacts)
    with pytest.raises(SelectionError, match="unseen_accuracy"):  # unseen falls to majority
        promote("toy", runs["memory_no_hint"], NOTE, tracker, artifacts, override="try #3")
    assert not (artifacts / "toy").exists()


def test_broken_model_fails_in_validation(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig
) -> None:
    with pytest.raises(ContractError, match="rows for"):
        run_experiment(make_config("broken", "toy/broken"), toy_data, tracker)
    assert tracker.find("toy", {}) == []  # the failed run isn't a finished experiment


def test_reproduce_poc_only_for_poc_configs(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig
) -> None:
    with pytest.raises(ValueError, match="not a POC configuration"):
        run_experiment(make_config("other", complexity=1), toy_data, tracker, reproduce_poc=True)

    result = run_experiment(make_config("poc"), toy_data, tracker, reproduce_poc=True)

    assert result.metrics["test_known_accuracy"] == 1.0
    assert tracker.find("toy", {}) == []  # kept off the leaderboard and out of finalize


def test_reproduction_runs_resume(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig
) -> None:
    first = run_experiment(make_config("poc"), toy_data, tracker, reproduce_poc=True)
    again = run_experiment(make_config("poc"), toy_data, tracker, reproduce_poc=True)

    assert again.skipped
    assert again.run_id == first.run_id
