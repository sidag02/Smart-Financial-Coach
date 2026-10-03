"""Shipping twins (FR-4 §1): comparison runs are ranked, their twins are finalized and promoted.

On `toy_ship`, training labels get injected noise unless a config sets `noise` explicitly, as
categorization does with `label_noise`.
"""

from pathlib import Path
from typing import Any

import pytest

from smart_financial_coach.evaluation.experiment import ExperimentConfig
from smart_financial_coach.evaluation.promote import (
    FINALIST_TAG,
    TEST_SCORED_TAG,
    SelectionError,
    finalize,
    leaderboard,
    promote,
)
from smart_financial_coach.evaluation.report import comparison_report
from smart_financial_coach.evaluation.runner import TWIN_TAG, run_experiment, run_with_twin
from smart_financial_coach.evaluation.tracking import Tracker

CLEAN = {"task_params": {"noise": 0.0}}
NOTE = "group lookup; no external dependencies"


def config(name: str, model: str = "toy/memory", **kwargs: Any) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {"name": name, "task": "toy_ship", "model": {"type": model}, **kwargs}
    )


def baseline(toy_data: Path, tracker: Tracker) -> None:
    run_experiment(config("majority", "toy/majority", baseline=True), toy_data, tracker)


def test_a_twin_is_the_config_with_its_shipping_params_written_out() -> None:
    twin = config("memory", task_params={"k": 3}, ship=CLEAN).twin()

    assert twin.name == "memory.ship"
    assert twin.task_params == {"k": 3, "noise": 0.0}
    assert twin.ship is None
    with pytest.raises(ValueError, match="no twin"):
        config("majority", "toy/majority", baseline=True, ship=CLEAN)


def test_twins_are_tagged_and_resume_with_their_comparison_run(
    toy_data: Path, tracker: Tracker
) -> None:
    first, twin = run_with_twin(config("memory", ship=CLEAN), toy_data, tracker)
    again, twin_again = run_with_twin(config("memory", ship=CLEAN), toy_data, tracker)
    forced, twin_forced = run_with_twin(config("memory", ship=CLEAN), toy_data, tracker, force=True)

    assert twin is not None
    assert twin_again is not None
    assert twin_forced is not None
    assert tracker.get(twin.run_id).tags[TWIN_TAG] == first.run_id
    assert again.skipped
    assert twin_again.skipped
    assert not twin_forced.skipped  # a new comparison run gets a new twin
    assert tracker.get(twin_forced.run_id).tags[TWIN_TAG] == forced.run_id


def test_finalize_scores_the_twins_of_ranked_comparison_runs(
    toy_data: Path, tracker: Tracker
) -> None:
    baseline(toy_data, tracker)
    comparison, twin = run_with_twin(config("memory", ship=CLEAN), toy_data, tracker)
    assert twin is not None

    standings = leaderboard("toy_ship", toy_data, tracker)
    scored = finalize("toy_ship", toy_data, tracker)

    assert [s.run_id for s in standings] == [comparison.run_id]  # twins aren't ranked
    assert standings[0].twin_id == twin.run_id
    assert twin.run_id in scored
    assert comparison.run_id not in scored
    assert tracker.get(twin.run_id).tags[FINALIST_TAG] == "true"
    assert TEST_SCORED_TAG not in tracker.get(comparison.run_id).tags


def test_a_finalist_without_a_twin_is_refused(toy_data: Path, tracker: Tracker) -> None:
    baseline(toy_data, tracker)
    run_experiment(config("memory"), toy_data, tracker)

    with pytest.raises(SelectionError, match="no finished shipping twin"):
        finalize("toy_ship", toy_data, tracker)


def test_promotion_needs_the_shipping_params_written_out(
    toy_data: Path, tracker: Tracker, tmp_path: Path
) -> None:
    baseline(toy_data, tracker)
    _, twin = run_with_twin(config("memory", ship={"task_params": {}}), toy_data, tracker)
    assert twin is not None
    finalize("toy_ship", toy_data, tracker)

    with pytest.raises(SelectionError, match="leaves noise to the task default"):
        promote("toy_ship", twin.run_id, NOTE, tracker, tmp_path / "artifacts")


def test_promotion_logs_the_shipping_params(
    toy_data: Path, tracker: Tracker, tmp_path: Path
) -> None:
    baseline(toy_data, tracker)
    _, twin = run_with_twin(config("memory", ship=CLEAN), toy_data, tracker)
    assert twin is not None
    finalize("toy_ship", toy_data, tracker)

    entry = promote("toy_ship", twin.run_id, NOTE, tracker, tmp_path / "artifacts")

    assert entry["task_params"] == {"noise": 0.0}


def test_a_twin_beating_rank_one_stops_finalize(toy_data: Path, tracker: Tracker) -> None:
    """Under noise the hint-trusting memory ranks first; clean, the pattern model wins."""
    baseline(toy_data, tracker)
    run_with_twin(config("memory", ship=CLEAN), toy_data, tracker)
    run_with_twin(config("pattern", "toy/pattern", ship=CLEAN), toy_data, tracker)

    standings = leaderboard("toy_ship", toy_data, tracker)
    report = comparison_report("toy_ship", toy_data, tracker)

    assert standings[0].name == "memory"
    assert [s.name for s in standings if s.reverses] == ["pattern"]
    assert "**reversal**" in report
    with pytest.raises(SelectionError, match="pattern beats rank 1's twin"):
        finalize("toy_ship", toy_data, tracker)
    finalize("toy_ship", toy_data, tracker, override="investigated: the noise hides the rule")


def test_the_report_says_when_no_candidate_has_a_twin(toy_data: Path, tracker: Tracker) -> None:
    """Runs from before twins sink on tie-breaks; the report says the order isn't the rule's."""
    baseline(toy_data, tracker)
    run_experiment(config("memory"), toy_data, tracker)

    assert "No candidate has a shipping twin" in comparison_report("toy_ship", toy_data, tracker)
