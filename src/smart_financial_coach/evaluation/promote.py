"""From runs to a served model: leaderboard, finalize (test sets, once), promote.

- `leaderboard` orders comparable runs (same data and splits) by the decision rule, on
  validation metrics only.
- `finalize` scores at most three finalists on the test sets, plus the round 0 baselines, and
  records it. A run is scored on test once.
- `promote` checks a finalist against the task's gates, exports it to `artifacts/<task>/`,
  points the MLflow `champion` alias and `PROMOTED` at it, and appends to `promotions.jsonl`.
"""

import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from smart_financial_coach.config import get_settings
from smart_financial_coach.evaluation.experiment import ExperimentConfig
from smart_financial_coach.evaluation.runner import (
    CONFIG_FILE,
    PREDICTIONS_FILE,
    PREDICTIONS_PATH,
    prepare,
)
from smart_financial_coach.evaluation.selection import MAX_FINALISTS, Candidate, rank
from smart_financial_coach.evaluation.splits import Splits
from smart_financial_coach.evaluation.tasks.base import Examples, Gate, Task, get_task
from smart_financial_coach.evaluation.tracking import MODEL_PATH, RunRecord, Tracker
from smart_financial_coach.intelligence.models.artifact import (
    MANIFEST_FILE,
    MODEL_FILE,
    load_artifact,
    read_manifest,
    record_promotion,
)
from smart_financial_coach.intelligence.models.base import Model
from smart_financial_coach.intelligence.models.contract import Checked
from smart_financial_coach.intelligence.service import get_service

FINALIST_TAG = "sfc.finalist"
TEST_SCORED_TAG = "sfc.test_scored"


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class Standing:
    run_id: str
    name: str
    estimate: float
    eligible: bool
    tied_with_leader: bool


def _config(tracker: Tracker, run: RunRecord, tmp: Path) -> ExperimentConfig:
    path = tracker.download(run.run_id, CONFIG_FILE, tmp / run.run_id)
    return ExperimentConfig.model_validate_json(path.read_text(encoding="utf-8"))


def _model(tracker: Tracker, run: RunRecord, tmp: Path) -> Model:
    folder = tracker.download(run.run_id, MODEL_PATH, tmp / run.run_id)
    return load_artifact(folder, trusted_root=tmp)


def _rebuild(
    task: Task, data: Path, tracker: Tracker, run: RunRecord, tmp: Path
) -> tuple[Examples, Splits]:
    """The run's examples and splits, rebuilt from the data and checked against the run's tags."""
    examples, splits = prepare(task, data, _config(tracker, run, tmp))
    if examples.data_hash != run.tags["sfc.data_hash"]:
        raise SelectionError(f"{data} is not the dataset run {run.run_id} was trained on")
    if splits.hash() != run.tags["sfc.split_hash"]:
        raise SelectionError(f"rebuilt splits don't match run {run.run_id}'s split hash")
    return examples, splits


def _comparable(tracker: Tracker, task: Task, split_hash: str | None) -> list[RunRecord]:
    runs = tracker.find(task.name, {})
    if split_hash is None and runs:
        split_hash = runs[-1].tags["sfc.split_hash"]  # the latest run's splits
    return [r for r in runs if r.tags.get("sfc.split_hash") == split_hash]


def _baselines(runs: list[RunRecord]) -> list[RunRecord]:
    return [r for r in runs if r.tags.get("sfc.baseline") == "true"]


def leaderboard(
    task_name: str, data: Path, tracker: Tracker, split_hash: str | None = None
) -> list[Standing]:
    """Candidate runs in decision-rule order (baselines are the floor, not candidates)."""
    task = get_task(task_name)
    runs = _comparable(tracker, task, split_hash)
    baselines = {r.name: r.metrics for r in _baselines(runs)}
    candidates = [r for r in runs if r.tags.get("sfc.baseline") != "true"]
    if not candidates:
        return []
    metric = f"val_{task.selection_metric}"
    keys = (*task.tiebreak_metrics, "complexity")
    by_id = {r.run_id: r for r in candidates}
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        examples, _ = _rebuild(task, data, tracker, candidates[0], tmp)
        predictions: dict[str, pd.DataFrame] = {}

        def pooled(run_id: str) -> pd.DataFrame:
            if run_id not in predictions:
                path = tracker.download(
                    run_id, f"{PREDICTIONS_PATH}/{PREDICTIONS_FILE}", tmp / run_id
                )
                predictions[run_id] = pd.read_parquet(path)
            return predictions[run_id]

        tied: set[str] = set()

        def tied_with_leader(leader: str, other: str) -> bool:
            tied.add(leader)  # the leader is in its own tie set
            if task.tied(examples, pooled(leader), pooled(other)):
                tied.add(other)
                return True
            return False

        order = rank(
            [
                Candidate(
                    run_id=r.run_id,
                    estimate=r.metrics.get(metric, float("nan")),
                    eligible=metric in r.metrics and task.eligible(r.metrics, baselines),
                    tiebreak=tuple(r.metrics.get(k, float("inf")) for k in keys),
                )
                for r in candidates
            ],
            tied_with_leader,
        )
    ranked = [by_id[i] for i in order]
    unranked = [r for r in candidates if r.run_id not in order]
    return [
        Standing(
            run_id=r.run_id,
            name=r.name,
            estimate=r.metrics.get(metric, float("nan")),
            eligible=r.run_id in order,
            tied_with_leader=i == 0 or r.run_id in tied,
        )
        for i, r in enumerate(ranked + unranked)
    ]


def finalize(
    task_name: str, run_ids: list[str], data: Path, tracker: Tracker
) -> dict[str, dict[str, float]]:
    """Score finalists and baselines on the test sets; each run is scored once."""
    if not 0 < len(run_ids) <= MAX_FINALISTS:
        raise SelectionError(f"finalize takes 1 to {MAX_FINALISTS} runs, got {len(run_ids)}")
    task = get_task(task_name)
    finalists = [tracker.get(r) for r in run_ids]
    for run in finalists:
        if run.status != "FINISHED" or run.tags.get("sfc.kind") != "experiment":
            raise SelectionError(f"{run.run_id} is not a finished experiment run")
        if run.tags.get("sfc.baseline") == "true":
            raise SelectionError(f"{run.run_id} is a baseline; baselines are scored automatically")
        if run.tags.get(TEST_SCORED_TAG) == "true":
            raise SelectionError(f"{run.run_id} was already scored on the test sets")
    if len({r.tags["sfc.split_hash"] for r in finalists}) > 1:
        raise SelectionError("finalists were trained on different splits")
    baselines = _baselines(_comparable(tracker, task, finalists[0].tags["sfc.split_hash"]))
    if not baselines:
        raise SelectionError("no baseline runs on these splits; run the round 0 baselines first")

    contract = get_service(task.name).contract
    scored: dict[str, dict[str, float]] = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        examples, splits = _rebuild(task, data, tracker, finalists[0], tmp)
        for run in finalists + baselines:
            if run in baselines and run.tags.get(TEST_SCORED_TAG) == "true":
                scored[run.run_id] = {k: v for k, v in run.metrics.items() if k.startswith("test_")}
                continue
            model = Checked(_model(tracker, run, tmp), contract)
            test = task.test_metrics(examples, splits, model)
            metrics = {f"test_{k}": v for k, v in test.items()}
            tags = {TEST_SCORED_TAG: "true"}
            if run in finalists:
                tags[FINALIST_TAG] = "true"
            tracker.log(run.run_id, metrics=metrics, tags=tags)
            scored[run.run_id] = metrics
    return scored


def check_gates(task: Task, run: RunRecord, tracker: Tracker) -> list[Gate]:
    baselines = _baselines(_comparable(tracker, task, run.tags["sfc.split_hash"]))
    return task.gates(run.metrics, {b.name: b.metrics for b in baselines})


def promote(
    task_name: str,
    run_id: str,
    note: str,
    tracker: Tracker,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    """Export a finalist that passes every gate and make it what `load_service` returns."""
    task = get_task(task_name)
    run = tracker.get(run_id)
    if run.tags.get(FINALIST_TAG) != "true":
        raise SelectionError(f"{run_id} is not a finalist; run `finalize` on it first")
    if not note.strip():
        raise SelectionError("a promotion needs a note on explainability and operations")
    gates = check_gates(task, run, tracker)
    if failed := [g for g in gates if not g.passed]:
        raise SelectionError("gates failed: " + "; ".join(f"{g.name} ({g.detail})" for g in failed))

    service_dir = (artifacts_dir or get_settings().artifacts_dir) / task.name
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = _model(tracker, run, Path(tmp_dir))  # verifies the checksum before export
        source = Path(tmp_dir) / run_id / MODEL_PATH
        target = service_dir / model.version
        if target.exists():
            if read_manifest(target)["model_sha256"] != read_manifest(source)["model_sha256"]:
                raise SelectionError(f"{target} exists with a different model")
        else:
            target.mkdir(parents=True)
            for name in (MODEL_FILE, MANIFEST_FILE):
                shutil.copy2(source / name, target / name)

    registry_version = tracker.register_champion(task.name, run, model.version)
    entry = {
        "version": model.version,
        "mlflow_run_id": run_id,
        "registry_version": registry_version,
        "promoted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail} for g in gates],
        "note": note.strip(),
    }
    record_promotion(service_dir, entry)
    return entry
