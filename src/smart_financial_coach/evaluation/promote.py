"""From runs to a served model: leaderboard, finalize (test sets, once), promote.

- `leaderboard` orders comparable runs (same data and splits) by the decision rule, on
  validation metrics only.
- `finalize` scores the leaderboard's top three on the test sets, plus the round 0 baselines,
  once per split.
- `promote` checks the rank-1 finalist against the task's gates, exports it to
  `artifacts/<task>/`, publishes the model file (a GitHub Release, see `publish`), points the
  MLflow `champion` alias and `PROMOTED` at it, and appends to `promotions.jsonl`.
- Departing from the rule (naming finalists, a second round, promoting another finalist) needs an
  override reason, recorded on the run and in the promotion log.
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
from smart_financial_coach.evaluation.publish import Publisher
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
    URL_KEY,
    load_artifact,
    read_manifest,
    record_promotion,
)
from smart_financial_coach.intelligence.models.base import Model
from smart_financial_coach.intelligence.models.contract import Checked
from smart_financial_coach.intelligence.service import get_service

FINALIST_TAG = "sfc.finalist"
RANK_TAG = "sfc.finalist_rank"  # the run's place in the decision rule's order when finalized
TEST_SCORED_TAG = "sfc.test_scored"
OVERRIDE_TAG = "sfc.override"  # why a run departed from the rule, when it did


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class Standing:
    run_id: str
    name: str
    estimate: float
    eligible: bool
    tied_with_leader: bool
    code: str  # the code version that produced the run (see `runner.code_version`)


def _config(tracker: Tracker, run: RunRecord, tmp: Path) -> ExperimentConfig:
    path = tracker.download(run.run_id, CONFIG_FILE, tmp / run.run_id)
    return ExperimentConfig.model_validate_json(path.read_text(encoding="utf-8"))


def _model(tracker: Tracker, run: RunRecord, tmp: Path) -> Model:
    folder = tracker.download(run.run_id, MODEL_PATH, tmp / run.run_id)
    return load_artifact(folder, trusted_root=tmp)


def rebuild_splits(
    task: Task, data: Path, tracker: Tracker, run: RunRecord, tmp: Path
) -> tuple[Examples, Splits]:
    """The run's examples and splits, rebuilt from the data and checked against the run's tags."""
    examples, splits = prepare(task, data, _config(tracker, run, tmp))
    if examples.data_hash != run.tags["sfc.data_hash"]:
        raise SelectionError(f"{data} is not the dataset run {run.run_id} was trained on")
    if splits.hash() != run.tags["sfc.split_hash"]:
        raise SelectionError(f"rebuilt splits don't match run {run.run_id}'s split hash")
    return examples, splits


def comparable_runs(tracker: Tracker, task: Task, split_hash: str | None) -> list[RunRecord]:
    """Runs on the same data and splits as `split_hash` (default: the latest run's)."""
    runs = tracker.find(task.name, {})
    if split_hash is None and runs:
        split_hash = runs[-1].tags["sfc.split_hash"]
    same = [r for r in runs if r.tags.get("sfc.split_hash") == split_hash]
    data = same[-1].tags.get("sfc.data_hash") if same else None
    return [r for r in same if r.tags.get("sfc.data_hash") == data]


def code_versions(runs: list[RunRecord]) -> set[str]:
    return {r.tags.get("sfc.code_version", "unknown") for r in runs}


def _require_baselines(task: Task, runs: list[RunRecord]) -> None:
    present = {r.name for r in baseline_runs(runs)}
    if missing := [b for b in task.required_baselines if b not in present]:
        raise SelectionError(
            f"no {', '.join(missing)} baseline run on these splits; run round 0 first "
            "(eligibility and the gates compare against it)"
        )


def baseline_runs(runs: list[RunRecord]) -> list[RunRecord]:
    return [r for r in runs if r.tags.get("sfc.baseline") == "true"]


def leaderboard(
    task_name: str, data: Path, tracker: Tracker, split_hash: str | None = None
) -> list[Standing]:
    """Candidate runs in decision-rule order (baselines are the floor, not candidates)."""
    task = get_task(task_name)
    runs = comparable_runs(tracker, task, split_hash)
    if any(r.tags.get("sfc.baseline") != "true" for r in runs):
        _require_baselines(task, runs)
    baselines = {r.name: r.metrics for r in baseline_runs(runs)}
    candidates = [r for r in runs if r.tags.get("sfc.baseline") != "true"]
    if not candidates:
        return []
    metric = f"val_{task.selection_metric}"
    keys = (*task.tiebreak_metrics, "complexity")
    by_id = {r.run_id: r for r in candidates}
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        examples, _ = rebuild_splits(task, data, tracker, candidates[0], tmp)
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
            code=r.tags.get("sfc.code_version", "unknown"),
        )
        for i, r in enumerate(ranked + unranked)
    ]


def finalize(
    task_name: str,
    data: Path,
    tracker: Tracker,
    *,
    run_ids: list[str] | None = None,
    override: str | None = None,
    split_hash: str | None = None,
) -> dict[str, dict[str, float]]:
    """Score the decision rule's top three, and the baselines, on the test sets: once per split.

    The finalists are the leaderboard's first `MAX_FINALISTS`, tagged with their rank. A split whose
    candidates were already test-scored is refused. Naming other runs, or a second round on the
    same split, needs `override`: a reason, recorded on every run it scores.
    """
    task = get_task(task_name)
    reason = (override or "").strip()
    if run_ids and not reason:
        raise SelectionError(
            "naming finalists departs from the decision rule; give an override reason"
        )
    if run_ids:
        split_hash = tracker.get(run_ids[0]).tags.get("sfc.split_hash")
    _require_baselines(task, comparable_runs(tracker, task, split_hash))  # else nothing is eligible
    order = [s.run_id for s in leaderboard(task_name, data, tracker, split_hash) if s.eligible]
    chosen = run_ids or order[:MAX_FINALISTS]
    if not chosen:
        raise SelectionError("no eligible candidate runs on these splits; see `leaderboard`")
    if not 0 < len(chosen) <= MAX_FINALISTS:
        raise SelectionError(f"finalize takes 1 to {MAX_FINALISTS} runs, got {len(chosen)}")
    finalists = [tracker.get(r) for r in chosen]
    for run in finalists:
        if run.status != "FINISHED" or run.tags.get("sfc.kind") != "experiment":
            raise SelectionError(f"{run.run_id} is not a finished experiment run")
        if run.tags.get("sfc.baseline") == "true":
            raise SelectionError(f"{run.run_id} is a baseline; baselines are scored automatically")
    if len({r.tags["sfc.split_hash"] for r in finalists}) > 1:
        raise SelectionError("finalists were trained on different splits")
    comparable = comparable_runs(tracker, task, finalists[0].tags["sfc.split_hash"])
    used = [
        r
        for r in comparable
        if r.tags.get(TEST_SCORED_TAG) == "true" and r.tags.get("sfc.baseline") != "true"
    ]
    if used and not reason:
        raise SelectionError(
            f"the test sets were already used on these splits ({len(used)} runs); "
            "a second round needs an override reason"
        )
    if again := [r.run_id for r in finalists if r.tags.get(TEST_SCORED_TAG) == "true"]:
        raise SelectionError(f"{', '.join(again)} already scored on the test sets")
    baselines = baseline_runs(comparable)
    if not baselines:
        raise SelectionError("no baseline runs on these splits; run the round 0 baselines first")
    if len(versions := code_versions(finalists + baselines)) > 1 and not reason:
        raise SelectionError(
            f"finalists and baselines come from {len(versions)} code versions; rerun them on one "
            "(`sfc-experiment run --force`) or give an override reason"
        )

    contract = get_service(task.name).contract
    scored: dict[str, dict[str, float]] = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        examples, splits = rebuild_splits(task, data, tracker, finalists[0], tmp)
        for run in finalists + baselines:
            if run in baselines and run.tags.get(TEST_SCORED_TAG) == "true":
                scored[run.run_id] = {k: v for k, v in run.metrics.items() if k.startswith("test_")}
                continue
            model = Checked(_model(tracker, run, tmp), contract)
            test = task.test_metrics(examples, splits, model)
            metrics = {f"test_{k}": v for k, v in test.items()}
            tags = {TEST_SCORED_TAG: "true"}
            if run in finalists:
                rank = order.index(run.run_id) + 1 if run.run_id in order else "unranked"
                tags |= {FINALIST_TAG: "true", RANK_TAG: str(rank)}
            if reason:
                tags[OVERRIDE_TAG] = reason
            tracker.log(run.run_id, metrics=metrics, tags=tags)
            scored[run.run_id] = metrics
    return scored


def check_gates(task: Task, run: RunRecord, tracker: Tracker) -> list[Gate]:
    baselines = baseline_runs(comparable_runs(tracker, task, run.tags["sfc.split_hash"]))
    return task.gates(run.metrics, {b.name: b.metrics for b in baselines})


def promote(
    task_name: str,
    run_id: str,
    note: str,
    tracker: Tracker,
    artifacts_dir: Path | None = None,
    *,
    override: str | None = None,
    publisher: Publisher | None = None,
) -> dict[str, Any]:
    """Export the rule's winner, if it passes every gate, and make it what `load_service` returns.

    Only the rank-1 finalist is promoted: finalists are never re-ranked on test scores. Promoting
    another needs `override`, a reason recorded in the promotion log.

    `publisher` makes the model file downloadable and its URL goes in the log. Without one, the
    file stays on this machine, and other clones can't load the promoted model.
    """
    task = get_task(task_name)
    run = tracker.get(run_id)
    reason = (override or "").strip()
    if run.tags.get(FINALIST_TAG) != "true":
        raise SelectionError(f"{run_id} is not a finalist; run `finalize` first")
    if run.tags.get(RANK_TAG) != "1" and not reason:
        raise SelectionError(
            f"{run_id} is finalist #{run.tags.get(RANK_TAG)}, not the decision rule's winner; "
            "promoting it needs an override reason"
        )
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

    url = None
    if publisher is not None:
        notes = (
            f"{task.name} model {model.version}, MLflow run {run_id}. Loaded only if it matches "
            f"artifacts/{task.name}/{model.version}/{MANIFEST_FILE}."
        )
        # The release is tagged at the training commit only if that commit *is* the training code
        commit = run.tags.get("sfc.git_commit")
        clean = run.tags.get("sfc.git_dirty") == "false"
        if commit and not clean:
            notes += (
                f" Trained from uncommitted changes on top of commit {commit}, so this release "
                "isn't tagged at it; the manifest's code version identifies the code."
                if run.tags.get("sfc.git_dirty") == "true"
                else f" Whether commit {commit} had uncommitted changes wasn't recorded, so this "
                "release isn't tagged at it."
            )
        url = publisher.publish(
            f"{task.name}-{model.version}",
            target / MODEL_FILE,
            notes,
            commit=commit if clean else None,
        )
    registry_version = tracker.register_champion(task.name, run, model.version)
    entry = {
        "version": model.version,
        "mlflow_run_id": run_id,
        "registry_version": registry_version,
        "promoted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail} for g in gates],
        "note": note.strip(),
        **({URL_KEY: url} if url else {}),
        **({"override": reason} if reason else {}),
    }
    record_promotion(service_dir, entry)
    return entry
