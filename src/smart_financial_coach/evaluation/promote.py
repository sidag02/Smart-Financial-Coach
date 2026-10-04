"""From runs to a served model: leaderboard, finalize (test sets, once), promote.

- `leaderboard` orders comparable runs (same data and splits) by the decision rule, on
  validation metrics only.
- `finalize` scores the leaderboard's top three on the test sets, plus the round 0 baselines,
  once per split. When the task requires shipping twins (`Task.shipping_params`), the leaderboard
  ranks comparison runs, breaks ties on their twins' validation metrics, and `finalize` scores
  the twins; a twin that beats rank 1's twin on validation stops the round (FR-4 §1, §5).
- `promote` checks the rank-1 finalist against the task's gates, exports it to
  `artifacts/<task>/`, publishes the model file (a GitHub Release, see `publish`), points the
  MLflow `champion` alias and `PROMOTED` at it, and appends to `promotions.jsonl`.
- Departing from the rule (naming finalists, a second round, promoting another finalist) needs an
  override reason, recorded on the run and in the promotion log.
"""

import hashlib
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
    TWIN_TAG,
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
    record_attachment,
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
    twin_id: str | None = None  # its shipping twin, when the task requires twins
    twin_estimate: float = float("nan")  # the twin's validation selection metric
    reverses: bool = False  # its twin beats rank 1's twin on validation (paired interval > 0)


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


REPORT_ONLY_TAG = "user.report_only"  # a config's `tags: {report_only: "true"}`


def is_report_only(run: RunRecord) -> bool:
    """A run that's reported beside the round but never ranked, gated or test-scored, such as an
    ablation (FR-8 §3). A tag, so it doesn't change the config hash."""
    return run.tags.get(REPORT_ONLY_TAG) == "true"


def report_only_runs(runs: list[RunRecord]) -> list[RunRecord]:
    return [r for r in runs if is_report_only(r) and r.tags.get("sfc.baseline") != "true"]


def is_twin(run: RunRecord) -> bool:
    return TWIN_TAG in run.tags


def twins_of(runs: list[RunRecord]) -> dict[str, RunRecord]:
    """Comparison run ID -> its latest finished shipping twin."""
    out: dict[str, RunRecord] = {}
    for run in runs:  # oldest first, so the latest twin wins
        if is_twin(run) and run.status == "FINISHED":
            out[run.tags[TWIN_TAG]] = run
    return out


def leaderboard(
    task_name: str, data: Path, tracker: Tracker, split_hash: str | None = None
) -> list[Standing]:
    """Candidate runs in decision-rule order (baselines are the floor, not candidates).

    With shipping twins (FR-4): eligibility, ranking and the F1 tie set come from comparison
    runs; tie-breaks come from their twins' validation metrics, the first with its own tie test;
    and each eligible run is marked if its twin beats rank 1's twin on validation.
    """
    task = get_task(task_name)
    runs = comparable_runs(tracker, task, split_hash)
    if any(r.tags.get("sfc.baseline") != "true" for r in runs):
        _require_baselines(task, runs)
    baselines = {r.name: r.metrics for r in baseline_runs(runs)}
    candidates = [
        r
        for r in runs
        if r.tags.get("sfc.baseline") != "true" and not is_twin(r) and not is_report_only(r)
    ]
    if not candidates:
        return []
    shipping = bool(task.shipping_params)
    twins = twins_of(runs) if shipping else {}
    metric = f"val_{task.selection_metric}"
    by_id = {r.run_id: r for r in candidates}

    def breaker(run: RunRecord) -> RunRecord | None:
        """Whose metrics break ties: the twin's when the task ships twins (none: sinks)."""
        return twins.get(run.run_id) if shipping else run

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

        def tiebreak_tied(first: str, other: str) -> bool:
            a, b = breaker(by_id[first]), breaker(by_id[other])
            if a is None or b is None:
                return False
            return task.tiebreak_tied(examples, pooled(a.run_id), pooled(b.run_id))

        def tiebreak(run: RunRecord) -> tuple[float, ...]:
            source = breaker(run)
            values = source.metrics if source is not None else {}
            return (
                *(values.get(k, float("inf")) for k in task.tiebreak_metrics),
                run.metrics.get("complexity", float("inf")),
            )

        order = rank(
            [
                Candidate(
                    run_id=r.run_id,
                    estimate=r.metrics.get(metric, float("nan")),
                    eligible=metric in r.metrics and task.eligible(r.metrics, baselines),
                    tiebreak=tiebreak(r),
                )
                for r in candidates
            ],
            tied_with_leader,
            tiebreak_tied,
        )
        reversing: set[str] = set()
        if shipping and order and (top := twins.get(order[0])) is not None:
            for run_id in order[1:]:
                twin = twins.get(run_id)
                if twin is None or twin.metrics.get(metric, 0.0) <= top.metrics.get(metric, 0.0):
                    continue  # can only reverse if its twin's point estimate is higher
                lo, _ = task.difference_interval(examples, pooled(twin.run_id), pooled(top.run_id))
                if lo > 0:
                    reversing.add(run_id)
    ranked = [by_id[i] for i in order]
    unranked = [r for r in candidates if r.run_id not in order]
    standings = []
    for i, r in enumerate(ranked + unranked):
        twin = twins.get(r.run_id)
        standings.append(
            Standing(
                run_id=r.run_id,
                name=r.name,
                estimate=r.metrics.get(metric, float("nan")),
                eligible=r.run_id in order,
                tied_with_leader=i == 0 or r.run_id in tied,
                code=r.tags.get("sfc.code_version", "unknown"),
                twin_id=twin.run_id if twin else None,
                twin_estimate=twin.metrics.get(metric, float("nan")) if twin else float("nan"),
                reverses=r.run_id in reversing,
            )
        )
    return standings


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

    When the task requires shipping twins, the finalists' twins are scored, and a finalist
    without a finished twin is refused. If any eligible run's twin beats rank 1's twin on
    validation, nothing is finalized without an override reason: the reversal is investigated,
    not resolved by promoting another twin (FR-4 §1).
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
    standings = leaderboard(task_name, data, tracker, split_hash)
    order = [s.run_id for s in standings if s.eligible]
    if (reversed_by := [s.name for s in standings if s.eligible and s.reverses]) and not reason:
        raise SelectionError(
            f"the shipping twin of {', '.join(reversed_by)} beats rank 1's twin on validation; "
            "investigate before finalizing (an override needs a reason)"
        )
    chosen = run_ids or order[:MAX_FINALISTS]
    if not chosen:
        raise SelectionError("no eligible candidate runs on these splits; see `leaderboard`")
    if not 0 < len(chosen) <= MAX_FINALISTS:
        raise SelectionError(f"finalize takes 1 to {MAX_FINALISTS} runs, got {len(chosen)}")
    finalists = [tracker.get(r) for r in chosen]
    compared = {r.run_id: r.run_id for r in finalists}  # finalist -> the run it is ranked as
    if task.shipping_params:
        twins = twins_of(comparable_runs(tracker, task, finalists[0].tags.get("sfc.split_hash")))
        shipped = []
        for run in finalists:
            found = run if is_twin(run) else twins.get(run.run_id)
            if found is None:
                raise SelectionError(
                    f"{run.name} has no finished shipping twin; run its config again "
                    "(`sfc-experiment run`) so the twin is trained"
                )
            shipped.append(found)
            compared[found.run_id] = found.tags[TWIN_TAG]
        finalists = shipped
    for run in finalists:
        if run.status != "FINISHED" or run.tags.get("sfc.kind") != "experiment":
            raise SelectionError(f"{run.run_id} is not a finished experiment run")
        if run.tags.get("sfc.baseline") == "true":
            raise SelectionError(f"{run.run_id} is a baseline; baselines are scored automatically")
        if is_report_only(run):
            raise SelectionError(f"{run.run_id} is report-only; it's never scored on test sets")
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
                source = compared[run.run_id]
                rank = order.index(source) + 1 if source in order else "unranked"
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

    When the task requires shipping params (e.g. `label_noise`), the run's config must set each
    one explicitly, so a promotion can't silently train with or without noise; their values go
    in the log.
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
    shipped: dict[str, Any] = {}
    if task.shipping_params:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _config(tracker, run, Path(tmp_dir))
        if missing := [p for p in task.shipping_params if p not in config.task_params]:
            raise SelectionError(
                f"{run.name} leaves {', '.join(missing)} to the task default; promote a run whose "
                "config sets it explicitly (its shipping twin)"
            )
        shipped = {p: config.task_params[p] for p in task.shipping_params}
    gates = check_gates(task, run, tracker)
    if failed := [g for g in gates if not g.passed and g.blocking]:
        raise SelectionError("gates failed: " + "; ".join(f"{g.name} ({g.detail})" for g in failed))

    service_dir = (artifacts_dir or get_settings().artifacts_dir) / task.name
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = _model(tracker, run, Path(tmp_dir))  # verifies the checksum before export
        source = Path(tmp_dir) / run_id / MODEL_PATH
        target = service_dir / model.version
        if target.exists() and (
            read_manifest(target)["model_sha256"] != read_manifest(source)["model_sha256"]
        ):
            raise SelectionError(f"{target} exists with a different model")
        if missing := [n for n in task.serving_files_required if not (source / n).exists()]:
            raise SelectionError(
                f"run {run_id} has no {', '.join(missing)}; rerun it with the current code"
            )
        if not target.exists():
            target.mkdir(parents=True)
            for name in (MODEL_FILE, MANIFEST_FILE):
                shutil.copy2(source / name, target / name)
        for name in task.serving_files_required:
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
        "gates": [
            {"name": g.name, "passed": g.passed, "detail": g.detail, "blocking": g.blocking}
            for g in gates
        ],
        "note": note.strip(),
        **({"task_params": shipped} if shipped else {}),
        **({URL_KEY: url} if url else {}),
        **({"override": reason} if reason else {}),
    }
    record_promotion(service_dir, entry)
    return entry


def attach_serving_files(
    task_name: str,
    run_id: str,
    version: str,
    data: Path,
    tracker: Tracker,
    artifacts_dir: Path | None = None,
    *,
    note: str,
) -> list[Path]:
    """Give a promoted model the serving files its task now requires, from a run reproducing it.

    For models promoted before the task required them (FR-5's review policy for the FR-4 model).
    The run must have the promoted model's config and data, and match its validation metrics,
    so its pooled predictions are the promoted model's; it is usually a rerun of the same config
    on newer code. The files are derived for `version`, recording the run they came from and
    `note` (what a reader of the files should know), and each attachment is logged with the
    file's SHA-256, since the model's manifest doesn't cover it.
    """
    if not note.strip():
        raise SelectionError("an attachment needs a note: why, and what a reader should know")
    task = get_task(task_name)
    run = tracker.get(run_id)
    service_dir = (artifacts_dir or get_settings().artifacts_dir) / task.name
    target = service_dir / version
    manifest = read_manifest(target)
    for key in ("config_hash", "data_hash"):
        if manifest.get(key) != run.tags.get(f"sfc.{key}"):
            raise SelectionError(f"run {run_id} has a different {key} from {version}")
    promoted = {k: float(v) for k, v in manifest.get("metrics", {}).items() if k.startswith("val_")}
    if not promoted:
        raise SelectionError(f"{version}'s manifest records no validation metrics to compare")
    if differ := [
        k for k, v in promoted.items() if abs(run.metrics.get(k, float("nan")) - v) > 1e-9
    ]:
        raise SelectionError(
            f"run {run_id} doesn't reproduce {version}'s validation metrics ({', '.join(differ)})"
        )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        examples, _ = rebuild_splits(task, data, tracker, run, tmp)
        path = tracker.download(run_id, f"{PREDICTIONS_PATH}/{PREDICTIONS_FILE}", tmp / "pooled")
        pooled = pd.read_parquet(path)
    source = {"mlflow_run_id": run_id, "reproduces": version, "note": note.strip()}
    files = task.serving_files(examples, pooled, version, source)
    if missing := [n for n in task.serving_files_required if n not in files]:
        raise SelectionError(f"run {run_id}'s predictions give no {', '.join(missing)}")
    written = []
    for name in task.serving_files_required:
        (target / name).write_text(files[name], encoding="utf-8")
        written.append(target / name)
        record_attachment(
            service_dir,
            {
                "version": version,
                "file": name,
                "sha256": hashlib.sha256(files[name].encode("utf-8")).hexdigest(),
                "mlflow_run_id": run_id,
                "attached_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "note": note.strip(),
            },
        )
    return written
