"""Run one experiment: split, choose params on validation folds, fit, validate, log to MLflow.

1. Load examples through the task; build the splits and run the leak checks (a leak stops the run).
2. For each grid point, fit a model per validation fold and pool its held-out predictions.
   The grid point with the best selection metric wins.
3. Fit the final model on all of `train` with the winning params; measure latency.
4. Log params, tags, `val_*` metrics, the pooled validation predictions and the model.

Test sets are not scored here; `finalize` does that once, for finalists. The one exemption is
`reproduce_poc`, which accepts only the task's published POC configurations and keeps the run off
the leaderboard.
"""

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.evaluation.experiment import ExperimentConfig
from smart_financial_coach.evaluation.splits import TRAIN, Splits, leak_errors
from smart_financial_coach.evaluation.tasks.base import Examples, Task, get_task, latency_ms
from smart_financial_coach.evaluation.tracking import KIND_TAG, MODEL_PATH, Tracker, flatten
from smart_financial_coach.intelligence.models.artifact import save_artifact
from smart_financial_coach.intelligence.models.contract import Checked
from smart_financial_coach.intelligence.models.registry import build
from smart_financial_coach.intelligence.service import get_service

CONFIG_FILE = "config.json"
PREDICTIONS_PATH = "validation"
PREDICTIONS_FILE = "predictions.parquet"
LATENCY_ROWS = 10_000


class LeakError(ValueError):
    pass


@dataclass(frozen=True)
class RunResult:
    run_id: str
    skipped: bool  # an identical run (config, data and splits) had already finished
    chosen: dict[str, Any]
    metrics: dict[str, float]


def git_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    commit = git("rev-parse", "HEAD") or "unknown"
    if not git("status", "--porcelain", "--untracked-files=no"):
        return commit
    # Uncommitted changes: name them by their diff, so a resumed sweep skips a run only if the
    # code is byte-for-byte what produced it
    diff = hashlib.sha256(git("diff", "HEAD").encode()).hexdigest()[:12]
    return f"{commit}-dirty-{diff}"


def prepare(task: Task, data: Path, config: ExperimentConfig) -> tuple[Examples, Splits]:
    examples = task.load(data)
    splits = task.split(examples, config.task_params, config.seed)
    errors = leak_errors(splits, examples.frame, id_column=examples.id_column)
    errors += task.leak_errors(examples, splits)
    if errors:
        raise LeakError("; ".join(errors))
    return examples, splits


def identity(config: ExperimentConfig, examples: Examples, splits: Splits) -> dict[str, str]:
    """Everything that changes a run's results: config, data content, splits and code.

    Resume skips a run only if all four match. The leaderboard compares runs on the same data and
    splits, and flags (and `finalize` refuses) a mix of code versions.
    """
    return {
        "sfc.config_hash": config.config_hash(),
        "sfc.data_hash": examples.data_hash,
        "sfc.split_hash": splits.hash(),
        "sfc.git_commit": git_commit(),
    }


def model_version(config: ExperimentConfig, examples: Examples) -> str:
    return f"{config.config_hash()[:8]}-{examples.data_hash[:8]}"


def cross_fit(
    task: Task, examples: Examples, splits: Splits, config: ExperimentConfig, point: dict[str, Any]
) -> pd.DataFrame:
    """Pooled out-of-fold predictions, with `fold` and `held_out` (which held-out set) columns."""
    contract = get_service(task.name).contract
    parts = []
    for i, fold in enumerate(splits.folds):
        x, y = task.training_rows(examples, fold.train, config.task_params, config.seed + i)
        model = Checked(build(config.model_spec(point)), contract).fit(x, y)
        for name, which in sorted(fold.held_out.items()):
            out = model.predict(examples.rows(which))
            parts.append(out.assign(fold=i, held_out=name))
    return pd.concat(parts, ignore_index=True)


def _best(scored: list[tuple[dict[str, Any], dict[str, float]]], metric: str) -> int:
    def score(i: int) -> float:
        value = scored[i][1].get(metric, float("nan"))
        return value if value == value else float("-inf")  # NaN ranks last

    return max(range(len(scored)), key=lambda i: (score(i), -i))


def run_experiment(
    config: ExperimentConfig,
    data: Path,
    tracker: Tracker,
    *,
    force: bool = False,
    reproduce_poc: bool = False,
) -> RunResult:
    task = get_task(config.task)
    if reproduce_poc and config.config_hash() not in task.reproduction_configs():
        raise ValueError(f"{config.name} is not a POC configuration; --reproduce-poc refuses it")
    examples, splits = prepare(task, data, config)
    tags = identity(config, examples, splits)
    kind = "reproduction" if reproduce_poc else "experiment"
    if not force and (done := tracker.find(task.name, {**tags, KIND_TAG: kind})):
        previous = done[-1]
        return RunResult(previous.run_id, True, {}, previous.metrics)

    points = config.grid_points()
    if len(points) > 1 and not splits.folds:
        raise ValueError(f"task {task.name} has no validation folds to choose among grid points")
    tags |= {
        KIND_TAG: kind,
        "sfc.task": task.name,
        "sfc.model_type": config.model.type,
        "sfc.baseline": str(config.baseline).lower(),
        **{f"user.{k}": v for k, v in config.tags.items()},
    }
    with tracker.run(task.name, config.name, tags) as run_id:
        tracker.log(
            run_id,
            params=flatten(config.model_dump(mode="json", exclude={"tags", "name", "task"})),
            metrics={"complexity": float(config.complexity)},
        )
        scored: list[tuple[dict[str, Any], dict[str, float]]] = []
        pooled: list[pd.DataFrame] = []
        for point in points:
            predictions = cross_fit(task, examples, splits, config, point) if splits.folds else None
            metrics = {} if predictions is None else task.validation_metrics(examples, predictions)
            if len(points) > 1:
                with tracker.run(task.name, f"{config.name} {point}", {}, parent=run_id) as child:
                    tracker.log(
                        child,
                        params={k: str(v) for k, v in point.items()},
                        metrics={f"val_{k}": v for k, v in metrics.items()},
                    )
            scored.append((point, metrics))
            if predictions is not None:
                pooled.append(predictions)
        best = _best(scored, task.selection_metric)
        chosen, val = scored[best]

        x, y = task.training_rows(examples, splits.sets[TRAIN], config.task_params, config.seed)
        model = build(config.model_spec(chosen)).fit(x, y)
        model.version = model_version(config, examples)
        checked = Checked(model, get_service(task.name).contract)
        sample = examples.rows(splits.sets[TRAIN][:LATENCY_ROWS])
        metrics = {f"val_{k}": v for k, v in val.items()} | latency_ms(checked, sample)
        if reproduce_poc:
            test = task.test_metrics(examples, splits, checked)
            metrics |= {f"test_{k}": v for k, v in test.items()}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / CONFIG_FILE).write_text(config.model_dump_json(indent=2), encoding="utf-8")
            tracker.log_artifacts(run_id, out, "")
            if pooled:
                (out / PREDICTIONS_PATH).mkdir()
                pooled[best].to_parquet(out / PREDICTIONS_PATH / PREDICTIONS_FILE, index=False)
                tracker.log_artifacts(run_id, out / PREDICTIONS_PATH, PREDICTIONS_PATH)
            manifest = {
                "task": task.name,
                "config": json.loads(config.model_dump_json()),
                "chosen_params": chosen,
                "mlflow_run_id": run_id,
                "metrics": metrics,
                **{k.removeprefix("sfc."): v for k, v in tags.items() if k.startswith("sfc.")},
            }
            save_artifact(model, out / MODEL_PATH, manifest)
            tracker.log_artifacts(run_id, out / MODEL_PATH, MODEL_PATH)
        tracker.log(
            run_id,
            params={f"chosen.{k}": str(v) for k, v in chosen.items()},
            metrics=metrics,
            tags={"sfc.version": model.version},
        )
    return RunResult(run_id, False, chosen, metrics)
