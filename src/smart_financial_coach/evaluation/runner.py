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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.evaluation.experiment import ExperimentConfig
from smart_financial_coach.evaluation.splits import TRAIN, Ids, Splits, leak_errors
from smart_financial_coach.evaluation.tasks.base import Examples, Task, get_task, latency_ms
from smart_financial_coach.evaluation.tracking import KIND_TAG, MODEL_PATH, Tracker, flatten
from smart_financial_coach.intelligence.models.artifact import save_artifact
from smart_financial_coach.intelligence.models.base import HeldOutFit
from smart_financial_coach.intelligence.models.contract import Checked
from smart_financial_coach.intelligence.models.registry import build
from smart_financial_coach.intelligence.service import get_service

CONFIG_FILE = "config.json"
PREDICTIONS_PATH = "validation"
PREDICTIONS_FILE = "predictions.parquet"
LATENCY_ROWS = 10_000
TWIN_TAG = "sfc.twin_of"  # on a shipping twin: the comparison run it was trained alongside


class LeakError(ValueError):
    pass


@dataclass(frozen=True)
class RunResult:
    run_id: str
    skipped: bool  # an identical run (config, data and splits) had already finished
    chosen: dict[str, Any]
    metrics: dict[str, float]


# What can change a run's results. Docs, tests and CI don't, so commits touching only them keep
# earlier runs valid (a full sweep takes hours). Not `configs/experiments/`: each run's own config
# is in its config hash, and adding a round's configs mustn't invalidate earlier rounds.
CODE_PATHS = ("src", "configs/models", "configs/data", "pyproject.toml", "uv.lock")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def git_commit() -> str:
    """The commit, for the record (`sfc.git_commit`); identity uses `code_version`."""
    return _git("rev-parse", "HEAD") or "unknown"


def git_dirty() -> bool:
    """Whether `CODE_PATHS` differ from `HEAD` (modified, staged or untracked), in which case `HEAD`
    isn't the code that trained the model (`sfc.git_dirty`; `code_version` hashes it either way)."""
    return bool(_git("status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS))


def code_version() -> str:
    """A hash of everything under `CODE_PATHS` as it is now: committed, modified or untracked.

    Committed content enters as git tree and blob hashes, uncommitted changes as their diff, and
    untracked files (a new module not yet `git add`ed) by name and content.
    """
    digest = hashlib.sha256()
    for path in CODE_PATHS:
        digest.update(f"{path}={_git('rev-parse', f'HEAD:{path}')}\n".encode())
    digest.update(
        _git("diff", "--binary", "HEAD", "--", *CODE_PATHS).encode()
    )  # binary content too
    # -z: NUL-separated and unquoted, so non-ASCII names resolve to files
    untracked = _git("ls-files", "-z", "--others", "--exclude-standard", "--", *CODE_PATHS)
    for name in sorted(n for n in untracked.split("\0") if n):
        digest.update(name.encode() + b"\0" + (PROJECT_ROOT / name).read_bytes())
    return digest.hexdigest()[:16]


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
        "sfc.code_version": code_version(),
    }


def model_version(config: ExperimentConfig, examples: Examples, code: str) -> str:
    """Names the fitted model: config, data and code version, so a retrain after a code fix gets a
    new version (and release tag) instead of a second file under the old one.

    Versions promoted before the code version was added have two parts (e.g. `3f0ccc82-2f0e60a6`).
    """
    return f"{config.config_hash()[:8]}-{examples.data_hash[:8]}-{code[:8]}"


@dataclass(frozen=True)
class Pooled:
    predictions: pd.DataFrame  # contract columns + `fold` + `held_out`, rows in fold order
    outputs: pd.DataFrame | None  # `HeldOutFit` models: their held-out outputs, same rows


def cross_fit(
    task: Task, examples: Examples, splits: Splits, config: ExperimentConfig, point: dict[str, Any]
) -> Pooled:
    """Pooled out-of-fold predictions, with `fold` and `held_out` (which held-out set) columns."""
    contract = get_service(task.name).contract
    parts, outputs = [], []
    for i, fold in enumerate(splits.folds):
        x, y = task.training_rows(examples, fold.train, config.task_params, config.seed + i)
        model = Checked(build(config.model_spec(point)), contract).fit(x, y)
        for name, which in sorted(fold.held_out.items()):
            rows = examples.rows(which)
            parts.append(model.predict(rows).assign(fold=i, held_out=name))
            if isinstance(model.model, HeldOutFit):
                outputs.append(model.model.held_out_outputs(rows).assign(fold=i))
    pooled = pd.concat(parts, ignore_index=True)
    return Pooled(pooled, pd.concat(outputs, ignore_index=True) if outputs else None)


def ids_in_order(frame: pd.DataFrame) -> Ids:
    """Row IDs in the frame's own order (not sorted): rows can repeat across folds."""
    return np.asarray(frame.iloc[:, 0].to_numpy(), dtype=str)


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
    extra_tags: Mapping[str, str] | None = None,
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
        "sfc.git_commit": git_commit(),
        "sfc.git_dirty": str(git_dirty()).lower(),
        **{f"user.{k}": v for k, v in config.tags.items()},
        **dict(extra_tags or {}),
    }
    with tracker.run(task.name, config.name, tags) as run_id:
        tracker.log(
            run_id,
            params=flatten(config.model_dump(mode="json", exclude={"tags", "name", "task"})),
            metrics={"complexity": float(config.complexity)},
        )
        scored: list[tuple[dict[str, Any], dict[str, float]]] = []
        pooled: list[Pooled] = []
        for point in points:
            result = cross_fit(task, examples, splits, config, point) if splits.folds else None
            metrics = (
                {} if result is None else task.validation_metrics(examples, result.predictions)
            )
            if len(points) > 1:
                with tracker.run(task.name, f"{config.name} {point}", {}, parent=run_id) as child:
                    tracker.log(
                        child,
                        params={k: str(v) for k, v in point.items()},
                        metrics={f"val_{k}": v for k, v in metrics.items()},
                    )
            scored.append((point, metrics))
            if result is not None:
                pooled.append(result)
        best = _best(scored, task.tuning_metric)
        chosen, val = scored[best]

        x, y = task.training_rows(examples, splits.sets[TRAIN], config.task_params, config.seed)
        model = build(config.model_spec(chosen)).fit(x, y)
        model.version = model_version(config, examples, tags["sfc.code_version"])
        held_out_report: dict[str, float] = {}
        if (
            pooled
            and isinstance(model, HeldOutFit)
            and (outputs := pooled[best].outputs) is not None
        ):
            # Fit held-out stages (calibration) on the pooled fold outputs with clean labels, then
            # recompute validation metrics on the out-of-fold calibrated confidences
            labels = examples.labels_for(ids_in_order(outputs))
            assert labels is not None
            pooled[best].predictions["confidence"] = model.fit_held_out(
                outputs.drop(columns="fold"), labels, outputs["fold"].to_numpy()
            )
            val = task.validation_metrics(examples, pooled[best].predictions)
            held_out_report = dict(getattr(model, "report", {}))
        checked = Checked(model, get_service(task.name).contract)
        sample = examples.rows(splits.sets[TRAIN][:LATENCY_ROWS])
        # The gate is about a freshly started server: models with process-wide caches (e.g.
        # embeddings) drop them first, so the timing includes first-sight costs
        if callable(reset := getattr(model, "reset_caches", None)):
            reset()
        latency = latency_ms(checked, sample)
        warm = latency_ms(checked, sample)["latency_p95_ms"]  # same rows again, caches warm
        metrics = {f"val_{k}": v for k, v in val.items()} | latency
        metrics["latency_warm_p95_ms"] = warm
        metrics |= {f"held_out.{k}": v for k, v in held_out_report.items()}
        if reproduce_poc:
            test = task.test_metrics(examples, splits, checked)
            metrics |= {f"test_{k}": v for k, v in test.items()}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / CONFIG_FILE).write_text(config.model_dump_json(indent=2), encoding="utf-8")
            tracker.log_artifacts(run_id, out, "")
            if pooled:
                (out / PREDICTIONS_PATH).mkdir()
                pooled[best].predictions.to_parquet(
                    out / PREDICTIONS_PATH / PREDICTIONS_FILE, index=False
                )
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


def run_with_twin(
    config: ExperimentConfig, data: Path, tracker: Tracker, *, force: bool = False
) -> tuple[RunResult, RunResult | None]:
    """The comparison run, then its shipping twin if the config declares one (FR-4 §1).

    The twin is tagged with the comparison run it belongs to. When the comparison run is new,
    its twin is retrained too, so a twin never points at a superseded comparison run.
    """
    result = run_experiment(config, data, tracker, force=force)
    if config.ship is None:
        return result, None
    twin = run_experiment(
        config.twin(),
        data,
        tracker,
        force=force or not result.skipped,
        extra_tags={TWIN_TAG: result.run_id},
    )
    return result, twin
