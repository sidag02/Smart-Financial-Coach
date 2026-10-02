"""Experiment tracking in MLflow. This is the only module that imports `mlflow`.

One MLflow experiment per task; one run per experiment config, with a child run per grid point.
By default the store is local (`mlruns/mlflow.db`, artifacts under `mlruns/artifacts/`, both
gitignored); `SFC_MLFLOW_TRACKING_URI` points it at a shared server instead.

    uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
"""

import os
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

from smart_financial_coach.config import PROJECT_ROOT, get_settings

LOCAL_STORE = PROJECT_ROOT / "mlruns"
INVALID_KEY = re.compile(r"[^\w.\-: /]")  # characters MLflow rejects in param names
KIND_TAG = "sfc.kind"  # "experiment" on runs the leaderboard and finalize consider
PARENT_TAG = "mlflow.parentRunId"
CHAMPION = "champion"
MODEL_PATH = "model"  # artifact folder holding the exported model


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    name: str
    status: str
    artifact_uri: str
    tags: dict[str, str]
    metrics: dict[str, float]
    params: dict[str, str]


def flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Nested config as dotted MLflow params: {"model": {"type": "x"}} -> {"model.type": "x"}."""
    out: dict[str, str] = {}
    for key, item in value.items():
        name = prefix + INVALID_KEY.sub("", str(key))  # "$model" -> "model"
        if isinstance(item, Mapping) and item:
            out |= flatten(item, f"{name}.")
        else:
            out[name] = str(item)
    return out


class Tracker:
    def __init__(self, uri: str | None = None, artifact_root: Path | None = None) -> None:
        configured = uri or get_settings().mlflow_tracking_uri
        if configured is None:
            LOCAL_STORE.mkdir(exist_ok=True)
            configured = f"sqlite:///{LOCAL_STORE / 'mlflow.db'}"
            artifact_root = artifact_root or LOCAL_STORE / "artifacts"
        self.uri = configured
        self.artifact_root = artifact_root
        self.client = MlflowClient(tracking_uri=configured, registry_uri=configured)

    # --- Runs ------------------------------------------------------------------------------

    def experiment_id(self, task: str) -> str:
        if (experiment := self.client.get_experiment_by_name(task)) is not None:
            return str(experiment.experiment_id)
        location = (self.artifact_root / task).as_uri() if self.artifact_root else None
        return str(self.client.create_experiment(task, artifact_location=location))

    @contextmanager
    def run(
        self, task: str, name: str, tags: Mapping[str, str], parent: str | None = None
    ) -> Iterator[str]:
        """A run that ends FINISHED, or FAILED if the block raises."""
        all_tags = {PARENT_TAG: parent, **tags} if parent else {KIND_TAG: "experiment", **tags}
        run = self.client.create_run(self.experiment_id(task), run_name=name, tags=all_tags)
        run_id = str(run.info.run_id)
        try:
            yield run_id
        except BaseException:
            self.client.set_terminated(run_id, "FAILED")
            raise
        self.client.set_terminated(run_id, "FINISHED")

    def log(
        self,
        run_id: str,
        *,
        params: Mapping[str, str] | None = None,
        metrics: Mapping[str, float] | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        now = int(time.time() * 1000)
        self.client.log_batch(
            run_id,
            metrics=[Metric(k, float(v), now, 0) for k, v in (metrics or {}).items()],
            params=[Param(k, v) for k, v in (params or {}).items()],
            tags=[RunTag(k, v) for k, v in (tags or {}).items()],
        )

    def log_artifacts(self, run_id: str, local_dir: Path, path: str) -> None:
        self.client.log_artifacts(run_id, str(local_dir), path or None)

    def download(self, run_id: str, path: str, dst: Path) -> Path:
        dst.mkdir(parents=True, exist_ok=True)
        return Path(self.client.download_artifacts(run_id, path, str(dst)))

    def get(self, run_id: str) -> RunRecord:
        run = self.client.get_run(run_id)
        return RunRecord(
            run_id=str(run.info.run_id),
            name=str(run.info.run_name),
            status=str(run.info.status),
            artifact_uri=str(run.info.artifact_uri),
            tags=dict(run.data.tags),
            metrics=dict(run.data.metrics),
            params=dict(run.data.params),
        )

    def find(self, task: str, tags: Mapping[str, str]) -> list[RunRecord]:
        """Finished runs whose tags all match: experiment runs unless `tags` sets the kind.

        Grid-point children carry no kind tag, so they never match.
        """
        if self.client.get_experiment_by_name(task) is None:
            return []
        tags = {KIND_TAG: "experiment", **tags}
        clauses = ["attributes.status = 'FINISHED'"]
        clauses += [f"tags.`{k}` = '{v}'" for k, v in tags.items()]
        runs = self.client.search_runs(
            [self.experiment_id(task)],
            filter_string=" and ".join(clauses),
            order_by=["attributes.start_time ASC"],
            max_results=5000,
        )
        return [self.get(str(r.info.run_id)) for r in runs]

    # --- Registry --------------------------------------------------------------------------

    def register_champion(self, task: str, run: RunRecord, version: str) -> str:
        """Register the run's model under the task's name and point `champion` at it."""
        if not self.client.search_registered_models(filter_string=f"name = '{task}'"):
            self.client.create_registered_model(task)
        registered = self.client.create_model_version(
            task,
            source=f"{run.artifact_uri}/{MODEL_PATH}",
            run_id=run.run_id,
            tags={"sfc.version": version},
        )
        self.client.set_registered_model_alias(task, CHAMPION, registered.version)
        return str(registered.version)

    def champion_run(self, task: str) -> str | None:
        try:
            return str(self.client.get_model_version_by_alias(task, CHAMPION).run_id)
        except Exception:  # no registered model or no alias yet
            return None
