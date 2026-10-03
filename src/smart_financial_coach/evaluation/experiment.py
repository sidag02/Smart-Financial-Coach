"""Experiment configs: one YAML file is one experiment.

name: linear_both
task: categorization
model: {type: categorization/linear_text, params: {ngrams: true, embeddings: true}}
grid: {C: [1.0, 3.0, 10.0]}  # chosen on validation folds; merged into model params
task_params: {label_noise: 0.02, max_rows_per_class: 20000}
ship: {task_params: {label_noise: 0}}  # the shipping twin: what is finalized and promoted
complexity: 2  # components and dependencies; lower is simpler (last tie-breaker)

Comparison runs are ranked; their shipping twins (the same config with `ship` applied, FR-4 §1)
are what gets finalized and promoted, when the task requires twins (`Task.shipping_params`).
"""

import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from smart_financial_coach.intelligence.models.registry import MODEL_MARKER


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSpec(_Model):
    type: str
    params: dict[str, Any] = {}


class ShipSpec(_Model):
    task_params: dict[str, Any] = {}


TWIN_SUFFIX = ".ship"


class ExperimentConfig(_Model):
    name: str
    task: str
    model: ModelSpec
    grid: dict[str, list[Any]] = {}
    seed: int = 0
    task_params: dict[str, Any] = {}
    baseline: bool = False  # round 0: always scored with the finalists
    complexity: int = Field(default=0, ge=0)
    tags: dict[str, str] = {}
    ship: ShipSpec | None = None  # the shipping twin's overrides; None: no twin

    @model_validator(mode="after")
    def _baselines_have_no_twin(self) -> "ExperimentConfig":
        if self.baseline and self.ship is not None:
            raise ValueError(f"{self.name}: baselines are scored as they are; they have no twin")
        return self

    def twin(self) -> "ExperimentConfig":
        """The shipping twin: this config with `ship`'s task params applied, written explicitly."""
        if self.ship is None:
            raise ValueError(f"{self.name} declares no shipping twin (`ship:`)")
        return self.model_copy(
            update={
                "name": self.name + TWIN_SUFFIX,
                "task_params": {**self.task_params, **self.ship.task_params},
                "ship": None,
            }
        )

    def config_hash(self) -> str:
        """Everything that changes the result; `tags` are annotations and don't count."""
        payload = self.model_dump(mode="json", exclude={"tags"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def grid_points(self) -> list[dict[str, Any]]:
        keys = sorted(self.grid)
        return [
            dict(zip(keys, values, strict=True))
            for values in itertools.product(*(self.grid[k] for k in keys))
        ]

    def model_spec(self, point: dict[str, Any]) -> dict[str, Any]:
        """The model spec with a grid point applied. Dotted keys reach into nested models:
        `base.C` sets `C` on the model in the `base` param (`{"$model": ...}`)."""
        spec: dict[str, Any] = {"type": self.model.type, "params": copy.deepcopy(self.model.params)}
        for key, value in point.items():
            *path, name = key.split(".")
            target: dict[str, Any] = spec
            for step in path:
                nested = target["params"].get(step)
                if not (isinstance(nested, dict) and MODEL_MARKER in nested):
                    raise ValueError(f"grid key {key!r}: {step!r} is not a nested model")
                target = nested[MODEL_MARKER]
                target.setdefault("params", {})
            target["params"][name] = value
        return spec


def load_experiment(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(raw)


def experiment_files(paths: list[Path]) -> list[Path]:
    """Config files from files and folders, folders expanded in name order."""
    out: list[Path] = []
    for path in paths:
        out += sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    return out
