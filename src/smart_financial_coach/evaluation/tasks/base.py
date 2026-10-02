"""A task owns everything problem-specific, so the runner, tracking and promotion stay generic.

A task loads examples and attaches labels (it may read truth; models never do), builds and checks
splits, prepares training rows (label noise, caps), and scores predictions. Metric names come back
without a prefix; the runner logs them as `val_*` or `test_*`.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from smart_financial_coach.evaluation.splits import Ids, Splits
from smart_financial_coach.intelligence.models.contract import Checked


@dataclass(frozen=True)
class Examples:
    frame: pd.DataFrame  # model-visible columns plus evaluation-only columns (groups, strata)
    labels: pd.Series | None  # aligned with `frame`; None for unsupervised tasks
    id_column: str
    model_columns: tuple[str, ...]  # what models see, including the ID column
    data_hash: str  # identifies the dataset (e.g. its spec hash and schema version)

    def rows(self, which: Ids) -> pd.DataFrame:
        """Model-visible rows for `which`, in that order."""
        frame = self.frame.set_index(self.id_column, drop=False)[list(self.model_columns)]
        rows: pd.DataFrame = frame.loc[which.tolist()]
        return rows.reset_index(drop=True)

    def labels_for(self, which: Ids) -> pd.Series | None:
        if self.labels is None:
            return None
        labels = pd.Series(self.labels.to_numpy(), index=self.frame[self.id_column].to_numpy())
        picked: pd.Series = labels.loc[which.tolist()]
        return picked.reset_index(drop=True)


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    detail: str


class Task(Protocol):
    name: str  # the service it trains, e.g. "categorization"
    selection_metric: str  # validation metric the decision rule ranks by; higher is better
    tuning_metric: str  # validation metric that picks a run's grid point; higher is better
    # Logged metrics that break ties, in order; lower is better. `complexity` is always last.
    tiebreak_metrics: tuple[str, ...]
    required_baselines: tuple[str, ...]  # baseline run names eligibility and gates compare to

    def load(self, data: Path) -> Examples: ...

    def split(self, examples: Examples, params: Mapping[str, Any], seed: int) -> Splits: ...

    def leak_errors(self, examples: Examples, splits: Splits) -> list[str]: ...

    def training_rows(
        self, examples: Examples, which: Ids, params: Mapping[str, Any], seed: int
    ) -> tuple[pd.DataFrame, pd.Series | None]: ...

    def validation_metrics(self, examples: Examples, pooled: pd.DataFrame) -> dict[str, float]:
        """Scores pooled out-of-fold predictions (columns: model output + `fold`, `held_out`)."""
        ...

    def test_metrics(
        self, examples: Examples, splits: Splits, model: Checked
    ) -> dict[str, float]: ...

    def eligible(
        self, metrics: Mapping[str, float], baselines: Mapping[str, Mapping[str, float]]
    ) -> bool:
        """Whether a run may be ranked, from its logged metrics and the baselines' (validation)."""
        ...

    def tied(self, examples: Examples, leader: pd.DataFrame, other: pd.DataFrame) -> bool:
        """Whether two runs' pooled validation predictions are statistically tied."""
        ...

    def gates(
        self, metrics: Mapping[str, float], baselines: Mapping[str, Mapping[str, float]]
    ) -> list[Gate]:
        """Promotion gates on a finalist's `test_*` and latency metrics, against baselines."""
        ...

    def reproduction_configs(self) -> set[str]:
        """Config hashes allowed to score test sets outside `finalize` (`--reproduce-poc`)."""
        ...


_TASKS: dict[str, Callable[[], Task]] = {}
# Modules whose import registers the product's tasks; tests register their own
TASK_MODULES: tuple[str, ...] = ("smart_financial_coach.evaluation.tasks.categorization",)


def register_task(name: str, factory: Callable[[], Task]) -> None:
    _TASKS[name] = factory


def get_task(name: str) -> Task:
    import importlib

    for module in TASK_MODULES:
        importlib.import_module(module)
    if name not in _TASKS:
        raise ValueError(f"unknown task {name!r}; registered: {', '.join(sorted(_TASKS))}")
    return _TASKS[name]()


def latency_ms(model: Checked, x: pd.DataFrame, *, rows: int = 200) -> dict[str, float]:
    """p50 and p95 wall time of one-row calls on distinct rows, and of the whole frame once.

    Distinct rows, not one row repeated: a repeated row measures only warm caches.
    """
    import time

    times = []
    for i in range(min(rows, len(x))):
        started = time.perf_counter()
        model.predict(x.iloc[i : i + 1])
        times.append((time.perf_counter() - started) * 1000)
    started = time.perf_counter()
    model.predict(x)
    batch = (time.perf_counter() - started) * 1000
    return {
        "latency_p50_ms": float(np.percentile(times, 50)),
        "latency_p95_ms": float(np.percentile(times, 95)),
        "latency_batch_ms": batch,
        "latency_batch_rows": float(len(x)),
    }
