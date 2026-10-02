"""A toy task, service and models that exercise the framework without the real categorizer.

Examples are rows of 30 groups (think merchants) with one label each. Groups g24-g29 appear only
for test users (holdout), and g00-g02 are protected: never held out in validation. A `hint`
column carries the label 80% of the time, so a model can generalize to unseen groups.
"""

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.evaluation.experiment import ExperimentConfig
from smart_financial_coach.evaluation.splits import (
    TRAIN,
    UNSEEN,
    Ids,
    Splits,
    group_kfold,
    ids,
    leak_errors,
    stratified_sample,
)
from smart_financial_coach.evaluation.tasks.base import Examples, Gate, register_task
from smart_financial_coach.evaluation.tracking import Tracker
from smart_financial_coach.intelligence.models import BaseModel, Checked, Contract, register
from smart_financial_coach.intelligence.service import register_service

LABELS = ("a", "b", "c")
PROTECTED = {"g00", "g01", "g02"}
HOLDOUT = {f"g{g:02d}" for g in range(24, 30)}


def _confidence_check(out: pd.DataFrame) -> list[str]:
    bad = ~out["confidence"].between(0, 1)
    return [f"{int(bad.sum())} confidences outside [0, 1]"] if bad.any() else []


TOY = register_service(Contract("toy", "id", ("id", "label", "confidence"), _confidence_check))


@register("toy/majority")
class Majority(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.label = ""

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        assert y is not None
        self.label = str(y.value_counts().sort_index().idxmax())
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"id": x["id"], "label": self.label, "confidence": 0.5})


@register("toy/memory")
class Memory(BaseModel):
    """Remembers each group's label; falls back to the hint (or the majority) for new groups."""

    def __init__(self, trust_hint: bool = True, confidence: float = 0.9) -> None:
        super().__init__(trust_hint=trust_hint, confidence=confidence)
        self.trust_hint = trust_hint
        self.confidence = confidence
        self.groups: dict[str, str] = {}
        self.majority = ""

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        assert y is not None
        frame = x.assign(label=y.to_numpy())
        modes = frame.groupby("group")["label"].agg(lambda s: s.mode().iloc[0])
        self.groups = {str(k): str(v) for k, v in modes.items()}
        self.majority = str(y.value_counts().sort_index().idxmax())
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        fallback = x["hint"] if self.trust_hint else pd.Series(self.majority, index=x.index)
        label = x["group"].map(self.groups).fillna(fallback)
        return pd.DataFrame({"id": x["id"], "label": label, "confidence": self.confidence})


@register("toy/memory_alt")
class MemoryAlt(Memory):
    """Another model type with the same behavior and a different confidence."""

    def __init__(self) -> None:
        super().__init__(confidence=0.75)


@register("toy/broken")
class Broken(Memory):
    """Drops the last row: breaks the contract."""

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return super().predict(x).iloc[:-1]


def accuracy(predictions: pd.DataFrame, examples: Examples) -> float:
    """Row by row: pooled predictions can repeat an ID (seen in more than one fold)."""
    truth = examples.frame.set_index("id")["label"]
    return float(
        (predictions["label"].to_numpy() == truth.loc[predictions["id"]].to_numpy()).mean()
    )


class ToyTask:
    name = "toy"
    selection_metric = "unseen_accuracy"
    tuning_metric = "unseen_accuracy"
    tiebreak_metrics: tuple[str, ...] = ("val_seen_accuracy_error",)
    required_baselines: tuple[str, ...] = ("majority",)
    reproduction = ExperimentConfig(name="poc", task="toy", model={"type": "toy/memory"})

    def load(self, data: Path) -> Examples:
        frame = pd.read_csv(data, dtype={"id": str, "group": str, "hint": str, "label": str})
        return Examples(
            frame=frame,
            labels=frame["label"],
            id_column="id",
            model_columns=("id", "group", "hint"),
            data_hash=hashlib.sha256(data.read_bytes()).hexdigest(),
        )

    def split(self, examples: Examples, params: Mapping[str, Any], seed: int) -> Splits:
        f = examples.frame
        train_users = f[f["split"] == "train"]
        test_known = stratified_sample(train_users, frac=0.2, by="label", seed=seed, id_column="id")
        train = train_users[~train_users["id"].isin(test_known)]
        test_all = f[f["split"] == "test"]
        folds = group_kfold(
            train,
            group="group",
            k=int(params.get("k", 3)),
            seed=seed,
            id_column="id",
            stratify="label",
            eligible=set(train["group"]) - PROTECTED,
            seen_frac=0.1,
        )
        return Splits(
            sets={
                TRAIN: ids(train["id"]),
                "test_known": test_known,
                "test_all": ids(test_all["id"]),
                "test_unseen": ids(test_all.loc[test_all["group"].isin(HOLDOUT), "id"]),
            },
            folds=folds,
        )

    def leak_errors(self, examples: Examples, splits: Splits) -> list[str]:
        return leak_errors(
            splits, examples.frame, id_column="id", group="group", never_held_out=PROTECTED
        )

    def training_rows(
        self, examples: Examples, which: Ids, params: Mapping[str, Any], seed: int
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        return examples.rows(which), examples.labels_for(which)

    def validation_metrics(self, examples: Examples, pooled: pd.DataFrame) -> dict[str, float]:
        seen = accuracy(pooled[pooled["held_out"] == "seen"], examples)
        return {
            "unseen_accuracy": accuracy(pooled[pooled["held_out"] == UNSEEN], examples),
            "seen_accuracy": seen,
            "seen_accuracy_error": 1 - seen,
        }

    def test_metrics(self, examples: Examples, splits: Splits, model: Checked) -> dict[str, float]:
        return {
            f"{name.removeprefix('test_')}_accuracy": accuracy(
                model.predict(examples.rows(splits.sets[name])), examples
            )
            for name in ("test_known", "test_all", "test_unseen")
        }

    def eligible(
        self, metrics: Mapping[str, float], baselines: Mapping[str, Mapping[str, float]]
    ) -> bool:
        floor = max((b.get("val_seen_accuracy", 0.0) for b in baselines.values()), default=0.0)
        return metrics.get("val_seen_accuracy", 0.0) > floor

    def tied(self, examples: Examples, leader: pd.DataFrame, other: pd.DataFrame) -> bool:
        def unseen(p: pd.DataFrame) -> float:
            return accuracy(p[p["held_out"] == UNSEEN], examples)

        return abs(unseen(leader) - unseen(other)) < 0.02

    def gates(
        self, metrics: Mapping[str, float], baselines: Mapping[str, Mapping[str, float]]
    ) -> list[Gate]:
        known = metrics.get("test_known_accuracy", 0.0)
        unseen = metrics.get("test_unseen_accuracy", 0.0)
        floor = max(b.get("test_known_accuracy", 0.0) for b in baselines.values())
        return [
            Gate("known_accuracy", known >= 0.9, f"{known:.3f} vs 0.9"),
            Gate("unseen_accuracy", unseen >= 0.5, f"{unseen:.3f} vs 0.5"),
            Gate("beats_baseline", known > floor, f"{known:.3f} vs {floor:.3f}"),
        ]

    def reproduction_configs(self) -> set[str]:
        return {self.reproduction.config_hash()}


register_task("toy", ToyTask)


def toy_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(30):
        group = f"g{g:02d}"
        label = LABELS[g % 3]
        for i in range(20):
            test_user = group in HOLDOUT or i >= 16
            hint = label if rng.random() < 0.8 else LABELS[int(rng.integers(3))]
            rows.append(
                {
                    "id": f"{group}-{i:02d}",
                    "group": group,
                    "hint": hint,
                    "label": label,
                    "split": "test" if test_user else "train",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def toy_data(tmp_path: Path) -> Path:
    path = tmp_path / "toy.csv"
    toy_frame().to_csv(path, index=False)
    return path


@pytest.fixture
def tracker(tmp_path: Path) -> Tracker:
    return Tracker(f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=tmp_path / "mlartifacts")


class MakeConfig(Protocol):
    def __call__(self, name: str, model_type: str = ..., **kwargs: Any) -> ExperimentConfig: ...


@pytest.fixture
def make_config() -> MakeConfig:
    def make(name: str, model_type: str = "toy/memory", **kwargs: Any) -> ExperimentConfig:
        return ExperimentConfig.model_validate(
            {"name": name, "task": "toy", "model": {"type": model_type}, **kwargs}
        )

    return make
