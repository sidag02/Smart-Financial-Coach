"""Registry, contract checks and artifacts, on models defined here."""

import json
from pathlib import Path
from typing import Self

import pandas as pd
import pytest

from smart_financial_coach.intelligence.models import (
    BaseModel,
    Checked,
    Contract,
    ContractError,
    build,
    register,
)
from smart_financial_coach.intelligence.models.artifact import (
    LOG_FILE,
    MODEL_FILE,
    POINTER_FILE,
    ArtifactError,
    load_artifact,
    promotion_errors,
    read_manifest,
    record_promotion,
    save_artifact,
)


@register("test/constant")
class Constant(BaseModel):
    def __init__(self, label: str = "x", confidence: float = 1.0) -> None:
        super().__init__(label=label, confidence=confidence)
        self.label = label
        self.confidence = confidence

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"id": x["id"], "label": self.label, "confidence": self.confidence})


@register("test/pair")
class Pair(BaseModel):
    """A wrapper: built from nested specs."""

    def __init__(self, first: BaseModel, second: BaseModel) -> None:
        super().__init__(first=first, second=second)
        self.first = first

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return self.first.predict(x)


def _range(out: pd.DataFrame) -> list[str]:
    return [] if out["confidence"].between(0, 1).all() else ["confidence outside [0, 1]"]


CONTRACT = Contract("test", "id", ("id", "label", "confidence"), _range)
X = pd.DataFrame({"id": ["1", "2", "3"]})


def test_build_from_spec_with_nested_models() -> None:
    model = build(
        {
            "type": "test/pair",
            "params": {
                "first": {"type": "test/constant", "params": {"label": "y"}},
                "second": {"type": "test/constant"},
            },
        }
    )

    assert isinstance(model, Pair)
    assert model.predict(X)["label"].tolist() == ["y", "y", "y"]


def test_unknown_type_and_duplicate_registration() -> None:
    with pytest.raises(ValueError, match="unknown model type"):
        build({"type": "test/missing"})
    with pytest.raises(ValueError, match="already registered"):
        register("test/constant")(Pair)


def test_contract_passes_a_good_model() -> None:
    assert Checked(Constant(), CONTRACT).predict(X)["id"].tolist() == ["1", "2", "3"]


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        (lambda out: out.iloc[:-1], "2 rows for 3 inputs"),
        (lambda out: out.iloc[::-1], "doesn't match the input rows in order"),
        (lambda out: out.drop(columns="confidence"), "columns"),
        (lambda out: out.assign(label=None), "missing values"),
        (lambda out: out.assign(confidence=1.5), "outside [0, 1]"),
    ],
)
def test_contract_catches_broken_output(broken: object, message: str) -> None:
    class Broken(Constant):
        def predict(self, x: pd.DataFrame) -> pd.DataFrame:
            return broken(super().predict(x))  # type: ignore[operator, no-any-return]

    with pytest.raises(ContractError, match=message.replace("[", r"\[").replace("]", r"\]")):
        Checked(Broken(), CONTRACT).predict(X)


def test_artifact_round_trip(tmp_path: Path) -> None:
    model = Constant(label="z")
    model.version = "v1"
    folder = save_artifact(model, tmp_path / "v1", {"task": "test"})

    loaded = load_artifact(folder, trusted_root=tmp_path)

    assert loaded.predict(X)["label"].tolist() == ["z", "z", "z"]
    assert read_manifest(folder)["params"] == {"label": "z", "confidence": 1.0}


def test_artifact_refuses_tampering_and_untrusted_folders(tmp_path: Path) -> None:
    model = Constant()
    model.version = "v1"
    folder = save_artifact(model, tmp_path / "store" / "v1", {})

    with pytest.raises(ArtifactError, match="not under"):
        load_artifact(folder, trusted_root=tmp_path / "elsewhere")
    (folder / MODEL_FILE).write_bytes((folder / MODEL_FILE).read_bytes() + b"\0")
    with pytest.raises(ArtifactError, match="checksum"):
        load_artifact(folder, trusted_root=tmp_path)


def test_promotion_pointer_follows_the_log(tmp_path: Path) -> None:
    for version in ("v1", "v2"):
        model = Constant()
        model.version = version
        save_artifact(model, tmp_path / version, {})
        record_promotion(tmp_path, {"version": version, "note": "n"})

    assert (tmp_path / POINTER_FILE).read_text().strip() == "v2"
    assert promotion_errors(tmp_path) == []

    (tmp_path / POINTER_FILE).write_text("v1\n")
    assert promotion_errors(tmp_path) == [
        f"{tmp_path}: PROMOTED is v1, not the last logged promotion"
    ]
    with pytest.raises(ArtifactError, match="not an exported artifact"):
        record_promotion(tmp_path, {"version": "v3"})
    assert len((tmp_path / LOG_FILE).read_text().splitlines()) == 2
    assert json.loads((tmp_path / LOG_FILE).read_text().splitlines()[0])["version"] == "v1"
