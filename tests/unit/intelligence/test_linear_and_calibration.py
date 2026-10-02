"""The linear text model and the calibration wrapper, with the stub embedder (no download)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.data.store import load_transactions
from smart_financial_coach.intelligence.categorization.calibration import (
    Calibrated,
    Calibrator,
    brier,
)
from smart_financial_coach.intelligence.categorization.contract import CONTRACT
from smart_financial_coach.intelligence.categorization.embeddings import EmbeddingError
from smart_financial_coach.intelligence.categorization.linear import LinearText
from smart_financial_coach.intelligence.models import Checked, build
from smart_financial_coach.intelligence.models.artifact import load_artifact, save_artifact

STUB = "stub:16"


@pytest.fixture(scope="module")
def data(small_sqlite: Path) -> tuple[pd.DataFrame, pd.Series]:
    import sqlite3

    x = load_transactions(small_sqlite)
    with sqlite3.connect(small_sqlite) as conn:
        y = pd.read_sql("SELECT category FROM truth_transactions ORDER BY transaction_id", conn)[
            "category"
        ]
    return x, y


def linear(**params: object) -> LinearText:
    model = build({"type": "categorization/linear_text", "params": {"embeddings": STUB, **params}})
    assert isinstance(model, LinearText)
    return model


def test_fits_and_meets_the_contract(data: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = data
    model = linear().fit(x.iloc[::2], y.iloc[::2])
    out = Checked(model, CONTRACT).predict(x.iloc[1::2])

    assert (out["category"].to_numpy() == y.iloc[1::2].to_numpy()).mean() > 0.95
    assert set(model.categories) == set(y)


@pytest.mark.parametrize(
    "params",
    [
        {"embeddings": None},
        {"ngrams": False},
        {"amount": False, "hour": False, "channel": False},
        {"ngram_min": 1, "ngram_max": 5},
    ],
)
def test_feature_blocks_switch(
    data: tuple[pd.DataFrame, pd.Series], params: dict[str, object]
) -> None:
    x, y = data
    model = linear(**params).fit(x.iloc[::3], y.iloc[::3])

    assert len(model.predict(x.iloc[:5])) == 5


def test_needs_some_text_features() -> None:
    with pytest.raises(ValueError, match="n-grams, embeddings or both"):
        linear(ngrams=False, embeddings=None)


def test_familiarity_is_recorded_before_the_cap(data: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = data
    model = linear(max_rows_per_class=5).fit(x, y)  # the cap keeps 5 rows per class
    _, familiar = model.scores(x)

    assert familiar.all()  # every training string is familiar, sampled out or not
    _, unfamiliar = model.scores(x.iloc[:1].assign(merchant_raw="ZORBLAX GADGETS"))
    assert not unfamiliar.any()


def test_fit_is_deterministic(data: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = data
    a = linear(max_rows_per_class=50).fit(x, y).predict(x.iloc[:200])
    b = linear(max_rows_per_class=50).fit(x, y).predict(x.iloc[:200])

    assert a.equals(b)


def test_artifact_checks_the_embedding_file(
    data: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    x, y = data
    model = linear().fit(x.iloc[::4], y.iloc[::4])
    model.version = "v1"
    save_artifact(model, tmp_path / "ok", {})
    assert load_artifact(tmp_path / "ok", trusted_root=tmp_path).predict(x.iloc[:3]) is not None

    model.embedding_file = {**model.embedding_file, "sha256": "f" * 64}
    save_artifact(model, tmp_path / "bad", {})
    with pytest.raises(EmbeddingError, match="trained with"):
        load_artifact(tmp_path / "bad", trusted_root=tmp_path)


def _overconfident(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two-class probabilities that claim 0.99 but are right 70% of the time."""
    rng = np.random.default_rng(seed)
    proba = np.tile([0.99, 0.01], (n, 1))
    truth = (rng.random(n) > 0.7).astype(np.int64)  # class 0 right 70% of the time
    return proba, truth


@pytest.mark.parametrize("method", ["temperature", "isotonic"])
def test_calibrators_fix_overconfidence(method: str) -> None:
    proba, truth = _overconfident(2000, 0)
    calibrated = Calibrator(method).fit(proba, truth).apply(proba)
    correct = truth == 0

    assert brier(calibrated, correct) < brier(proba[:, 0], correct)
    assert calibrated.mean() == pytest.approx(0.7, abs=0.03)


def test_calibrated_wrapper_never_changes_the_category(
    data: tuple[pd.DataFrame, pd.Series],
) -> None:
    x, y = data
    base = linear().fit(x.iloc[::2], y.iloc[::2])
    wrapper = Calibrated(base=linear(), method="isotonic")
    wrapper.fit(x.iloc[::2], y.iloc[::2])
    held = x.iloc[1::2].reset_index(drop=True)
    outputs = wrapper.held_out_outputs(held)
    folds = np.arange(len(held)) % 3

    oof = wrapper.fit_held_out(outputs, y.iloc[1::2].reset_index(drop=True), folds)
    out = Checked(wrapper, CONTRACT).predict(held)

    assert out["category"].equals(base.predict(held)["category"])
    assert len(oof) == len(held)
    assert set(wrapper.calibrators) <= {"familiar", "unfamiliar"}
    assert any(k.endswith("_brier") for k in wrapper.report)


def test_auto_picks_by_cross_validated_brier() -> None:
    proba, truth = _overconfident(3000, 1)
    outputs = pd.DataFrame(
        {
            "transaction_id": np.arange(3000).astype(str),
            "familiar": False,
            "p::a": proba[:, 0],
            "p::b": proba[:, 1],
        }
    )
    wrapper = Calibrated(base=linear(), method="auto")
    wrapper.categories = ("a", "b")
    labels = pd.Series(np.where(truth == 0, "a", "b"))

    wrapper.fit_held_out(outputs, labels, np.arange(3000) % 5)

    assert wrapper.calibrators["unfamiliar"].method in {"temperature", "isotonic"}
    assert (
        wrapper.report["unfamiliar.none_brier"]
        > wrapper.report[f"unfamiliar.{wrapper.calibrators['unfamiliar'].method}_brier"]
    )
    assert "familiar" not in wrapper.calibrators  # no familiar rows: left uncalibrated
