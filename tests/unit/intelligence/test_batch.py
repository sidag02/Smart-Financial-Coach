"""Batch categorization into a predictions file, with a trivial promoted model on the small data."""

from pathlib import Path

import pandas as pd
import pytest

from smart_financial_coach.data.predictions import (
    CategoryWriter,
    load_categories,
    load_prediction_meta,
)
from smart_financial_coach.data.store import iter_transactions, load_meta, load_transactions
from smart_financial_coach.evaluation.cli import model_main
from smart_financial_coach.intelligence.categorization.baseline import Majority
from smart_financial_coach.intelligence.categorization.batch import categorize_dataset
from smart_financial_coach.intelligence.categorization.review import (
    PolicyError,
    ReviewPolicy,
    save_review_policy,
)
from smart_financial_coach.intelligence.models.artifact import (
    record_promotion,
    save_artifact,
)


@pytest.fixture
def artifacts(small_sqlite: Path, tmp_path: Path) -> Path:
    """An artifacts folder whose promoted categorizer calls everything Dining."""
    transactions = load_transactions(small_sqlite)
    model = Majority().fit(transactions, pd.Series(["Dining"] * len(transactions)))
    root = tmp_path / "artifacts"
    save_artifact(model, root / "categorization" / model.version, {})
    save_review_policy(
        ReviewPolicy(model.version, 0.5, 0.8), root / "categorization" / model.version
    )
    record_promotion(root / "categorization", {"version": model.version})
    return root


def test_rows_are_flagged_by_the_promoted_models_review_policy(
    small_sqlite: Path, tmp_path: Path
) -> None:
    """A model fitted on half the rows, 75% confident everywhere: with thresholds 0.5 (familiar)
    and 0.8 (unfamiliar), exactly the rows at strings it never saw are flagged, as new."""
    transactions = load_transactions(small_sqlite)
    half = transactions.iloc[: len(transactions) // 2]
    labels = (["Dining"] * 3 + ["Shopping"]) * (len(half) // 4 + 1)
    model = Majority().fit(half, pd.Series(labels[: len(half)]))
    root = tmp_path / "artifacts" / "categorization"
    save_artifact(model, root / model.version, {})
    save_review_policy(ReviewPolicy(model.version, 0.5, 0.8), root / model.version)
    record_promotion(root, {"version": model.version})
    out = tmp_path / "predictions.sqlite"

    categorize_dataset(small_sqlite, out, artifacts_dir=tmp_path / "artifacts")

    cats = load_categories(out)
    assert cats["familiar"].any()
    assert not cats["familiar"].all()
    assert cats["needs_review"].equals(~cats["familiar"])
    assert set(cats.loc[cats["needs_review"], "review_reason"]) == {"new_merchant"}
    assert (cats.loc[~cats["needs_review"], "review_reason"] == "").all()
    meta = load_prediction_meta(out)
    assert (meta["review_familiar_below"], meta["review_unfamiliar_below"]) == ("0.5", "0.8")


def test_rows_predicted_income_are_never_flagged(small_sqlite: Path, tmp_path: Path) -> None:
    """The policy's thresholds were chosen on spending rows only (owner, on #32)."""
    transactions = load_transactions(small_sqlite)
    half = transactions.iloc[: len(transactions) // 2]
    labels = (["Income"] * 3 + ["Shopping"]) * (len(half) // 4 + 1)
    model = Majority().fit(half, pd.Series(labels[: len(half)]))  # Income everywhere, at 0.75
    root = tmp_path / "artifacts" / "categorization"
    save_artifact(model, root / model.version, {})
    save_review_policy(ReviewPolicy(model.version, 0.95, 0.95), root / model.version)
    record_promotion(root, {"version": model.version})
    out = tmp_path / "predictions.sqlite"

    categorize_dataset(small_sqlite, out, artifacts_dir=tmp_path / "artifacts")

    cats = load_categories(out)
    assert set(cats["category"]) == {"Income"}
    assert not cats["needs_review"].any()
    assert (cats["review_reason"] == "").all()


def test_a_model_without_a_review_policy_isnt_served(
    artifacts: Path, small_sqlite: Path, tmp_path: Path
) -> None:
    version = (artifacts / "categorization" / "PROMOTED").read_text().strip()
    (artifacts / "categorization" / version / "review_policy.json").unlink()

    with pytest.raises(PolicyError, match=r"has no review_policy\.json"):
        categorize_dataset(small_sqlite, tmp_path / "p.sqlite", artifacts_dir=artifacts)
    assert not (tmp_path / "p.sqlite").exists()


def test_every_transaction_is_categorized_once(
    small_sqlite: Path, artifacts: Path, tmp_path: Path
) -> None:
    out = tmp_path / "predictions.sqlite"

    run = categorize_dataset(small_sqlite, out, batch_rows=1_000, artifacts_dir=artifacts)

    transactions = load_transactions(small_sqlite)
    categories = load_categories(out)
    assert run.rows == len(transactions) > 1_000  # several batches
    assert run.rows_per_second > 0
    assert categories["transaction_id"].tolist() == transactions["transaction_id"].tolist()
    assert (categories["user_id"] == transactions["user_id"]).all()
    assert set(categories["category"]) == {"Dining"}
    assert set(categories["model_version"]) == {run.model_version}
    assert categories["familiar"].dtype == bool
    assert categories["familiar"].all()  # the model was fitted on these very strings
    meta = load_prediction_meta(out)
    assert meta["model_version"] == run.model_version
    assert meta["rows"] == str(run.rows)
    assert meta["data_spec_hash"] == load_meta(small_sqlite)["spec_hash"]


def test_reads_are_scoped_to_one_user(small_sqlite: Path, artifacts: Path, tmp_path: Path) -> None:
    out = tmp_path / "predictions.sqlite"
    categorize_dataset(small_sqlite, out, artifacts_dir=artifacts)
    user = load_transactions(small_sqlite)["user_id"].iloc[0]

    mine = load_categories(out, user_id=user)

    assert not mine.empty
    assert set(mine["user_id"]) == {user}


def test_existing_file_is_replaced_only_when_asked(
    small_sqlite: Path, artifacts: Path, tmp_path: Path
) -> None:
    out = tmp_path / "predictions.sqlite"
    categorize_dataset(small_sqlite, out, artifacts_dir=artifacts)

    with pytest.raises(FileExistsError):
        categorize_dataset(small_sqlite, out, artifacts_dir=artifacts)
    categorize_dataset(small_sqlite, out, artifacts_dir=artifacts, overwrite=True)


def test_a_failed_run_leaves_no_file(tmp_path: Path) -> None:
    out = tmp_path / "predictions.sqlite"
    batch = pd.DataFrame(
        {
            "transaction_id": ["t1"],
            "category": ["Dining"],
            "confidence": [0.9],
            "model_version": ["v1"],
            "familiar": [True],
            "needs_review": [False],
            "review_reason": [""],
        }
    )

    def run() -> None:
        with CategoryWriter(out, {}) as writer:
            writer.append(pd.Series(["u1"]), batch)
            raise RuntimeError("model failed mid-run")

    with pytest.raises(RuntimeError):
        run()

    assert list(tmp_path.iterdir()) == []


def test_overlapping_runs_keep_their_own_partial_files(tmp_path: Path) -> None:
    """A retried job overlapping the old one: neither deletes or shares the other's partial file."""
    out = tmp_path / "predictions.sqlite"

    def batch(version: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "transaction_id": ["t1"],
                "category": ["Dining"],
                "confidence": [0.9],
                "model_version": [version],
                "familiar": [True],
                "needs_review": [False],
                "review_reason": [""],
            }
        )

    with CategoryWriter(out, {}) as first:
        first.append(pd.Series(["u1"]), batch("first"))
        with CategoryWriter(out, {}) as second:
            second.append(pd.Series(["u1"]), batch("second"))
            assert first.partial != second.partial
        first.append(pd.Series(["u2"]), batch("first").assign(transaction_id="t2"))

    assert load_categories(out)["model_version"].tolist() == ["first", "first"]  # last to finish
    assert list(tmp_path.iterdir()) == [out]


def test_chunked_reads_match_one_read(small_sqlite: Path) -> None:
    chunks = list(iter_transactions(small_sqlite, chunk_rows=1_000))

    assert len(chunks) > 1
    assert all(len(c) <= 1_000 for c in chunks)
    pd.testing.assert_frame_equal(
        pd.concat(chunks, ignore_index=True), load_transactions(small_sqlite)
    )
    with pytest.raises(ValueError, match="positive"):
        next(iter_transactions(small_sqlite, chunk_rows=0))


def test_cli_predict(
    small_sqlite: Path, artifacts: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "predictions.sqlite"
    args = ["--data", str(small_sqlite), "--out", str(out), "--artifacts-dir", str(artifacts)]

    assert model_main(["predict", "--task", "categorization", *args]) == 0
    assert "rows per second" in capsys.readouterr().out
    assert model_main(["predict", "--task", "categorization", *args]) == 1  # exists
    assert model_main(["predict", "--task", "toy", *args, "--overwrite"]) == 1
    assert "supports categorization and unusual_transactions" in capsys.readouterr().err
