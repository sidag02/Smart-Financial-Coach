"""The data-access layer reads model-visible tables only, optionally for one user."""

import sqlite3
from pathlib import Path

import pytest

from smart_financial_coach.data import store
from smart_financial_coach.data.generator.dataset import TABLES


def test_reads_model_tables_scoped_to_a_user(small_sqlite: Path) -> None:
    everyone = store.load_transactions(small_sqlite)
    user = everyone["user_id"].iloc[0]
    mine = store.load_transactions(small_sqlite, user_id=user)

    assert list(everyone.columns) == TABLES["transactions"].column_names
    assert set(mine["user_id"]) == {user}
    assert len(mine) == (everyone["user_id"] == user).sum()
    assert mine["transaction_id"].is_monotonic_increasing
    assert set(store.load_users(small_sqlite, user_id=user)["user_id"]) == {user}
    assert store.load_meta(small_sqlite)["schema_version"] == "4"


def test_refuses_truth_tables_and_missing_files(small_sqlite: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a model-visible table"):
        store._read(small_sqlite, "truth_transactions", None)
    with pytest.raises(FileNotFoundError):
        store.load_transactions(tmp_path / "missing.sqlite")


def test_opens_read_only(small_sqlite: Path) -> None:
    with store._connect(small_sqlite) as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM transactions")
