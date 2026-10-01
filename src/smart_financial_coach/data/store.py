"""Model-visible data access: the only way services read a dataset.

Reads `users`, `transactions`, `goals` and `meta` from a generated SQLite file, read-only, and never
a ground-truth table. Every reader can be scoped to one user, which is what the tool server will
build on (Technical Design build order step 6).

    txns = load_transactions("data/synthetic/default.sqlite", user_id="u_te_yp_0007")
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from smart_financial_coach.data.generator.dataset import MODEL_TABLES, TABLES


@contextmanager
def _connect(path: str | Path) -> Iterator[sqlite3.Connection]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        yield conn
    finally:
        conn.close()


def _read(path: str | Path, table: str, user_id: str | None) -> pd.DataFrame:
    if table not in MODEL_TABLES:
        raise ValueError(f"{table} is not a model-visible table")
    t = TABLES[table]
    where, params = ("WHERE user_id = ?", (user_id,)) if user_id is not None else ("", ())
    columns, order = ", ".join(t.column_names), ", ".join(t.primary_key)
    sql = f"SELECT {columns} FROM {table} {where} ORDER BY {order}"
    with _connect(path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def load_transactions(path: str | Path, user_id: str | None = None) -> pd.DataFrame:
    return _read(path, "transactions", user_id)


def load_users(path: str | Path, user_id: str | None = None) -> pd.DataFrame:
    return _read(path, "users", user_id)


def load_goals(path: str | Path, user_id: str | None = None) -> pd.DataFrame:
    return _read(path, "goals", user_id)


def load_meta(path: str | Path) -> dict[str, str]:
    with _connect(path) as conn:
        return dict(conn.execute("SELECT key, value FROM meta ORDER BY key").fetchall())
