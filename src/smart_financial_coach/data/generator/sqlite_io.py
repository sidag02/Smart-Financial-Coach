"""Write a Dataset to one SQLite file, and read it back."""

import math
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from smart_financial_coach.data.generator.dataset import TABLES, Dataset, Table

INDEXES = (
    "CREATE INDEX idx_transactions_user_ts ON transactions (user_id, ts)",
    "CREATE INDEX idx_goals_user ON goals (user_id)",
    "CREATE INDEX idx_truth_periods_user ON truth_periods (user_id)",
)


def _ddl(table: Table) -> str:
    cols = [f"{c.name} {c.type}{'' if c.nullable else ' NOT NULL'}" for c in table.columns]
    cols.append(f"PRIMARY KEY ({', '.join(table.primary_key)})")
    return f"CREATE TABLE {table.name} ({', '.join(cols)}) WITHOUT ROWID"


def _python_value(value: Any, sqltype: str) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if sqltype == "INTEGER":
        return int(value)
    if sqltype == "REAL":
        return float(value)
    return str(value)


def _rows(df: pd.DataFrame, table: Table) -> Iterator[tuple[Any, ...]]:
    df = df.sort_values(list(table.primary_key), kind="mergesort")
    columns = [[_python_value(v, c.type) for v in df[c.name].tolist()] for c in table.columns]
    return zip(*columns, strict=True)


def write_sqlite(dataset: Dataset, path: Path, *, overwrite: bool = False) -> Path:
    """Write atomically: build a temporary file next to `path`, then move it into place."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    written = False
    try:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        for table in TABLES.values():
            conn.execute(_ddl(table))
            placeholders = ", ".join("?" * len(table.columns))
            conn.executemany(
                f"INSERT INTO {table.name} VALUES ({placeholders})",
                _rows(dataset.tables[table.name], table),
            )
        conn.execute("CREATE TABLE meta (key TEXT NOT NULL PRIMARY KEY, value TEXT NOT NULL)")
        conn.executemany("INSERT INTO meta VALUES (?, ?)", sorted(dataset.meta.items()))
        for statement in INDEXES:
            conn.execute(statement)
        conn.commit()
        written = True
    finally:
        conn.close()
        if not written:
            tmp.unlink(missing_ok=True)
    tmp.replace(path)
    return path


def read_sqlite(path: str | Path) -> Dataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {
            name: pd.read_sql_query(
                f"SELECT {', '.join(t.column_names)} FROM {name}"
                f" ORDER BY {', '.join(t.primary_key)}",
                conn,
            )
            for name, t in TABLES.items()
        }
        meta = dict(conn.execute("SELECT key, value FROM meta ORDER BY key").fetchall())
    finally:
        conn.close()
    return Dataset(tables=tables, meta=meta)
