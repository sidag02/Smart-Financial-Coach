"""Model outputs, kept in their own SQLite file rather than the generated dataset (FR-3 option C-b).

The generator's file stays a pure function of its spec; a predictions file holds one model
version's output for one dataset, and every row carries that version.

    transaction_categories  transaction_id (PK), user_id (indexed), category, confidence,
                            model_version, familiar (0/1: the string occurs in the model's
                            training rows), needs_review (0/1) and review_reason
                            ("new_merchant", "low_confidence" or ""), from the model's review
                            policy (FR-5 §1)
    meta                    model_version, data_spec_hash, data_spec_name, created_at, rows,
                            review_familiar_below, review_unfamiliar_below

A run writes to its own temporary file next to the target and renames it into place when complete,
so readers never see half a run, and overlapping runs (a retried job, two workers) don't share or
delete each other's partial files.

    with CategoryWriter("data/predictions/default.sqlite", meta) as writer:
        writer.append(user_ids, categorizer.categorize(batch))
    load_categories("data/predictions/default.sqlite", user_id="u_te_yp_0007")
"""

import sqlite3
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Self

import pandas as pd

CATEGORIES_TABLE = "transaction_categories"
COLUMNS = (
    "transaction_id",
    "user_id",
    "category",
    "confidence",
    "model_version",
    "familiar",
    "needs_review",
    "review_reason",
)
FLAGS = ("familiar", "needs_review")  # stored as 0/1, read back as bool
_SCHEMA = f"""
CREATE TABLE {CATEGORIES_TABLE} (
    transaction_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    category TEXT NOT NULL,
    confidence REAL NOT NULL,
    model_version TEXT NOT NULL,
    familiar INTEGER NOT NULL CHECK (familiar IN (0, 1)),
    needs_review INTEGER NOT NULL CHECK (needs_review IN (0, 1)),
    review_reason TEXT NOT NULL CHECK (review_reason IN ('', 'new_merchant', 'low_confidence'))
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
# Built after the rows are in: one index build is faster than maintaining it per insert
_INDEX = f"CREATE INDEX idx_{CATEGORIES_TABLE}_user ON {CATEGORIES_TABLE} (user_id)"


class CategoryWriter:
    """Writes one categorization run; the file appears at `path` only if the run completes."""

    def __init__(
        self, path: str | Path, meta: Mapping[str, str], *, overwrite: bool = False
    ) -> None:
        self.path = Path(path)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"{self.path} exists; pass overwrite to replace it")
        self.meta = dict(meta)
        self.rows = 0

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=self.path.parent, prefix=f"{self.path.name}.", suffix=".part", delete=False
        ) as f:
            self.partial = Path(f.name)
        self.conn = sqlite3.connect(self.partial)
        self.conn.executescript(_SCHEMA)
        return self

    def append(self, user_ids: pd.Series, categories: pd.DataFrame) -> None:
        """Store one batch: the contract's output and its review flags, plus each row's user
        (the input's order)."""
        frame = categories.assign(user_id=user_ids.to_numpy())[list(COLUMNS)]
        frame = frame.assign(**{c: frame[c].astype(int) for c in FLAGS})
        frame.to_sql(CATEGORIES_TABLE, self.conn, if_exists="append", index=False)
        self.rows += len(frame)

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if kind is None:
                self.conn.execute(_INDEX)
                meta = {**self.meta, "rows": str(self.rows)}
                self.conn.executemany("INSERT INTO meta VALUES (?, ?)", sorted(meta.items()))
                self.conn.commit()
        finally:
            self.conn.close()
        if kind is None:
            self.partial.replace(self.path)
        else:
            self.partial.unlink(missing_ok=True)


def load_categories(path: str | Path, user_id: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    where, params = ("WHERE user_id = ?", (user_id,)) if user_id is not None else ("", ())
    sql = f"SELECT {', '.join(COLUMNS)} FROM {CATEGORIES_TABLE} {where} ORDER BY transaction_id"
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    return frame.assign(**{c: frame[c].astype(bool) for c in FLAGS})


def load_prediction_meta(path: str | Path) -> dict[str, str]:
    conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        return dict(conn.execute("SELECT key, value FROM meta ORDER BY key").fetchall())
    finally:
        conn.close()
