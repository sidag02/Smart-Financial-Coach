"""Unusual-transaction flags, kept in their own SQLite file like categorization predictions (FR-7
§8; FR-3 option C-b). One file holds one model version's flags for one dataset.

    unusual_flags  flag_id (PK), transaction_id, user_id (indexed), score, reason_code,
                   evidence (JSON text), model_version
    meta           model_version, data_spec_hash, data_spec_name, created_at, rows

Only flagged charges are stored. A flag's id is the model version and the transaction id, so it
stays stable for the v1.1 flag actions (FR-9) to key on. The file appears only when a run
completes, as `CategoryWriter`'s does.

    with FlagWriter("data/flags/default.sqlite", meta) as writer:
        writer.append(user_ids, scorer.score_transactions(rows))
    load_flags("data/flags/default.sqlite", user_id="u_te_yp_0007")
"""

import sqlite3
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Self

import pandas as pd

FLAGS_TABLE = "unusual_flags"
COLUMNS = (
    "flag_id",
    "transaction_id",
    "user_id",
    "score",
    "reason_code",
    "evidence",
    "model_version",
)
_SCHEMA = f"""
CREATE TABLE {FLAGS_TABLE} (
    flag_id TEXT PRIMARY KEY,
    transaction_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    score REAL NOT NULL,
    reason_code TEXT NOT NULL,
    evidence TEXT NOT NULL,
    model_version TEXT NOT NULL
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
_INDEX = f"CREATE INDEX idx_{FLAGS_TABLE}_user ON {FLAGS_TABLE} (user_id)"


def flag_id(model_version: str, transaction_id: str) -> str:
    return f"{model_version}:{transaction_id}"


class FlagWriter:
    """Writes one scoring run's flags; the file appears at `path` only if the run completes."""

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

    def append(self, user_ids: pd.Series, scored: pd.DataFrame) -> None:
        """Store one batch's flagged rows: the contract's output, plus each row's user."""
        frame = scored.assign(user_id=user_ids.to_numpy())
        frame = frame[frame["is_flagged"].astype(bool)]
        frame = frame.assign(
            flag_id=[
                flag_id(v, t)
                for v, t in zip(frame["model_version"], frame["transaction_id"], strict=True)
            ]
        )
        frame[list(COLUMNS)].to_sql(FLAGS_TABLE, self.conn, if_exists="append", index=False)
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


def load_flags(path: str | Path, user_id: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    where, params = ("WHERE user_id = ?", (user_id,)) if user_id is not None else ("", ())
    sql = f"SELECT {', '.join(COLUMNS)} FROM {FLAGS_TABLE} {where} ORDER BY transaction_id"
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def load_flag_meta(path: str | Path) -> dict[str, str]:
    conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        return dict(conn.execute("SELECT key, value FROM meta ORDER BY key").fetchall())
    finally:
        conn.close()
