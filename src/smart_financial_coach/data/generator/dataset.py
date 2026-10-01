"""The generated dataset: table schemas, the in-memory container, and its content hash.

Model-visible tables have plain names. Ground truth lives in `truth_*` tables, which the feature
pipeline must never read.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

SqlType = Literal["TEXT", "REAL", "INTEGER"]


@dataclass(frozen=True)
class Column:
    name: str
    type: SqlType
    nullable: bool = False


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...]

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


def _t(name: str, pk: tuple[str, ...], *columns: Column) -> Table:
    return Table(name, columns, pk)


C = Column
TABLES: dict[str, Table] = {
    t.name: t
    for t in (
        _t(
            "users",
            ("user_id",),
            C("user_id", "TEXT"),
            C("split", "TEXT"),
            C("persona", "TEXT"),
            C("timezone", "TEXT"),
            C("monthly_income_estimate", "REAL"),
            C("starting_balance", "REAL"),
        ),
        _t(
            "transactions",
            ("transaction_id",),
            C("transaction_id", "TEXT"),
            C("user_id", "TEXT"),
            C("ts", "TEXT"),
            C("amount", "REAL"),
            C("currency", "TEXT"),
            C("merchant_raw", "TEXT"),
            C("channel", "TEXT"),
        ),
        _t(
            "goals",
            ("goal_id",),
            C("goal_id", "TEXT"),
            C("user_id", "TEXT"),
            C("name", "TEXT"),
            C("target_amount", "REAL"),
            C("created_date", "TEXT"),
            C("target_date", "TEXT"),
            C("as_of_date", "TEXT"),
            C("current_balance", "REAL"),
        ),
        _t(
            "truth_transactions",
            ("transaction_id",),
            C("transaction_id", "TEXT"),
            C("category", "TEXT"),
            C("merchant_id", "TEXT"),
            C("process", "TEXT"),
            C("is_recurring", "INTEGER"),
            C("anomaly_kind", "TEXT", nullable=True),
            # Unusual charges only: weak when an amount outlier isn't above normal charges
            C("tier", "TEXT", nullable=True),
            # Duplicate: the original charge. Refund: the purchase refunded.
            C("related_transaction_id", "TEXT", nullable=True),
        ),
        _t(
            "truth_periods",
            ("spike_id",),
            C("spike_id", "TEXT"),
            C("user_id", "TEXT"),
            C("granularity", "TEXT"),
            C("period_start", "TEXT"),
            C("period_end", "TEXT"),  # inclusive
            C("category", "TEXT"),
            C("multiplier", "REAL"),
            C("expected_count", "REAL"),  # Poisson-process purchases, without the spike
            C("expected_spend", "REAL"),  # all processes, without the spike
            C("base_spend", "REAL"),  # realized, everything except the spike's extra purchases
            C("extra_spend", "REAL"),  # realized, the spike's extra purchases
            C("tier", "TEXT"),  # clear or weak
        ),
        _t(
            "truth_expected",
            ("user_id", "category", "granularity", "period_start"),
            C("user_id", "TEXT"),
            C("category", "TEXT"),
            C("granularity", "TEXT"),
            C("period_start", "TEXT"),
            C("expected_count", "REAL"),
            C("expected_spend", "REAL"),
        ),
        _t(
            "truth_goals",
            ("goal_id",),
            C("goal_id", "TEXT"),
            C("outcome_class", "TEXT"),
            C("met", "INTEGER", nullable=True),
        ),
        _t(
            "truth_merchants",
            ("merchant_id",),
            C("merchant_id", "TEXT"),
            C("canonical_name", "TEXT"),
            C("category", "TEXT"),
            C("subtype", "TEXT"),
            C("price_median", "REAL"),
            C("price_sigma", "REAL"),
            C("is_ambiguous", "INTEGER"),
            C("holdout", "INTEGER"),
        ),
    )
}
MODEL_TABLES = ("users", "transactions", "goals")
TRUTH_TABLES = (
    "truth_transactions",
    "truth_periods",
    "truth_expected",
    "truth_goals",
    "truth_merchants",
)


def _canonical(values: pd.Series, sqltype: SqlType) -> list[str]:
    """Type-normalized text so in-memory and SQLite round-tripped data hash identically."""
    missing = values.isna().tolist()
    if sqltype == "INTEGER":
        return [
            "\x00" if na else str(int(v)) for v, na in zip(values.tolist(), missing, strict=True)
        ]
    if sqltype == "REAL":
        return [
            "\x00" if na else repr(float(v)) for v, na in zip(values.tolist(), missing, strict=True)
        ]
    return ["\x00" if na else str(v) for v, na in zip(values.tolist(), missing, strict=True)]


@dataclass(frozen=True)
class Dataset:
    tables: dict[str, pd.DataFrame]
    meta: dict[str, str]

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables[name]

    def content_hash(self) -> str:
        """SHA-256 over every table's rows in primary-key order, plus meta."""
        digest = hashlib.sha256()
        for name, table in TABLES.items():
            df = self.tables[name].sort_values(list(table.primary_key), kind="mergesort")
            digest.update(f"\x1e{name}\x1e".encode())
            for col in table.columns:
                digest.update(f"\x1d{col.name}\x1d".encode())
                digest.update("\x1f".join(_canonical(df[col.name], col.type)).encode())
        for key in sorted(self.meta):
            digest.update(f"\x1e{key}={self.meta[key]}".encode())
        return digest.hexdigest()

    def to_sqlite(self, path: str | Path, *, overwrite: bool = False) -> Path:
        from smart_financial_coach.data.generator.sqlite_io import write_sqlite

        return write_sqlite(self, Path(path), overwrite=overwrite)
