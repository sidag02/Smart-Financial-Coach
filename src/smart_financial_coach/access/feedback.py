"""Category feedback (FR-5, FR-6): corrections are events; overrides are state derived from them.

Design: FR-5 and FR-6 feature design, §3. A correction or confirmation applies to its subject at
once, for one merchant (every transaction at that normalized merchant string) or one transaction.
Overrides are rebuilt from the event log, so undo marks an event and replays the rest, and the log
stays an audit trail. Effective categories are applied per subject after the shared predictions
(`effective_categories`): a transaction override, then a merchant override, then the model.

    store = FeedbackStore(path, categories)
    c = store.record(subject, user_id, action="correct", scope="merchant",
                     merchant_key="costco whse", to_category="Groceries", from_category="Shopping",
                     source="review", model_version="20eea4fb-...")
    store.undo(subject, c.correction_id)
    view = effective_categories(ledger.transactions, store.overrides(subject, user_id))

`subject` is who gave the feedback, and every read is scoped by it: in production the signed-in
user; in the demo, where visitors share accounts, the browser session (owner, Oct 3, 2026), so
two visitors on one account never see each other's corrections. `user_id` is whose ledger the
feedback is about. Neither ever comes from a tool argument (Technical Design, "Security and data
isolation").
"""

import sqlite3
import threading
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

INCOME = "Income"
ACTIONS = ("confirm", "correct")
SCOPES = ("merchant", "transaction")
SOURCES = ("review", "edit", "coach")
CATEGORY_SOURCES = ("model", "you")  # where a transaction's effective category came from

_SCHEMA = """
CREATE TABLE IF NOT EXISTS category_corrections (
    correction_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL UNIQUE,  -- order of events; later wins
    subject TEXT NOT NULL,
    user_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('confirm', 'correct')),
    scope TEXT NOT NULL CHECK (scope IN ('merchant', 'transaction')),
    merchant_key TEXT NOT NULL,
    transaction_id TEXT,  -- set for scope 'transaction' only
    from_category TEXT NOT NULL,
    to_category TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('review', 'edit', 'coach')),
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    undone_at TEXT,
    CHECK ((scope = 'transaction') = (transaction_id IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS idx_corrections_subject ON category_corrections (subject, user_id, seq);
CREATE INDEX IF NOT EXISTS idx_corrections_merchant ON category_corrections (merchant_key);
"""
_COLUMNS = (
    "correction_id",
    "seq",
    "subject",
    "user_id",
    "action",
    "scope",
    "merchant_key",
    "transaction_id",
    "from_category",
    "to_category",
    "source",
    "model_version",
    "created_at",
    "undone_at",
)


class FeedbackError(ValueError):
    pass


@dataclass(frozen=True)
class Correction:
    correction_id: str
    seq: int
    subject: str
    user_id: str
    action: str
    scope: str
    merchant_key: str
    transaction_id: str | None
    from_category: str
    to_category: str
    source: str
    model_version: str
    created_at: str
    undone_at: str | None

    @property
    def undone(self) -> bool:
        return self.undone_at is not None


@dataclass(frozen=True)
class Overrides:
    """One subject's current overrides for one user's ledger."""

    merchant: dict[str, str] = field(default_factory=dict)  # merchant_key -> category
    transaction: dict[str, str] = field(default_factory=dict)  # transaction_id -> category

    @classmethod
    def replay(cls, events: Iterable[Correction]) -> "Overrides":
        """Current state from events in order: undone events are skipped, later ones win.

        A merchant-wide change supersedes earlier single-transaction changes at that merchant
        (review on #35): "every <merchant> transaction" means every one, including a row the user
        had set on its own. A single-transaction change made after it still wins for its row.
        """
        merchant: dict[str, str] = {}
        transaction: dict[str, tuple[str, str]] = {}  # transaction_id -> (merchant_key, category)
        for e in events:
            if e.undone:
                continue
            if e.scope == "transaction" and e.transaction_id is not None:
                transaction[e.transaction_id] = (e.merchant_key, e.to_category)
            else:
                merchant[e.merchant_key] = e.to_category
                transaction = {t: v for t, v in transaction.items() if v[0] != e.merchant_key}
        return cls(merchant, {t: category for t, (_, category) in transaction.items()})


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class FeedbackStore:
    """The feedback store: a SQLite file per deployment (Postgres with row-level security in v2).

    Safe to share across request threads: each call opens its own connection, and writes are
    serialized by a lock (one app replica in the demo).
    """

    def __init__(self, path: Path, categories: Iterable[str]) -> None:
        self.path = Path(path)
        self.categories = frozenset(categories)
        self._write = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record(
        self,
        subject: str,
        user_id: str,
        *,
        action: str,
        scope: str,
        merchant_key: str,
        from_category: str,
        to_category: str,
        source: str,
        model_version: str,
        transaction_id: str | None = None,
    ) -> Correction:
        """Store one confirmation or correction. A confirmation keeps the category it was shown
        (`to_category == from_category`): it pins that category for the subject, so a later
        model can't silently change it, and it counts as a vote (§4)."""
        if not subject or not user_id:
            raise FeedbackError("feedback needs a subject and a user")
        if action not in ACTIONS or scope not in SCOPES or source not in SOURCES:
            raise FeedbackError(f"unknown action, scope or source: {action}, {scope}, {source}")
        if to_category not in self.categories:
            raise FeedbackError(f"unknown category {to_category!r}")
        if action == "confirm" and to_category != from_category:
            raise FeedbackError("a confirmation keeps the category it was shown")
        if action == "correct" and to_category == from_category:
            raise FeedbackError(f"the category is already {to_category}")
        if not merchant_key:
            raise FeedbackError("feedback needs the merchant")
        if (scope == "transaction") != (transaction_id is not None):
            raise FeedbackError("a transaction-scope correction needs exactly one transaction")
        with self._write, self._connect() as conn:
            (seq,) = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM category_corrections"
            ).fetchone()
            row = (
                uuid.uuid4().hex,
                seq,
                subject,
                user_id,
                action,
                scope,
                merchant_key,
                transaction_id,
                from_category,
                to_category,
                source,
                model_version,
                _now(),
                None,
            )
            conn.execute(
                f"INSERT INTO category_corrections ({', '.join(_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _COLUMNS)})",
                row,
            )
        return Correction(*row)

    def undo(self, subject: str, correction_id: str) -> Correction:
        """Mark the subject's correction undone; nothing is deleted. Someone else's correction is
        indistinguishable from one that doesn't exist."""
        with self._write, self._connect() as conn:
            found = self._select(
                conn, "subject = ? AND correction_id = ?", (subject, correction_id)
            )
            if not found:
                raise FeedbackError(f"no correction {correction_id!r}")
            if found[0].undone:
                raise FeedbackError("that correction is already undone")
            undone_at = _now()
            conn.execute(
                "UPDATE category_corrections SET undone_at = ? WHERE correction_id = ?",
                (undone_at, correction_id),
            )
        return Correction(**{**found[0].__dict__, "undone_at": undone_at})

    def corrections(self, subject: str, user_id: str, limit: int | None = None) -> list[Correction]:
        """The subject's feedback on one user's ledger, newest first."""
        with self._connect() as conn:
            events = self._select(conn, "subject = ? AND user_id = ?", (subject, user_id))
        newest = sorted(events, key=lambda e: e.seq, reverse=True)
        return newest[:limit] if limit is not None else newest

    def overrides(self, subject: str, user_id: str) -> Overrides:
        with self._connect() as conn:
            events = self._select(conn, "subject = ? AND user_id = ?", (subject, user_id))
        return Overrides.replay(sorted(events, key=lambda e: e.seq))

    @staticmethod
    def _select(conn: sqlite3.Connection, where: str, params: tuple[str, ...]) -> list[Correction]:
        sql = f"SELECT {', '.join(_COLUMNS)} FROM category_corrections WHERE {where}"
        return [Correction(*row) for row in conn.execute(sql, params).fetchall()]


def effective_categories(transactions: pd.DataFrame, overrides: Overrides) -> pd.DataFrame:
    """The ledger as one subject sees it: `category` becomes the effective category.

    Adds `model_category` (the model's), `category_source` ("model" or "you") and keeps
    `needs_review` only where the model's category still stands: an override settles review.
    `transactions` needs `transaction_id`, `merchant_key`, `category` and `needs_review`.

    A merchant override leaves rows the model predicted as Income alone: review covers spending
    rows only (owner, on #32), so settling a spending flag never moves income, such as a refund
    at that merchant (review on #35). Those rows can still be changed one at a time.
    """
    out = transactions.copy()
    out["model_category"] = out["category"]
    spending = out["model_category"] != INCOME
    by_merchant = out["merchant_key"].map(overrides.merchant).where(spending)
    by_transaction = out["transaction_id"].map(overrides.transaction)
    chosen = by_transaction.fillna(by_merchant)
    yours = chosen.notna()
    out["category"] = chosen.where(yours, out["category"])
    out["category_source"] = pd.Series("model", index=out.index).where(~yours, "you")
    out["needs_review"] = out["needs_review"] & ~yours
    return out
