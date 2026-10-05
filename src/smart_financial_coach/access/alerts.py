"""Alert sensitivity and flag actions (FR-9 §3): events in the feedback store, replayed per user.

Two event tables live in the FR-5/FR-6 feedback store's SQLite file and are scoped the same way:
`subject` is who acted (the signed-in user in production; the browser session in the demo, where
visitors share accounts), and `user_id` is whose ledger it's about. Neither ever comes from a tool
argument (Technical Design, "Security and data isolation").

- **The setting** is the subject's latest `alert_settings` event for the user; with none, Balanced.
  Changing it back is the undo.
- **Flag actions** are events that are undone by marking them; nothing is deleted. What each
  hides is replayed from the actions that aren't undone (decisions 9 and 14):

  - `recognize` on an amount or new-merchant flag hides it and every later flag with the same
    reason at the same merchant (`merchant_key`, `reason_code`, `transaction_ts`);
  - `recognize` on a duplicate hides that flag only (`transaction_id`);
  - `expected` hides that spike's category and month (`category`, `period_start`);
  - `not_me` hides nothing; the flag is marked (`transaction_id`).

  Every action is matched on stored fields, never on `flag_id` alone, so it survives a promotion
  that gives flags new ids. Actions never reach the model, another user or another subject.

    store = AlertStore(path)
    store.set_level(subject, user_id, "more", source="page")
    a = store.record(subject, user_id, flag_id=..., kind="charge", action="recognize", ...)
    view = AlertView.replay(store.actions(subject, user_id))
    view.hiding_charge(transaction_id, merchant_key, reason_code, ts)  # the action, or None
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

from smart_financial_coach.intelligence.presets import DEFAULT_LEVEL, LEVELS, check_level

KINDS = ("charge", "spike")
ACTIONS = {"charge": ("recognize", "not_me"), "spike": ("expected",)}
SOURCES = ("page", "coach", "assistant")
DUPLICATE = "duplicate"  # a duplicate's "I recognize this" hides that flag only (decision 9)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alert_settings (
    setting_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL UNIQUE,
    subject TEXT NOT NULL,
    user_id TEXT NOT NULL,
    level TEXT NOT NULL CHECK (level IN ('less', 'balanced', 'more')),
    source TEXT NOT NULL CHECK (source IN ('page', 'coach', 'assistant')),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alert_settings_subject ON alert_settings (subject, user_id, seq);
CREATE TABLE IF NOT EXISTS flag_actions (
    action_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL UNIQUE,
    subject TEXT NOT NULL,
    user_id TEXT NOT NULL,
    flag_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('charge', 'spike')),
    action TEXT NOT NULL CHECK (action IN ('recognize', 'not_me', 'expected')),
    transaction_id TEXT,  -- charges
    transaction_ts TEXT,
    merchant_key TEXT,
    reason_code TEXT,
    category TEXT,  -- spikes
    period_start TEXT,
    source TEXT NOT NULL CHECK (source IN ('page', 'coach', 'assistant')),
    created_at TEXT NOT NULL,
    undone_at TEXT,
    CHECK ((kind = 'charge') = (transaction_id IS NOT NULL)),
    CHECK ((kind = 'spike') = (period_start IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS idx_flag_actions_subject ON flag_actions (subject, user_id, seq);
"""
_ACTION_COLUMNS = (
    "action_id",
    "seq",
    "subject",
    "user_id",
    "flag_id",
    "kind",
    "action",
    "transaction_id",
    "transaction_ts",
    "merchant_key",
    "reason_code",
    "category",
    "period_start",
    "source",
    "created_at",
    "undone_at",
)


class AlertError(ValueError):
    pass


@dataclass(frozen=True)
class FlagAction:
    action_id: str
    seq: int
    subject: str
    user_id: str
    flag_id: str
    kind: str
    action: str
    transaction_id: str | None
    transaction_ts: str | None
    merchant_key: str | None
    reason_code: str | None
    category: str | None
    period_start: str | None
    source: str
    created_at: str
    undone_at: str | None

    @property
    def undone(self) -> bool:
        return self.undone_at is not None


@dataclass(frozen=True)
class AlertView:
    """What one subject's actions hide or mark on one user's flags. Each map's value is the
    action that does it, so a hidden alert can be shown again by undoing that action."""

    # (merchant_key, reason_code) -> (the recognized charge's time, action): it and later ones
    recognized: dict[tuple[str, str], tuple[pd.Timestamp, str]] = field(default_factory=dict)
    recognized_ids: dict[str, str] = field(default_factory=dict)  # transaction_id -> action
    expected: dict[tuple[str, str], str] = field(
        default_factory=dict
    )  # (category, month) -> action
    not_me: dict[str, str] = field(default_factory=dict)  # transaction_id -> action

    @classmethod
    def replay(cls, actions: Iterable[FlagAction]) -> "AlertView":
        recognized: dict[tuple[str, str], tuple[pd.Timestamp, str]] = {}
        ids: dict[str, str] = {}
        expected: dict[tuple[str, str], str] = {}
        not_me: dict[str, str] = {}
        for a in sorted(actions, key=lambda a: a.seq):
            if a.undone:
                continue
            if a.action == "expected" and a.category and a.period_start:
                expected[(a.category, a.period_start[:10])] = a.action_id
            elif a.action == "not_me" and a.transaction_id:
                not_me[a.transaction_id] = a.action_id
            elif a.action == "recognize" and a.transaction_id:
                ids[a.transaction_id] = a.action_id
                if a.reason_code != DUPLICATE and a.merchant_key and a.reason_code:
                    assert a.transaction_ts is not None  # a charge's action records its time
                    key = (a.merchant_key, a.reason_code)
                    ts = pd.Timestamp(a.transaction_ts)
                    if key not in recognized or ts < recognized[key][0]:
                        recognized[key] = (ts, a.action_id)
        return cls(recognized, ids, expected, not_me)

    def hiding_charge(
        self, transaction_id: str, merchant_key: str, reason_code: str, ts: pd.Timestamp
    ) -> str | None:
        """The action that hides this charge's flag, or None."""
        if transaction_id in self.recognized_ids:
            return self.recognized_ids[transaction_id]
        if reason_code == DUPLICATE:
            return None
        since = self.recognized.get((merchant_key, reason_code))
        return since[1] if since is not None and ts >= since[0] else None

    def hiding_spike(self, category: str, period_start: str) -> str | None:
        """The action that hides this spike, or None."""
        return self.expected.get((category, period_start[:10]))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class AlertStore:
    """Settings and flag actions, in the feedback store's SQLite file (Postgres in v2).

    Safe to share across request threads, as `FeedbackStore` is: each call opens its own
    connection, and writes are serialized by a lock (one app replica in the demo).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
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

    # --- The setting ---------------------------------------------------------------------

    def level(self, subject: str, user_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT level FROM alert_settings WHERE subject = ? AND user_id = ? "
                "ORDER BY seq DESC LIMIT 1",
                (subject, user_id),
            ).fetchone()
        return str(row[0]) if row else DEFAULT_LEVEL

    def set_level(self, subject: str, user_id: str, level: str, *, source: str) -> str:
        _check(subject, user_id, source)
        check_level(level)
        with self._write, self._connect() as conn:
            (seq,) = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM alert_settings").fetchone()
            conn.execute(
                "INSERT INTO alert_settings VALUES (?, ?, ?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, seq, subject, user_id, level, source, _now()),
            )
        return level

    # --- Flag actions --------------------------------------------------------------------

    def record(
        self,
        subject: str,
        user_id: str,
        *,
        flag_id: str,
        kind: str,
        action: str,
        source: str,
        transaction_id: str | None = None,
        transaction_ts: str | None = None,
        merchant_key: str | None = None,
        reason_code: str | None = None,
        category: str | None = None,
        period_start: str | None = None,
    ) -> FlagAction:
        _check(subject, user_id, source)
        if kind not in KINDS or action not in ACTIONS[kind]:
            raise AlertError(f"{action!r} isn't an action on a {kind}")
        if kind == "charge" and not (transaction_id and transaction_ts and merchant_key):
            raise AlertError("an action on a charge needs its transaction, time and merchant")
        if kind == "charge" and not reason_code:
            raise AlertError("an action on a charge needs its reason")
        if kind == "spike" and not (category and period_start):
            raise AlertError("an action on a spike needs its category and month")
        if any(
            a.flag_id == flag_id and a.action == action and not a.undone
            for a in self.actions(subject, user_id)
        ):
            raise AlertError("you've already done that for this alert")
        with self._write, self._connect() as conn:
            (seq,) = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM flag_actions").fetchone()
            row = (
                uuid.uuid4().hex,
                seq,
                subject,
                user_id,
                flag_id,
                kind,
                action,
                transaction_id if kind == "charge" else None,
                transaction_ts if kind == "charge" else None,
                merchant_key if kind == "charge" else None,
                reason_code if kind == "charge" else None,
                category if kind == "spike" else None,
                period_start if kind == "spike" else None,
                source,
                _now(),
                None,
            )
            conn.execute(
                f"INSERT INTO flag_actions ({', '.join(_ACTION_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _ACTION_COLUMNS)})",
                row,
            )
        return FlagAction(*row)

    def undo(self, subject: str, action_id: str) -> FlagAction:
        """Mark the subject's action undone. Someone else's is indistinguishable from none."""
        with self._write, self._connect() as conn:
            found = self._select(conn, "subject = ? AND action_id = ?", (subject, action_id))
            if not found:
                raise AlertError(f"no alert action {action_id!r}")
            if found[0].undone:
                raise AlertError("that action is already undone")
            undone_at = _now()
            conn.execute(
                "UPDATE flag_actions SET undone_at = ? WHERE action_id = ?", (undone_at, action_id)
            )
        return FlagAction(**{**found[0].__dict__, "undone_at": undone_at})

    def actions(self, subject: str, user_id: str) -> list[FlagAction]:
        """The subject's actions on one user's flags, newest first."""
        with self._connect() as conn:
            found = self._select(conn, "subject = ? AND user_id = ?", (subject, user_id))
        return sorted(found, key=lambda a: a.seq, reverse=True)

    def view(self, subject: str, user_id: str) -> AlertView:
        return AlertView.replay(self.actions(subject, user_id))

    @staticmethod
    def _select(conn: sqlite3.Connection, where: str, params: tuple[str, ...]) -> list[FlagAction]:
        sql = f"SELECT {', '.join(_ACTION_COLUMNS)} FROM flag_actions WHERE {where}"
        return [FlagAction(*row) for row in conn.execute(sql, params).fetchall()]


def _check(subject: str, user_id: str, source: str) -> None:
    if not subject or not user_id:
        raise AlertError("alert settings need a subject and a user")
    if source not in SOURCES:
        raise AlertError(f"unknown source {source!r}")


__all__ = ["LEVELS", "AlertError", "AlertStore", "AlertView", "FlagAction"]
