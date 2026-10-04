"""Savings goals (FR-10): changes are events; a user's goals are state derived from them.

Design: FR-10 Savings Goals feature design, §1-3. A user's goals are the dataset's generated goals
with one subject's changes replayed over them by `goal_id`: an edit replaces a goal, an archive
hides it, and a creation adds one. Each event stores the goal's full state after it, so replay is
"the last live event per goal", and undo marks an event and replays the rest. Validation is one
function for `check_goal`, the write tools and the form, so a message reads the same everywhere.

    store = GoalStore(path)
    rev = store.create(ledger, subject, GoalDraft("Trip", 3000, "2027-06-01"), source="edit")
    store.goals(ledger, subject)  # [..., Goal(name="Trip", target_date=date(2027, 6, 30), ...)]
    store.undo(ledger, subject, rev.revision_id)

`subject` is whose changes these are, as in FR-5's feedback store: the signed-in user in
production, the browser session in the demo, so two visitors on one demo account never see each
other's goals. Neither `subject` nor the user ever comes from a tool argument (Technical Design,
"Security and data isolation"). Amounts are integer cents in the store and the `Goal`; callers
pass dollars, parsed exactly (`to_cents`).
"""

import calendar
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal, DecimalException, Inexact, localcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from smart_financial_coach.access.ledger import Ledger

OPS = ("create", "update", "archive")
SOURCES = ("edit", "coach", "assistant")
STATUSES = ("active", "reached", "ended", "archived")

# Product limits (FR-10 design, §3; owner, Oct 3, 2026)
NAME_MAX = 40
AMOUNT_MIN_CENTS = 50_00
AMOUNT_MAX_CENTS = 1_000_000_00
MONTHS_MAX = 120
ACTIVE_MAX = 10
# Longer text isn't an amount a person typed; refusing it keeps parsing cheap (review on #42)
AMOUNT_TEXT_MAX = 32

_SCHEMA = """
CREATE TABLE IF NOT EXISTS goal_revisions (
    revision_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL UNIQUE,  -- order of events; later wins
    subject TEXT NOT NULL,
    user_id TEXT NOT NULL,
    goal_id TEXT NOT NULL,
    op TEXT NOT NULL CHECK (op IN ('create', 'update', 'archive')),
    name TEXT NOT NULL,           -- the goal's full state after this event
    target_amount_cents INTEGER NOT NULL,
    target_date TEXT NOT NULL,    -- a month end
    saved_cents INTEGER NOT NULL,
    saved_as_of TEXT NOT NULL,
    created_date TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('edit', 'coach', 'assistant')),
    created_at TEXT NOT NULL,
    undone_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_goal_revisions_subject ON goal_revisions (subject, user_id, seq);
"""
_COLUMNS = (
    "revision_id",
    "seq",
    "subject",
    "user_id",
    "goal_id",
    "op",
    "name",
    "target_amount_cents",
    "target_date",
    "saved_cents",
    "saved_as_of",
    "created_date",
    "source",
    "created_at",
    "undone_at",
)


# Months and money


def month_end(day: date) -> date:
    return day.replace(day=calendar.monthrange(day.year, day.month)[1])


def _month_index(day: date) -> int:
    return day.year * 12 + day.month - 1


def months_left(today: date, target: date) -> int:
    """Month ends after `today` up to `target`: from Sep 30, 2026 to Jun 30, 2027 is 9."""
    if target <= today:
        return 0
    count = _month_index(target) - _month_index(today)
    return count + (0 if today == month_end(today) else 1)


class FractionOfCentError(ValueError):
    pass


class AmountTooLargeError(ValueError):
    pass


def to_cents(amount: Any) -> int:
    """Dollars to integer cents, exactly: through `Decimal(str(x))`, never `x * 100`, which turns
    19.99 into 1998.999…. Raises ValueError for anything that isn't a number,
    `FractionOfCentError` for a fraction of a cent, and `AmountTooLargeError` past a quadrillion
    dollars (far beyond any limit), so a huge exponent never reaches the arithmetic."""
    if isinstance(amount, bool) or not isinstance(amount, int | float | str | Decimal):
        raise ValueError(f"not an amount: {amount!r}")
    text = str(amount).strip()
    if len(text) > AMOUNT_TEXT_MAX:
        raise ValueError("not an amount: too long")
    try:
        value = Decimal(text)
        if not value.is_finite():
            raise ValueError(f"not an amount: {amount!r}")
        if value.adjusted() > 15:
            raise AmountTooLargeError(f"{amount!r} is too large")
        if value != 0 and value.adjusted() < -2:  # below a cent, however many zeros
            raise FractionOfCentError(f"{amount!r} has a fraction of a cent")
        with localcontext() as exact:
            exact.prec = 2 * AMOUNT_TEXT_MAX  # enough for every digit of the text, and the cents
            exact.traps[Inexact] = True  # so nothing below is ever rounded
            cents = value.scaleb(2)
            if cents != cents.to_integral_value():
                raise FractionOfCentError(f"{amount!r} has a fraction of a cent")
            return int(cents)
    except DecimalException as error:
        raise ValueError(f"not an amount: {amount!r}") from error


def _money(cents: int) -> str:
    return f"${cents // 100:,}" if cents % 100 == 0 else f"${cents / 100:,.2f}"


def _month_name(index: int) -> str:
    return f"{calendar.month_name[index % 12 + 1]} {index // 12}"


# The goal record (§1)


@dataclass(frozen=True)
class Goal:
    goal_id: str
    name: str
    target_cents: int
    target_date: date  # a month end
    saved_cents: int
    saved_as_of: date
    created_date: date
    origin: str  # "existing" (generated) or "yours"
    archived: bool = False
    undo_revision_id: str | None = None  # this subject's latest change that can be undone

    def status(self, today: date) -> str:
        """Checked in this order: archived, ended (whatever was saved), reached, active."""
        if self.archived:
            return "archived"
        if self.target_date <= today:
            return "ended"
        if self.saved_cents >= self.target_cents:
            return "reached"
        return "active"

    def running(self, today: date) -> bool:
        return self.status(today) in ("active", "reached")

    def months_left(self, today: date) -> int:
        return months_left(today, self.target_date)

    def needed_per_month_cents(self, today: date) -> int:
        """(target - saved) / months left, rounded up to the dollar; 0 unless active."""
        if self.status(today) != "active":
            return 0
        months = self.months_left(today)
        return -(-(self.target_cents - self.saved_cents) // (months * 100)) * 100


def generated_goals(frame: pd.DataFrame) -> list[Goal]:
    """The dataset's goals (FR-1) as records; balances convert to cents exactly."""
    return [
        Goal(
            goal_id=str(g["goal_id"]),
            name=str(g["name"]),
            target_cents=to_cents(g["target_amount"]),
            target_date=date.fromisoformat(str(g["target_date"])),
            saved_cents=to_cents(g["current_balance"]),
            saved_as_of=date.fromisoformat(str(g["as_of_date"])),
            created_date=date.fromisoformat(str(g["created_date"])),
            origin="existing",
        )
        for g in frame.to_dict("records")
    ]


# Validation (§3)


@dataclass(frozen=True)
class Problem:
    field: str | None  # the form field it belongs to; None for the goal as a whole
    code: str
    message: str


class GoalError(ValueError):
    """A change that breaks a rule (with its problems), or a goal or change that isn't there."""

    def __init__(self, message: str, problems: tuple[Problem, ...] = ()) -> None:
        super().__init__(message)
        self.problems = problems


@dataclass(frozen=True)
class GoalDraft:
    """What a caller asks for, as given: dollars and a `YYYY-MM-DD` day (or a `date`)."""

    name: Any
    target_amount: Any
    target_date: Any
    saved: Any = 0


@dataclass(frozen=True)
class Checked:
    """A draft that passed: the goal it would be, or the problems that stop it."""

    problems: tuple[Problem, ...]
    goal: Goal | None

    @property
    def valid(self) -> bool:
        return not self.problems


def edited_draft(goal: Goal, changes: dict[str, Any]) -> GoalDraft:
    """The draft an edit makes: `changes` (any of name, target_amount, target_date, saved) over
    the goal's current values."""
    _check_changes(changes)
    return GoalDraft(
        name=changes.get("name", goal.name),
        target_amount=changes.get("target_amount", Decimal(goal.target_cents) / 100),
        target_date=changes.get("target_date", goal.target_date),
        saved=changes.get("saved", Decimal(goal.saved_cents) / 100),
    )


def _check_changes(changes: dict[str, Any]) -> None:
    unknown = set(changes) - {"name", "target_amount", "target_date", "saved"}
    if unknown:
        raise GoalError(f"can't change {', '.join(sorted(unknown))}")


def _parse_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value.strip())
    raise ValueError(f"not a date: {value!r}")


def check_draft(
    draft: GoalDraft,
    today: date,
    others: list[Goal],
    *,
    editing: Goal | None = None,
) -> Checked:
    """Validate a new goal (`editing` None) or an edit of `editing` against the user's `others`
    (every listed goal but this one). Returns the goal it would be, dated as a month end."""
    if editing is not None and editing.status(today) == "ended":
        return Checked((Problem(None, "goal_not_editable", "This goal's date has passed."),), None)

    problems: list[Problem] = []
    name = draft.name.strip() if isinstance(draft.name, str) else ""
    if not name:
        problems.append(Problem("name", "name_missing", "Give the goal a name."))
    elif len(name) > NAME_MAX:
        problems.append(
            Problem("name", "name_too_long", f"Keep the name to {NAME_MAX} characters.")
        )
    else:
        problems += _name_taken(name, others, today)

    target_cents = _amount(draft.target_amount, "target_amount", problems)
    if target_cents is not None and not AMOUNT_MIN_CENTS <= target_cents <= AMOUNT_MAX_CENTS:
        if target_cents < AMOUNT_MIN_CENTS:
            message = f"Goals start at {_money(AMOUNT_MIN_CENTS)}."
            problems.append(Problem("target_amount", "amount_range", message))
        else:
            problems.append(_too_large("target_amount"))
        target_cents = None

    target: date | None = None
    try:
        target = month_end(_parse_date(draft.target_date))
    except (TypeError, ValueError):
        problems.append(Problem("target_date", "date_invalid", "Use a date like 2027-06-30."))
    # An edit that keeps the date isn't held to today's window: a goal due this month can still be
    # renamed when today isn't a month end (review on #42)
    if target is not None and not (editing is not None and target == editing.target_date):
        ahead = _month_index(target) - _month_index(today)
        if ahead < 1:
            first = _month_name(_month_index(today) + 1)
            problems.append(Problem("target_date", "date_too_soon", f"Pick {first} or later."))
        elif ahead > MONTHS_MAX:
            problems.append(Problem("target_date", "date_too_far", "Pick a date within 10 years."))

    saved_cents = _amount(draft.saved, "saved", problems)
    if saved_cents is not None and saved_cents < 0:
        problems.append(Problem("saved", "saved_range", "The saved amount can't be negative."))
    elif saved_cents is not None and saved_cents > AMOUNT_MAX_CENTS:
        problems.append(_too_large("saved"))
        saved_cents = None
    elif (
        editing is None
        and saved_cents is not None
        and target_cents is not None
        and saved_cents >= target_cents
    ):
        problems.append(Problem("saved", "saved_range", "That's already the whole amount."))

    # The goal will be active (its date is in the future) unless the saved amount reaches it
    if target_cents is not None and saved_cents is not None and saved_cents < target_cents:
        problems += _too_many(others, today)

    if problems or target_cents is None or saved_cents is None or target is None:
        return Checked(tuple(problems), None)
    saved_as_of = today
    if editing is not None and saved_cents == editing.saved_cents:
        saved_as_of = editing.saved_as_of  # an unchanged balance keeps the day it was recorded
    goal = Goal(
        goal_id=editing.goal_id if editing else f"gu_{uuid.uuid4().hex[:12]}",
        name=name,
        target_cents=target_cents,
        target_date=target,
        saved_cents=saved_cents,
        saved_as_of=saved_as_of,
        created_date=editing.created_date if editing else today,
        origin=editing.origin if editing else "yours",
    )
    return Checked((), goal)


def _amount(value: Any, field: str, problems: list[Problem]) -> int | None:
    try:
        return to_cents(value)
    except FractionOfCentError:
        problems.append(Problem(field, "amount_cents", "Use dollars and cents."))
    except AmountTooLargeError:
        problems.append(_too_large(field))
    except ValueError:
        problems.append(Problem(field, "amount_invalid", "Enter an amount in dollars."))
    return None


def _too_large(field: str) -> Problem:
    if field == "saved":
        return Problem(field, "saved_range", f"Saved amounts go up to {_money(AMOUNT_MAX_CENTS)}.")
    return Problem(field, "amount_range", f"Goals go up to {_money(AMOUNT_MAX_CENTS)}.")


# The store (§2)


@dataclass(frozen=True)
class Revision:
    revision_id: str
    seq: int
    subject: str
    user_id: str
    goal_id: str
    op: str
    name: str
    target_amount_cents: int
    target_date: str
    saved_cents: int
    saved_as_of: str
    created_date: str
    source: str
    created_at: str
    undone_at: str | None

    @property
    def undone(self) -> bool:
        return self.undone_at is not None

    def goal(self, origin: str) -> Goal:
        return Goal(
            goal_id=self.goal_id,
            name=self.name,
            target_cents=self.target_amount_cents,
            target_date=date.fromisoformat(self.target_date),
            saved_cents=self.saved_cents,
            saved_as_of=date.fromisoformat(self.saved_as_of),
            created_date=date.fromisoformat(self.created_date),
            origin=origin,
            archived=self.op == "archive",
            undo_revision_id=self.revision_id,
        )


def replay(base: list[Goal], events: list[Revision]) -> list[Goal]:
    """Every goal the subject has, archived ones included: the generated goals, then each goal's
    last live event in place of it. Ordered by target date, then name."""
    goals = {g.goal_id: g for g in base}
    for e in sorted(events, key=lambda e: e.seq):
        if e.undone:
            continue
        origin = goals[e.goal_id].origin if e.goal_id in goals else "yours"
        goals[e.goal_id] = e.goal(origin)
    return sorted(goals.values(), key=lambda g: (g.target_date, g.name.casefold(), g.goal_id))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class GoalStore:
    """The goal store: a SQLite file per deployment (Postgres with row-level security in v2).

    Safe to share across request threads: each call opens its own connection, and writes are
    serialized by a lock and validated again inside it (one app replica in the demo).
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

    # Reads

    def goals(
        self, ledger: "Ledger", subject: str, *, include_archived: bool = False
    ) -> list[Goal]:
        """The user's goals as `subject` sees them."""
        with self._connect() as conn:
            goals = self._replay(conn, ledger, subject)
        return [g for g in goals if include_archived or not g.archived]

    def goal(self, ledger: "Ledger", subject: str, goal_id: str) -> Goal:
        """One listed goal. Someone else's is indistinguishable from one that doesn't exist."""
        with self._connect() as conn:
            return _find(self._replay(conn, ledger, subject), goal_id)

    def revisions(self, subject: str, user_id: str) -> list[Revision]:
        """The subject's changes to one user's goals, newest first."""
        with self._connect() as conn:
            events = self._select(conn, "subject = ? AND user_id = ?", (subject, user_id))
        return sorted(events, key=lambda e: e.seq, reverse=True)

    def check(
        self, ledger: "Ledger", subject: str, draft: GoalDraft, goal_id: str | None = None
    ) -> Checked:
        """Validate a new goal, or an edit of `goal_id`, without writing anything."""
        with self._connect() as conn:
            goals = self._replay(conn, ledger, subject)
        return _check(goals, draft, goal_id, ledger.as_of)

    # Writes

    def create(self, ledger: "Ledger", subject: str, draft: GoalDraft, *, source: str) -> Revision:
        return self._write_checked(ledger, subject, lambda _: draft, None, "create", source)

    def update(
        self, ledger: "Ledger", subject: str, goal_id: str, changes: dict[str, Any], *, source: str
    ) -> Revision:
        """Change any of `name`, `target_amount`, `target_date`, `saved`; the rest stay. A new
        saved amount is recorded as of today. The fields left out come from the goal as it is
        inside the write lock, so two edits at once never undo each other (review on #42)."""
        _check_changes(changes)

        def draft(goals: list[Goal]) -> GoalDraft:
            return edited_draft(_find(goals, goal_id), changes)

        return self._write_checked(ledger, subject, draft, goal_id, "update", source)

    def archive(self, ledger: "Ledger", subject: str, goal_id: str, *, source: str) -> Revision:
        _check_source(source)
        with self._write, self._connect() as conn:
            goal = _find(self._replay(conn, ledger, subject), goal_id)
            return self._insert(conn, subject, ledger.user_id, "archive", goal, source)

    def undo(self, ledger: "Ledger", subject: str, revision_id: str) -> Goal | None:
        """Undo the subject's latest live change to a goal. Returns the goal as it is now, or None
        when undoing its creation removed it. Refused when the result would break a rule that
        depends on other goals (a duplicate name, an 11th active goal); nothing changes then."""
        today = ledger.as_of
        with self._write, self._connect() as conn:
            events = self._select(conn, "subject = ? AND user_id = ?", (subject, ledger.user_id))
            event = next((e for e in events if e.revision_id == revision_id), None)
            if event is None:
                raise GoalError(f"no change {revision_id!r}")
            if event.undone:
                raise GoalError("that change is already undone")
            latest = max(
                (e for e in events if e.goal_id == event.goal_id and not e.undone),
                key=lambda e: e.seq,
            )
            if latest.revision_id != revision_id:
                raise GoalError("only the latest change to a goal can be undone")
            undone_at = _now()
            after = replay(
                generated_goals(ledger.goals),
                [replace(e, undone_at=undone_at) if e is event else e for e in events],
            )
            restored = next((g for g in after if g.goal_id == event.goal_id), None)
            if restored is not None and restored.running(today):
                others = [g for g in after if g.goal_id != restored.goal_id and not g.archived]
                problems = _name_taken(restored.name, others, today)
                if restored.status(today) == "active":
                    problems += _too_many(others, today)
                if problems:
                    raise GoalError(problems[0].message, tuple(problems))
            conn.execute(
                "UPDATE goal_revisions SET undone_at = ? WHERE revision_id = ?",
                (undone_at, revision_id),
            )
        return restored

    # Internals

    def _write_checked(
        self,
        ledger: "Ledger",
        subject: str,
        draft: Callable[[list[Goal]], GoalDraft],
        goal_id: str | None,
        op: str,
        source: str,
    ) -> Revision:
        """Validate and record a change. `draft` makes it from the goals as they are inside the
        lock, so it's checked and stored against the same state."""
        _check_source(source)
        with self._write, self._connect() as conn:
            goals = self._replay(conn, ledger, subject)
            checked = _check(goals, draft(goals), goal_id, ledger.as_of)
            if checked.goal is None:
                raise GoalError(checked.problems[0].message, checked.problems)
            return self._insert(conn, subject, ledger.user_id, op, checked.goal, source)

    def _replay(self, conn: sqlite3.Connection, ledger: "Ledger", subject: str) -> list[Goal]:
        events = self._select(conn, "subject = ? AND user_id = ?", (subject, ledger.user_id))
        return replay(generated_goals(ledger.goals), events)

    @staticmethod
    def _insert(
        conn: sqlite3.Connection, subject: str, user_id: str, op: str, goal: Goal, source: str
    ) -> Revision:
        if not subject or not user_id:
            raise GoalError("a goal change needs a subject and a user")
        (seq,) = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM goal_revisions").fetchone()
        rev = Revision(
            revision_id=uuid.uuid4().hex,
            seq=seq,
            subject=subject,
            user_id=user_id,
            goal_id=goal.goal_id,
            op=op,
            name=goal.name,
            target_amount_cents=goal.target_cents,
            target_date=goal.target_date.isoformat(),
            saved_cents=goal.saved_cents,
            saved_as_of=goal.saved_as_of.isoformat(),
            created_date=goal.created_date.isoformat(),
            source=source,
            created_at=_now(),
            undone_at=None,
        )
        conn.execute(
            f"INSERT INTO goal_revisions ({', '.join(_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in _COLUMNS)})",
            tuple(getattr(rev, c) for c in _COLUMNS),
        )
        return rev

    @staticmethod
    def _select(conn: sqlite3.Connection, where: str, params: tuple[str, ...]) -> list[Revision]:
        sql = f"SELECT {', '.join(_COLUMNS)} FROM goal_revisions WHERE {where}"
        return [Revision(*row) for row in conn.execute(sql, params).fetchall()]


def _check_source(source: str) -> None:
    if source not in SOURCES:
        raise GoalError(f"unknown source {source!r}")


def _find(goals: list[Goal], goal_id: str) -> Goal:
    found = next((g for g in goals if g.goal_id == goal_id and not g.archived), None)
    if found is None:
        raise GoalError(f"no goal {goal_id!r}")
    return found


def _check(goals: list[Goal], draft: GoalDraft, goal_id: str | None, today: date) -> Checked:
    editing = _find(goals, goal_id) if goal_id is not None else None
    others = [g for g in goals if g.goal_id != goal_id and not g.archived]
    return check_draft(draft, today, others, editing=editing)


def _name_taken(name: str, others: list[Goal], today: date) -> list[Problem]:
    """Names are unique among running goals (active or reached), ignoring case."""
    if any(g.running(today) and g.name.casefold() == name.casefold() for g in others):
        return [Problem("name", "name_in_use", f"You already have a goal called {name}.")]
    return []


def _too_many(others: list[Goal], today: date) -> list[Problem]:
    """For a goal that will be active: at most ACTIVE_MAX active goals, counting it."""
    if sum(g.status(today) == "active" for g in others) >= ACTIVE_MAX:
        message = f"You have {ACTIVE_MAX} active goals; finish or remove one first."
        return [Problem(None, "too_many_goals", message)]
    return []
