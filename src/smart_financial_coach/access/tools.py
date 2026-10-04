"""The tools the dashboard and the coach call (Technical Design, "Tools exposed to assistants").

No tool takes a `user_id`: a `Tools` is built around one user's `Ledger`, which the caller makes
from the signed-in session, so a tool argument can't name whose data is read. Every result is
JSON with amounts in the user's currency, so the coach quotes numbers without doing arithmetic.
Services that aren't released yet (spending spikes, forecasts, and unusual charges until an FR-7
model is promoted) return a typed "not available yet", never a number (FR-14, Delivery Plan sync
rule).

For the Oct 6 demo the tools run inside the web app; the tool server (Delivery Plan P1) will serve
the same functions over HTTP and MCP.

    tools = Tools(Ledger.load(sources, session_user_id))
    result = tools.call("get_spending_summary", {"start_date": "2026-09-01", ...})
    result.data, result.source

Review and corrections (FR-5, FR-6): with a `Feedback` (the store, whose feedback this is, and
where calls come from), the tools read the ledger as that subject sees it, with their overrides
applied, and `list_review_items`, `resolve_review_item`, `correct_category`, `undo_correction`
and `list_corrections` read and write their feedback. A change the coach makes to more than one
transaction is previewed, not applied, until the call says `confirm` (#15 \u00a73).

Savings goals (FR-10): with a `GoalAccess` (the goal store, whose goals these are, and where calls
come from), `list_goals` shows the subject's goals and `create_goal`, `update_goal`,
`archive_goal` and `undo_goal_change` change them. `check_goal` validates a goal and states the
facts for the setup screen (FR-10 design, "The setup check") without writing. Writes from the coach
or an outside assistant are previewed, not applied, until the call says `confirm`; the Goals page
applies on submit. Without a `GoalAccess` the goal tools are read-only.

Goal forecasts (FR-11, FR-12): once a goal-forecasting model is promoted, `forecast_goal` says
whether a goal is on track, `check_goal` adds the same forecast and a fit badge for a draft, and
`list_goals` each running goal's status. A user's running goals and a draft are forecast together,
as one set over one future, so a draft's numbers are the saved goal's (design §3, §7).
"""

import json
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Protocol, TypeVar

import pandas as pd

from smart_financial_coach.access.feedback import Correction, FeedbackError, FeedbackStore
from smart_financial_coach.access.goal_forecasts import (
    FITS,
    first_entries,
    forecast_fields,
    goal_rows,
    thin,
)
from smart_financial_coach.access.goals import (
    Checked,
    Goal,
    GoalDraft,
    GoalError,
    GoalStore,
    Problem,
    check_draft,
    edited_draft,
    generated_goals,
    median_monthly_savings,
    replay,
)
from smart_financial_coach.access.ledger import INCOME, Ledger
from smart_financial_coach.access.review_items import item_id, open_review_items
from smart_financial_coach.intelligence.anomaly.reasons import KIND_LABELS, reason
from smart_financial_coach.intelligence.forecasting.contract import history_json
from smart_financial_coach.intelligence.forecasting.savings import monthly_net

CURRENCY = "USD"
DASH = "\u2013"  # en dash, for date ranges
MAX_TRANSACTIONS = 50
MAX_ITEMS = 25
LOW_CONFIDENCE = 0.5  # below this a review item's confidence band is "low", else "medium"

ToolSpec = dict[str, Any]
T = TypeVar("T")

_DATE = {"type": "string", "format": "date", "description": "YYYY-MM-DD, inclusive"}
_GOAL_ID = {"type": "string", "description": "a goal_id from list_goals"}
_GOAL_FIELDS = {
    "name": {"type": "string", "description": "what the user is saving for, up to 40 characters"},
    "target_amount": {"type": "number", "description": "US dollars, to the cent"},
    "target_date": {
        "type": "string",
        "format": "date",
        "description": "YYYY-MM-DD; the goal is due at the end of that month",
    },
    "saved": {"type": "number", "description": "US dollars saved toward it so far"},
}
_CONFIRM = {"type": "boolean", "description": "the user agreed to the preview"}
_ASK_FIRST = (
    "Only when the user asked. From an assistant, returns a preview (status "
    "`needs_confirmation`) and changes nothing until called again with `confirm: true` after the "
    "user agrees. A change that breaks a rule returns status `invalid` with the problems; tell "
    "the user their messages."
)
TOOL_SPECS: list[ToolSpec] = [
    {
        "name": "get_spending_summary",
        "description": (
            "Spending and income for the signed-in user over a date range: totals, spending by "
            "category, and spending and income by calendar month. Spending excludes Income; "
            "refunds reduce their category's spending. `unreviewed_spend` is spending whose "
            "category the model is unsure about and the user hasn't confirmed; mention it when "
            "it affects an answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"start_date": _DATE, "end_date": _DATE},
            "required": ["start_date", "end_date"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_transactions",
        "description": (
            "The signed-in user's categorized transactions in a date range, optionally filtered "
            "by category or merchant text, newest or largest first. Returns the match count and "
            f"total as well as up to {MAX_TRANSACTIONS} rows. Amounts are negative for money out. "
            "Each row has `needs_review` and `review_reason` (`new_merchant` or "
            "`low_confidence`), from the categorization model's review policy: whether its "
            "category is one the model is unsure about."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": _DATE,
                "end_date": _DATE,
                "category": {"type": "string", "description": "one of the 13 categories"},
                "search": {"type": "string", "description": "merchant text to match"},
                "sort": {"type": "string", "enum": ["newest", "largest"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_TRANSACTIONS},
            },
            "required": ["start_date", "end_date"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_review_items",
        "description": (
            "Categories the model is unsure about, for the signed-in user to confirm or correct: "
            "one item per merchant, most unreviewed spending first, with the suggested category, "
            "why it's flagged (`new_merchant` or `low_confidence`), and how many transactions and "
            "how much spending it covers. Also the total open items and unreviewed spending."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": MAX_ITEMS}},
            "additionalProperties": False,
        },
    },
    {
        "name": "resolve_review_item",
        "description": (
            "Settle a review item for every transaction at its merchant: `confirm` keeps the "
            "suggested category, `correct` moves them to `category`. Only when the user asked. "
            "A change to more than one transaction returns a preview (status "
            "`needs_confirmation`) until called again with `confirm: true` after the user agrees."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item_id": {"type": "string"},
                "action": {"type": "string", "enum": ["confirm", "correct"]},
                "category": {"type": "string", "description": "the new category, for correct"},
                "confirm": {"type": "boolean", "description": "the user agreed to the preview"},
            },
            "required": ["item_id", "action"],
            "additionalProperties": False,
        },
    },
    {
        "name": "correct_category",
        "description": (
            "Change a transaction's category, flagged or not: for every transaction at its "
            "merchant (scope `merchant`, the default) or just this one (`transaction`). Only when "
            "the user asked. A change to more than one transaction returns a preview (status "
            "`needs_confirmation`) until called again with `confirm: true` after the user agrees."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {"type": "string"},
                "category": {"type": "string", "description": "one of the 13 categories"},
                "scope": {"type": "string", "enum": ["merchant", "transaction"]},
                "confirm": {"type": "boolean", "description": "the user agreed to the preview"},
            },
            "required": ["transaction_id", "category"],
            "additionalProperties": False,
        },
    },
    {
        "name": "undo_correction",
        "description": (
            "Undo one of the signed-in user's corrections or confirmations: the categories return "
            "to what they were before it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"correction_id": {"type": "string"}},
            "required": ["correction_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_corrections",
        "description": (
            "The signed-in user's recent corrections and confirmations, newest first, with what "
            "changed and whether each can still be undone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": MAX_ITEMS}},
            "additionalProperties": False,
        },
    },
    {
        "name": "list_goals",
        "description": (
            "The signed-in user's savings goals: name, target amount and date (the end of the "
            "month it's due), the amount saved and the day it was recorded, and `status`: "
            "`active`, `reached` (saved the whole amount before the date) or `ended` (the date has "
            "passed; the outcome isn't known). Active goals have `months_left` and "
            "`needed_per_month`. `undo_revision_id` is the goal's latest change that can be "
            "undone. Also the user's median monthly savings over the last 12 full months. Active "
            "and reached goals have `forecast_status` (`on_track`, `either_way`, `off_track` or "
            "`reached`), `p_goal_met`, the chance of reaching it by its date, and "
            "`projected_balance`, the likely amount by then (both null once reached); call "
            "forecast_goal for the range and what would close the gap."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "include_ended": {"type": "boolean", "description": "default true"},
                "include_archived": {
                    "type": "boolean",
                    "description": "also removed goals, to undo a removal; default false",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "check_goal",
        "description": (
            "Check a savings goal before creating or editing it; saves nothing. Returns any "
            "problems, each with a message for the user, and when it's valid: the date it's due "
            "(the end of its month), the months left, the amount needed each month, what the "
            "user's other active goals need each month, and their median monthly savings. A valid "
            "goal also gets `forecast` (as forecast_goal returns it) and `fit`: `within_reach`, "
            "`either_way` or `stretch`, from the chance of making it alongside the user's other "
            "goals. Call it before suggesting or creating a goal and quote its numbers; never work "
            "out a monthly amount or a chance yourself. For an edit, pass `goal_id` and only the "
            "fields that change."
        ),
        "input_schema": {
            "type": "object",
            "properties": {**_GOAL_FIELDS, "goal_id": _GOAL_ID},
            "additionalProperties": False,
        },
    },
    {
        "name": "create_goal",
        "description": f"Create a savings goal; `saved` defaults to 0. {_ASK_FIRST}",
        "input_schema": {
            "type": "object",
            "properties": {**_GOAL_FIELDS, "confirm": _CONFIRM},
            "required": ["name", "target_amount", "target_date"],
            "additionalProperties": False,
        },
    },
    {
        "name": "update_goal",
        "description": (
            "Change a savings goal's name, target amount, date or the amount saved so far. "
            "Fields left out stay as they are; a new saved amount is recorded as of today. "
            f"{_ASK_FIRST}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {"goal_id": _GOAL_ID, **_GOAL_FIELDS, "confirm": _CONFIRM},
            "required": ["goal_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "archive_goal",
        "description": (
            "Remove a savings goal from the user's goals. Nothing is deleted, and it can be "
            f"undone. {_ASK_FIRST}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {"goal_id": _GOAL_ID, "confirm": _CONFIRM},
            "required": ["goal_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "undo_goal_change",
        "description": (
            "Undo the latest change to a savings goal (creating, editing or removing it), by the "
            "`revision_id` a change returned or a goal's `undo_revision_id` from list_goals. Only "
            "when the user asks to undo. Returns the goal as it is now, or `removed: true` when "
            "undoing its creation; an undo that would break a rule returns status `invalid`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"revision_id": {"type": "string"}},
            "required": ["revision_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "detect_anomalies",
        "description": (
            "Unusual charges in a date range, each with its kind, a plain-language reason and the "
            "numbers behind it, and spending spikes (not available yet)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"start_date": _DATE, "end_date": _DATE},
            "required": ["start_date", "end_date"],
            "additionalProperties": False,
        },
    },
    {
        "name": "forecast_goal",
        "description": (
            "Whether the user is on track for an active or reached savings goal, from simulated "
            "futures of their own monthly savings. `status`: `on_track` (a 70%+ chance), "
            "`either_way`, `off_track` (under 30%) or `reached`. `p_goal_met` is the chance; "
            "`projected_balance` the likely amount by the date, `range` the 80% range around it; "
            "`gap` how far the likely amount falls short; `extra_per_month` the monthly amount "
            "that would put it on track (null when it already is). `share_source` says how much "
            "of their savings the goal is assumed to get: `track_record` (its own history), "
            "`your_entries` (their saved amounts over time) or `typical` (a new goal: a typical "
            "share). `short_history` means fewer than 6 months of history. A reached goal has no "
            "chance, range, gap or top-up; `may_draw_down` says months of spending more than "
            "they earn could take it back below the target. The forecast assumes such months "
            "draw on what's set aside."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"goal_id": _GOAL_ID},
            "required": ["goal_id"],
            "additionalProperties": False,
        },
    },
]
_SPECS = {spec["name"]: spec for spec in TOOL_SPECS}


class ToolError(ValueError):
    """A tool call the caller can fix (bad dates, unknown category): reported, not raised on."""


class GoalProblemsError(ToolError):
    """A goal change that breaks a rule: the problems, each for a form field, with its message."""

    def __init__(self, problems: tuple[Problem, ...]) -> None:
        super().__init__(" ".join(p.message for p in problems))
        self.problems = problems


class ToolsUnavailableError(Exception):
    """The tools can't be reached at all (the MCP server refused or failed): no answer possible."""


def _check_type(tool: str, key: str, value: Any, prop: dict[str, Any]) -> None:
    """Arguments come from the model, so check their JSON types before any code uses them."""
    kind = prop["type"]
    if kind == "string":
        ok = isinstance(value, str)
    elif kind == "integer":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "boolean":
        ok = isinstance(value, bool)
    elif kind == "number":
        ok = isinstance(value, int | float) and not isinstance(value, bool)
    else:
        ok = True
    if not ok:
        article = "an" if kind[0] in "aeiou" else "a"
        raise ToolError(f"{tool}: {key} must be {article} {kind}, not {type(value).__name__}")
    if "enum" in prop and value not in prop["enum"]:
        raise ToolError(f"{tool}: {key} must be one of {prop['enum']}")


@dataclass(frozen=True)
class Source:
    """Where a result's numbers came from, shown next to coach answers (FR-16)."""

    title: str
    detail: str


@dataclass(frozen=True)
class ToolResult:
    data: dict[str, Any]
    source: Source


def money(value: float) -> float:
    return round(float(value), 2)


def span_label(start: date, end: date) -> str:
    if start.year == end.year and start.month == end.month:
        if start.day == 1 and (end + timedelta(days=1)).day == 1:
            return start.strftime("%b %Y")
        return f"{start:%b} {start.day}{DASH}{end.day}, {end.year}"
    if start.year == end.year:
        return f"{start:%b} {start.day} {DASH} {end:%b} {end.day}, {end.year}"
    return f"{start:%b} {start.day}, {start.year} {DASH} {end:%b} {end.day}, {end.year}"


def _one_per_pair(flags: pd.DataFrame) -> pd.DataFrame:
    """Flags with each same-minute duplicate pair shown once (FR-7 §8, review on #33).

    Two identical charges in one minute are each other's repeat, since their order is unknowable,
    so both can be flagged. The user sees one "Possible duplicate", on the charge with the later
    ID; the stored flags keep both."""
    duplicates = flags[flags["reason_code"] == "duplicate"]
    if duplicates.empty:  # an empty .map is string-typed under pandas 3, and & would raise
        return flags
    evidence = [json.loads(e) for e in duplicates["evidence"]]
    ids = duplicates["transaction_id"].to_numpy()
    flagged = set(ids)
    hidden = {
        t
        for t, e in zip(ids, evidence, strict=True)
        if e["minutes_apart"] == 0
        and e["original_transaction_id"] in flagged
        and t < e["original_transaction_id"]
    }
    return flags[~flags["transaction_id"].isin(hidden)]


class ToolGateway(Protocol):
    """What the coach needs from its tools: `Tools` in-process, or `McpTools` over MCP."""

    @property
    def specs(self) -> list[ToolSpec]: ...

    @property
    def as_of(self) -> date: ...

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...


@dataclass(frozen=True)
class Feedback:
    """Whose feedback the tools read and write, and where calls come from (FR-5, FR-6).

    `subject` comes from the session or the bearer token, never from a tool argument: the
    signed-in user in production, the browser session in the demo. `source` is "edit" for the
    dashboard and "coach" for assistants, whose multi-transaction changes need `confirm`.
    """

    store: FeedbackStore
    subject: str
    source: str = "edit"


@dataclass(frozen=True)
class GoalAccess:
    """Whose goals the tools read and write, and where calls come from (FR-10).

    `subject` comes from the session or the bearer token, as for `Feedback`. `source` is "edit"
    for the Goals page, which applies on submit, and "coach" (Wren) or "assistant" (any other MCP
    client), whose writes are previewed until they say `confirm`.
    """

    store: GoalStore
    subject: str
    source: str = "edit"


class Tools:
    def __init__(
        self, ledger: Ledger, feedback: Feedback | None = None, goals: GoalAccess | None = None
    ) -> None:
        self.base = ledger  # the model's categories
        self.feedback = feedback
        self.goals = goals
        self.ledger = ledger.seen_by(feedback.store, feedback.subject) if feedback else ledger
        self._history: str | None = None  # the user's monthly net savings, for goal forecasts
        known = feedback.store.categories if feedback else set(ledger.transactions["category"])
        self.categories = sorted(known)
        self._handlers: dict[str, Callable[..., ToolResult]] = {
            "get_spending_summary": self.get_spending_summary,
            "get_transactions": self.get_transactions,
            "list_review_items": self.list_review_items,
            "resolve_review_item": self.resolve_review_item,
            "correct_category": self.correct_category,
            "undo_correction": self.undo_correction,
            "list_corrections": self.list_corrections,
            "list_goals": self.list_goals,
            "check_goal": self.check_goal,
            "create_goal": self.create_goal,
            "update_goal": self.update_goal,
            "archive_goal": self.archive_goal,
            "undo_goal_change": self.undo_goal_change,
            "detect_anomalies": self.detect_anomalies,
            "forecast_goal": self.forecast_goal,
        }

    @property
    def specs(self) -> list[ToolSpec]:
        return TOOL_SPECS

    @property
    def as_of(self) -> date:
        return self.ledger.as_of

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run a tool by name with arguments checked against its schema."""
        if name not in _SPECS:
            raise ToolError(f"unknown tool {name!r}")
        schema = _SPECS[name]["input_schema"]
        unknown = set(arguments) - set(schema["properties"])
        if unknown:  # including any attempt to pass a user_id
            raise ToolError(f"{name} takes no {', '.join(sorted(unknown))} argument")
        missing = set(schema.get("required", [])) - set(arguments)
        if missing:
            raise ToolError(f"{name} needs {', '.join(sorted(missing))}")
        for key, value in arguments.items():
            _check_type(name, key, value, schema["properties"][key])
        return self._handlers[name](**arguments)

    # Tools

    def get_spending_summary(self, start_date: str, end_date: str) -> ToolResult:
        start, end = self._range(start_date, end_date)
        rows = self.ledger.between(start, end)
        income_rows = rows[rows["category"] == INCOME]
        unsure = rows[rows["needs_review"]]
        spend_rows = rows[rows["category"] != INCOME]
        by_category = (
            spend_rows.assign(spent=-spend_rows["amount"])
            .groupby("category")
            .agg(amount=("spent", "sum"), transactions=("spent", "count"))
            .sort_values("amount", ascending=False)
        )
        months = rows.assign(month=rows["ts"].dt.strftime("%Y-%m"))
        spending_by_month = -months[months["category"] != INCOME].groupby("month")["amount"].sum()
        income_by_month = months[months["category"] == INCOME].groupby("month")["amount"].sum()
        month_keys = pd.period_range(start, end, freq="M").strftime("%Y-%m")
        spending = -spend_rows["amount"].sum()
        income = income_rows["amount"].sum()
        data = {
            "currency": CURRENCY,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "income": money(income),
            "spending": money(spending),
            "net": money(income - spending),
            "transactions": len(rows),
            # How firm these numbers are: spending in categories the model is unsure about and
            # the user hasn't confirmed or corrected (FR-5; #15 open question 5)
            "unreviewed_spend": money(-unsure["amount"].clip(upper=0).sum()),
            "open_review_items": int(unsure["merchant_key"].nunique()),
            "by_category": [
                {"category": c, "amount": money(r.amount), "transactions": int(r.transactions)}
                for c, r in by_category.iterrows()
            ],
            "by_month": [
                {
                    "month": m,
                    "spending": money(spending_by_month.get(m, 0.0)),
                    "income": money(income_by_month.get(m, 0.0)),
                }
                for m in month_keys
            ],
        }
        return ToolResult(
            data,
            Source(f"Spending summary · {span_label(start, end)}", f"{len(rows)} transactions"),
        )

    def get_transactions(
        self,
        start_date: str,
        end_date: str,
        category: str | None = None,
        search: str | None = None,
        sort: str = "newest",
        limit: int = 25,
    ) -> ToolResult:
        start, end = self._range(start_date, end_date)
        rows = self.ledger.between(start, end)
        if category is not None:
            if category not in self.categories:
                raise ToolError(f"unknown category {category!r}; use one of {self.categories}")
            rows = rows[rows["category"] == category]
        if search:
            text = search.strip()
            match = rows["merchant"].str.contains(text, case=False, regex=False) | rows[
                "merchant_raw"
            ].str.contains(text, case=False, regex=False)
            rows = rows[match]
        if sort not in ("newest", "largest"):
            raise ToolError(f"sort must be newest or largest, not {sort!r}")
        if sort == "largest":
            rows = rows.reindex(rows["amount"].abs().sort_values(ascending=False).index)
        limit = max(1, min(int(limit), MAX_TRANSACTIONS))
        shown = rows.head(limit)
        what = " · ".join(x for x in (category, f'"{search}"' if search else None) if x)
        title = f"Transactions{' · ' + what if what else ''} · {span_label(start, end)}"
        data = {
            "currency": CURRENCY,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "matched": len(rows),
            "total": money(rows["amount"].sum()),
            "shown": len(shown),
            "transactions": [
                {
                    "transaction_id": r["transaction_id"],
                    "date": r["day"].isoformat(),
                    "merchant": r["merchant"],
                    "description": r["merchant_raw"],
                    "amount": money(r["amount"]),
                    "category": r["category"],
                    "confidence": round(float(r["confidence"]), 2),
                    "category_source": r.get("category_source", "model"),
                    "needs_review": bool(r["needs_review"]),
                    "review_reason": (r["review_reason"] or None) if r["needs_review"] else None,
                    "review_item_id": (
                        item_id(self.ledger.user_id, r["merchant_key"])
                        if r["needs_review"]
                        else None
                    ),
                }
                for r in shown.to_dict("records")
            ],
        }
        return ToolResult(data, Source(title, f"{len(rows)} matched"))

    # Review and corrections (FR-5, FR-6)

    def list_review_items(self, limit: int = 10) -> ToolResult:
        items = open_review_items(self.ledger.transactions, self.ledger.user_id)
        data = {
            "currency": CURRENCY,
            "open_items": len(items),
            "unreviewed_spend": money(items["unreviewed_spend"].sum()) if len(items) else 0.0,
            "items": [self._item(r) for r in items.head(limit).to_dict("records")],
        }
        return ToolResult(data, Source("Categories to review", f"{len(items)} open"))

    def resolve_review_item(
        self, item_id: str, action: str, category: str | None = None, confirm: bool = False
    ) -> ToolResult:
        items = open_review_items(self.ledger.transactions, self.ledger.user_id)
        match = items[items["item_id"] == item_id]
        if match.empty:
            raise ToolError(f"no open review item {item_id!r}")
        item = match.iloc[0]
        suggested = str(item["suggested_category"])
        if action == "confirm":
            if category not in (None, suggested):
                raise ToolError("confirm keeps the suggested category; use correct to change it")
            target = suggested
        elif category is None:
            raise ToolError("correct needs the new category")
        else:
            target = category
        return self._apply(
            action=action,
            scope="merchant",
            merchant_key=str(item["merchant_key"]),
            transaction_id=None,
            shown=suggested,
            target=target,
            via="review",
            confirm=confirm,
        )

    def correct_category(
        self, transaction_id: str, category: str, scope: str = "merchant", confirm: bool = False
    ) -> ToolResult:
        rows = self.ledger.transactions
        match = rows[rows["transaction_id"] == transaction_id]
        if match.empty:
            raise ToolError(f"no transaction {transaction_id!r}")
        row = match.iloc[0]
        if scope not in ("merchant", "transaction"):
            raise ToolError(f"scope must be merchant or transaction, not {scope!r}")
        if row.get("model_category", row["category"]) == INCOME:
            # Merchant overrides leave Income rows alone (review on #35): change just this one
            scope = "transaction"
        shown = str(row["category"])
        if scope == "merchant":
            # Record what the merchant's rows held, not the clicked row's own category (review
            # on #35): the rows that will move, by their most common current category
            at = rows[rows["merchant_key"] == row["merchant_key"]]
            spending = at[at.get("model_category", at["category"]) != INCOME]
            moving = spending.loc[spending["category"] != category, "category"]
            if moving.empty:
                raise ToolError(f"every transaction at {row['merchant']} is already {category}")
            shown = str(moving.mode().sort_values().iloc[0])
        return self._apply(
            action="correct",
            scope=scope,
            merchant_key=str(row["merchant_key"]),
            transaction_id=transaction_id if scope == "transaction" else None,
            shown=shown,
            target=category,
            via="edit",
            confirm=confirm,
        )

    def undo_correction(self, correction_id: str) -> ToolResult:
        feedback = self._writable()
        before = self.ledger.transactions
        try:
            undone = feedback.store.undo(feedback.subject, correction_id)
        except FeedbackError as error:
            raise ToolError(str(error)) from error
        self._refresh()
        effect = self._effect(before, self._covered(undone.merchant_key, undone.transaction_id))
        data = {"currency": CURRENCY, "undone": self._correction(undone), **effect}
        detail = f"{effect['transactions_changed']} transactions restored"
        return ToolResult(data, Source("Undid a category change", detail))

    def list_corrections(self, limit: int = 10) -> ToolResult:
        feedback = self._writable()
        events = feedback.store.corrections(feedback.subject, self.ledger.user_id, limit)
        data = {"corrections": [self._correction(e) for e in events]}
        return ToolResult(data, Source("Your category changes", f"{len(events)} shown"))

    def _apply(
        self,
        *,
        action: str,
        scope: str,
        merchant_key: str,
        transaction_id: str | None,
        shown: str,
        target: str,
        via: str,
        confirm: bool,
    ) -> ToolResult:
        """Record a confirmation or correction, or preview it for the coach (#15 §3)."""
        feedback = self._writable()
        if target not in self.categories:
            raise ToolError(f"unknown category {target!r}; use one of {self.categories}")
        covered = self._covered(merchant_key, transaction_id)
        rows = self.ledger.transactions[covered]
        name = str(rows["merchant"].iloc[0])
        if transaction_id is None:  # a merchant override leaves Income rows alone (review on #36)
            rows = rows[rows["model_category"] != INCOME]
            if rows.empty:
                raise ToolError(f"{name} has only income; change its transactions one at a time")
        if feedback.source == "coach" and len(rows) > 1 and not confirm:
            moving = rows[rows["category"] != target]
            data = {
                "status": "needs_confirmation",
                "currency": CURRENCY,
                "merchant": name,
                "action": action,
                "category": target,
                "transactions": len(rows),
                "transactions_changing": len(moving),
                "spend_changing": money((-moving["amount"]).clip(lower=0).sum()),
                "message": (
                    f"This changes {len(rows)} transactions at {name}. Ask the user, and call "
                    "again with confirm: true only if they agree."
                ),
            }
            return ToolResult(data, Source(f"Preview · {name}", f"{len(rows)} transactions"))
        before = self.ledger.transactions
        try:
            recorded = feedback.store.record(
                feedback.subject,
                self.ledger.user_id,
                action=action,
                scope=scope,
                merchant_key=merchant_key,
                transaction_id=transaction_id,
                from_category=shown,
                to_category=target,
                source="coach" if feedback.source == "coach" else via,
                model_version=str(rows["model_version"].iloc[0]),
            )
        except FeedbackError as error:
            raise ToolError(str(error)) from error
        self._refresh()
        effect = self._effect(before, covered)
        data = {
            "status": "applied",
            "currency": CURRENCY,
            "merchant": name,
            "correction": self._correction(recorded),
            **effect,
        }
        verb = "Confirmed" if action == "confirm" else "Moved"
        title = f"{verb} {name} · {target}"
        return ToolResult(data, Source(title, f"{effect['transactions_changed']} transactions"))

    def _writable(self) -> Feedback:
        if self.feedback is None:
            raise ToolError("reviewing and correcting categories isn't available here")
        return self.feedback

    def _refresh(self) -> None:
        feedback = self._writable()
        self.ledger = self.base.seen_by(feedback.store, feedback.subject)

    def _covered(self, merchant_key: str, transaction_id: str | None) -> pd.Series:
        t = self.ledger.transactions
        if transaction_id is not None:
            return t["transaction_id"] == transaction_id
        return t["merchant_key"] == merchant_key

    def _effect(self, before: pd.DataFrame, covered: pd.Series) -> dict[str, Any]:
        """How many transactions changed category, and the money that moved between them."""
        old = before[covered].set_index("transaction_id")
        new = self.ledger.transactions[covered].set_index("transaction_id")
        moved = old.assign(to=new["category"])
        moved = moved[moved["category"] != moved["to"]]
        flows = (
            moved.assign(spend=-moved["amount"])
            .groupby(["category", "to"])["spend"]
            .agg(["sum", "count"])
            .reset_index()
        )
        return {
            "transactions_changed": len(moved),
            "spend_moved": [
                {
                    "from_category": r["category"],
                    "to_category": r["to"],
                    "amount": money(r["sum"]),
                    "transactions": int(r["count"]),
                }
                for r in flows.to_dict("records")
            ],
        }

    def _item(self, r: Mapping[Any, Any]) -> dict[str, Any]:
        return {
            "item_id": r["item_id"],
            "merchant": r["merchant"],
            "sample_description": r["sample_description"],
            "transaction_count": int(r["transaction_count"]),
            "flagged_count": int(r["flagged_count"]),
            "total_spend": money(r["total_spend"]),
            "unreviewed_spend": money(r["unreviewed_spend"]),
            "suggested_category": r["suggested_category"],
            "confidence_band": "low" if r["confidence"] < LOW_CONFIDENCE else "medium",
            "reason": r["reason"],
            "first_seen": r["first_seen"].date().isoformat(),
        }

    def _correction(self, c: Correction) -> dict[str, Any]:
        t = self.ledger.transactions
        names = t.loc[t["merchant_key"] == c.merchant_key, "merchant"]
        return {
            "correction_id": c.correction_id,
            "action": c.action,
            "scope": c.scope,
            "merchant": str(names.iloc[0]) if len(names) else c.merchant_key,
            "transaction_id": c.transaction_id,
            "from_category": c.from_category,
            "to_category": c.to_category,
            "source": c.source,
            "created_at": c.created_at,
            "undoable": not c.undone,
        }

    # Savings goals (FR-10)

    def list_goals(self, include_ended: bool = True, include_archived: bool = False) -> ToolResult:
        today = self.as_of
        goals = [
            g
            for g in self._goals(include_archived=include_archived)
            if include_ended or g.status(today) != "ended"
        ]
        forecasts = self._forecasts([g for g in self._goals() if g.running(today)])
        listed = []
        for g in goals:
            data = self._goal_data(g)
            if forecasts is not None and g.goal_id in forecasts:
                f = forecasts[g.goal_id]
                reached = f["status"] == "reached"
                data |= {
                    "forecast_status": f["status"],
                    "p_goal_met": None if reached else round(float(f["p_goal_met"]), 3),
                    "projected_balance": None if reached else money(f["projected_balance"]),
                }
            listed.append(data)
        data = {
            "currency": CURRENCY,
            "as_of": today.isoformat(),
            "goals": listed,
            **self._savings(),
            "forecast": "available" if forecasts is not None else "not_available",
        }
        return ToolResult(data, Source("Savings goals", f"{len(goals)} goals"))

    def check_goal(
        self,
        name: str | None = None,
        target_amount: float | None = None,
        target_date: str | None = None,
        saved: float | None = None,
        goal_id: str | None = None,
    ) -> ToolResult:
        fields = {"name": name, "target_amount": target_amount, "target_date": target_date}
        checked, others = self._check({**fields, "saved": saved}, goal_id)
        data = self._check_data(checked, others)
        if checked.goal is not None:
            title = f"Goal check · {checked.goal.name}"
            detail = f"${data['needed_per_month']:,.0f} a month"
        else:
            title, detail = "Goal check", f"{len(checked.problems)} problems"
        return ToolResult(data, Source(title, detail))

    def create_goal(
        self,
        name: str,
        target_amount: float,
        target_date: str,
        saved: float | None = None,
        confirm: bool = False,
    ) -> ToolResult:
        access = self._goal_writes()
        fields = {"name": name, "target_amount": target_amount, "target_date": target_date}
        checked, others = self._check({**fields, "saved": saved}, None)
        if checked.goal is None:
            raise GoalProblemsError(checked.problems)
        if self._needs_confirmation(confirm):
            return self._preview(checked, others, f"This creates the goal {checked.goal.name}.")
        draft = GoalDraft(name, target_amount, target_date, 0 if saved is None else saved)
        revision = self._goal_write(
            lambda: access.store.create(self.ledger, access.subject, draft, source=access.source)
        )
        return self._applied(revision.goal_id, revision.revision_id, f"Created {checked.goal.name}")

    def update_goal(
        self,
        goal_id: str,
        name: str | None = None,
        target_amount: float | None = None,
        target_date: str | None = None,
        saved: float | None = None,
        confirm: bool = False,
    ) -> ToolResult:
        access = self._goal_writes()
        changes = {
            k: v
            for k, v in {
                "name": name,
                "target_amount": target_amount,
                "target_date": target_date,
                "saved": saved,
            }.items()
            if v is not None
        }
        if not changes:
            raise ToolError(
                "update_goal needs at least one of name, target_amount, target_date, saved"
            )
        checked, others = self._check(changes, goal_id)
        if checked.goal is None:
            raise GoalProblemsError(checked.problems)
        if self._needs_confirmation(confirm):
            return self._preview(checked, others, f"This changes the goal {checked.goal.name}.")
        revision = self._goal_write(
            lambda: access.store.update(
                self.ledger, access.subject, goal_id, changes, source=access.source
            )
        )
        return self._applied(goal_id, revision.revision_id, f"Changed {checked.goal.name}")

    def archive_goal(self, goal_id: str, confirm: bool = False) -> ToolResult:
        access = self._goal_writes()
        goal = self._goal_by_id(goal_id, self._goals())
        if self._needs_confirmation(confirm):
            data = {
                "status": "needs_confirmation",
                "currency": CURRENCY,
                "goal": self._goal_data(goal),
                "message": (
                    f"This removes the goal {goal.name}. Ask the user, and call again with "
                    "confirm: true only if they agree."
                ),
            }
            return ToolResult(data, Source(f"Preview · {goal.name}", "remove"))
        revision = self._goal_write(
            lambda: access.store.archive(self.ledger, access.subject, goal_id, source=access.source)
        )
        archived = self._goal_by_id(goal_id, self._goals(include_archived=True))
        data = {
            "status": "applied",
            "currency": CURRENCY,
            "goal": self._goal_data(archived),
            "revision_id": revision.revision_id,
        }
        return ToolResult(data, Source(f"Removed {goal.name}", "can be undone"))

    def undo_goal_change(self, revision_id: str) -> ToolResult:
        access = self._goal_writes()
        goal = self._goal_write(lambda: access.store.undo(self.ledger, access.subject, revision_id))
        data = {
            "status": "applied",
            "currency": CURRENCY,
            "removed": goal is None,
            "goal": None if goal is None else self._goal_data(goal),
        }
        title = "Undid a goal change"
        detail = "the goal was removed" if goal is None else goal.name
        return ToolResult(data, Source(title, detail))

    def _goals(self, include_archived: bool = False) -> list[Goal]:
        """The user's goals as this subject sees them; just the generated ones without access."""
        if self.goals is None:
            return replay(generated_goals(self.ledger.goals), [])
        access = self.goals
        return access.store.goals(self.ledger, access.subject, include_archived=include_archived)

    @staticmethod
    def _goal_by_id(goal_id: str, goals: list[Goal]) -> Goal:
        found = next((g for g in goals if g.goal_id == goal_id), None)
        if found is None:
            raise ToolError(f"no goal {goal_id!r}; goal ids come from list_goals")
        return found

    def _check(self, fields: dict[str, Any], goal_id: str | None) -> tuple[Checked, list[Goal]]:
        """Validate a new goal, or an edit of `goal_id` with the fields given; and the other
        goals it was checked against."""
        goals = self._goals()
        if goal_id is None:
            saved = 0 if fields.get("saved") is None else fields["saved"]
            draft = GoalDraft(
                fields.get("name"), fields.get("target_amount"), fields.get("target_date"), saved
            )
            others, editing = goals, None
        else:
            editing = self._goal_by_id(goal_id, goals)
            draft = edited_draft(editing, {k: v for k, v in fields.items() if v is not None})
            others = [g for g in goals if g.goal_id != goal_id]
        return check_draft(draft, self.as_of, others, editing=editing), others

    def _check_data(self, checked: Checked, others: list[Goal]) -> dict[str, Any]:
        today = self.as_of
        data: dict[str, Any] = {
            "currency": CURRENCY,
            "as_of": today.isoformat(),
            "valid": checked.valid,
            "problems": [
                {"field": p.field, "code": p.code, "message": p.message} for p in checked.problems
            ],
        }
        goal = checked.goal
        if goal is not None:
            this = goal.needed_per_month_cents(today)
            other = sum(g.needed_per_month_cents(today) for g in others)
            data |= {
                "name": goal.name,
                "target_amount": goal.target_cents / 100,
                "target_date": goal.target_date.isoformat(),
                "saved": goal.saved_cents / 100,
                "months_left": goal.months_left(today),
                "needed_per_month": this / 100,
                "other_goals_per_month": other / 100,
                "all_goals_per_month": (this + other) / 100,
            }
            forecast = self._forecast_of(goal, others)
            if forecast is None:
                return data | self._savings() | {"forecast": "not_available"}
            fit = FITS.get(str(forecast["status"]))  # none for a goal already reached
            return data | self._savings() | {"forecast": forecast, "fit": fit}
        return data | self._savings() | {"forecast": None}  # nothing to forecast until it's valid

    def _savings(self) -> dict[str, Any]:
        median, months = median_monthly_savings(self.ledger.transactions, self.as_of)
        return {
            "median_monthly_savings_12m": None if median is None else median / 100,
            "months_of_history": months,
        }

    def _goal_data(self, g: Goal) -> dict[str, Any]:
        today = self.as_of
        status = g.status(today)
        active = status == "active"
        return {
            "goal_id": g.goal_id,
            "name": g.name,
            "target_amount": g.target_cents / 100,
            "target_date": g.target_date.isoformat(),
            "saved": g.saved_cents / 100,
            "saved_as_of": g.saved_as_of.isoformat(),
            "created_date": g.created_date.isoformat(),
            "status": status,
            "origin": g.origin,
            "months_left": g.months_left(today) if active else None,
            "needed_per_month": g.needed_per_month_cents(today) / 100 if active else None,
            "undo_revision_id": g.undo_revision_id,
        }

    def _goal_writes(self) -> GoalAccess:
        if self.goals is None:
            raise ToolError("changing goals isn't available for this sign-in")
        return self.goals

    def _needs_confirmation(self, confirm: bool) -> bool:
        """Wren and outside assistants preview every goal change until the user agrees."""
        return self._goal_writes().source != "edit" and not confirm

    def _preview(self, checked: Checked, others: list[Goal], what: str) -> ToolResult:
        data = self._check_data(checked, others) | {
            "status": "needs_confirmation",
            "message": (
                f"{what} Tell the user the date and monthly amount above, ask, and call again "
                "with confirm: true only if they agree."
            ),
        }
        name = checked.goal.name if checked.goal else "goal"
        return ToolResult(
            data, Source(f"Preview · {name}", f"${data['needed_per_month']:,.0f} a month")
        )

    def _goal_write(self, write: Callable[[], T]) -> T:
        try:
            return write()
        except GoalError as error:
            if error.problems:
                raise GoalProblemsError(error.problems) from error
            raise ToolError(str(error)) from error

    def _applied(self, goal_id: str, revision_id: str, title: str) -> ToolResult:
        goal = self._goal_by_id(goal_id, self._goals())
        data = {
            "status": "applied",
            "currency": CURRENCY,
            "goal": self._goal_data(goal),
            "revision_id": revision_id,
        }
        return ToolResult(data, Source(title, "can be undone"))

    def detect_anomalies(self, start_date: str, end_date: str) -> ToolResult:
        start, end = self._range(start_date, end_date)
        flags = self.ledger.flags
        if flags is None:
            return self._not_available(
                "Unusual-spending alerts", f"{span_label(start, end)}", "FR-7 and FR-8"
            )
        rows = self.ledger.between(start, end)
        rows = rows.merge(
            _one_per_pair(flags)[["transaction_id", "reason_code", "evidence"]], on="transaction_id"
        )
        rows = rows.assign(evidence=[self._with_category(r) for r in rows.to_dict("records")])
        data = {
            "currency": CURRENCY,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "count": len(rows),
            "unusual_transactions": [
                {
                    "transaction_id": r["transaction_id"],
                    "date": r["day"].isoformat(),
                    "merchant": r["merchant"],
                    "description": r["merchant_raw"],
                    "amount": money(r["amount"]),
                    "category": r["category"],
                    "kind": KIND_LABELS[r["reason_code"]],
                    "reason_code": r["reason_code"],
                    "reason": reason(r["reason_code"], r["evidence"]),
                    "evidence": json.loads(r["evidence"]),
                }
                for r in rows.to_dict("records")
            ],
            "spending_spikes": self._not_available(
                "Spending spikes", span_label(start, end), "FR-8"
            ).data,
        }
        title = f"Unusual charges · {span_label(start, end)}"
        return ToolResult(data, Source(title, f"{len(rows)} flagged"))

    def _with_category(self, flag: Mapping[Hashable, Any]) -> str:
        """A new-merchant flag's evidence with its predicted category and the latest earlier
        charge in that category at least as large, from the user's own ledger (FR-7 §7)."""
        evidence = json.loads(flag["evidence"])
        if flag["reason_code"] != "new_merchant":
            return str(flag["evidence"])
        t = self.ledger.transactions
        earlier = t[
            (t["category"] == flag["category"])
            & (t["ts"] < flag["ts"])
            & (t["amount"] <= flag["amount"])  # outflows: at least as large
        ]
        latest = earlier["ts"].max() if len(earlier) else None
        evidence |= {
            "category": flag["category"],
            "date": flag["day"].isoformat(),
            "category_largest_since": latest.strftime("%Y-%m-%d") if latest is not None else None,
        }
        return json.dumps(evidence)

    def forecast_goal(self, goal_id: str) -> ToolResult:
        goals = self._goals(include_archived=False)
        goal = self._goal_by_id(goal_id, goals)
        today = self.as_of
        if goal.status(today) == "ended":
            ended = {
                "status": "ended",
                "message": (
                    f"{goal.name}'s date has passed, so there's nothing to forecast; whether "
                    "it was reached isn't known."
                ),
            }
            return ToolResult(ended, Source(f"Goal forecast · {goal.name}", "ended"))
        forecast = self._forecast_of(goal, [g for g in goals if g.goal_id != goal_id], monthly=True)
        if forecast is None:
            return self._not_available("Goal forecast", goal.name, "FR-11 and FR-12")
        data: dict[str, Any] = {
            "currency": CURRENCY,
            "as_of": today.isoformat(),
            "goal_id": goal.goal_id,
            "name": goal.name,
            "target_amount": goal.target_cents / 100,
            "target_date": goal.target_date.isoformat(),
            "saved": goal.saved_cents / 100,
            "months_left": goal.months_left(today),
            **forecast,
        }
        labels = {
            "on_track": "on track",
            "either_way": "could go either way",
            "off_track": "off track",
            "reached": "reached",
        }
        return ToolResult(
            data, Source(f"Goal forecast · {goal.name}", labels[str(forecast["status"])])
        )

    def _forecasts(self, goals: list[Goal]) -> dict[str, dict[str, Any]] | None:
        """The forecast of one goal set, by goal id: the user's running goals, a draft
        included. None when there's no forecast for this user (no model promoted)."""
        forecaster = self.ledger.forecaster
        user = self.ledger.user_id
        if forecaster is None or forecaster.state(user) is None or forecaster.as_of != self.as_of:
            return None
        if not goals:
            return {}
        if self._history is None:
            columns = self.ledger.transactions[["ts", "amount"]]
            self._history = history_json(monthly_net(columns, self.as_of))
        entries = (
            first_entries(self.goals.store.revisions(self.goals.subject, user))
            if self.goals is not None
            else {}
        )
        out = forecaster.forecast(goal_rows(user, goals, self.as_of, self._history, entries))
        return {
            str(r["example_id"]): {str(k): v for k, v in r.items()} for r in out.to_dict("records")
        }

    def _forecast_of(
        self, goal: Goal, others: list[Goal], *, monthly: bool = False
    ) -> dict[str, Any] | None:
        """`goal`'s forecast alongside the user's other running goals (`others` may list ended
        or archived ones too), as the tools return it; None without a forecast. An ended goal,
        or a draft that's already ended, has none."""
        today = self.as_of
        if not goal.running(today):
            return None
        running = [g for g in others if g.running(today) and g.goal_id != goal.goal_id]
        forecasts = self._forecasts([*running, goal])
        if forecasts is None:
            return None
        out = forecasts[goal.goal_id]
        forecaster = self.ledger.forecaster
        assert forecaster is not None  # _forecasts returned forecasts
        state = forecaster.state(self.ledger.user_id)
        assert state is not None
        fields = forecast_fields(out, state.months, money)
        if monthly:
            series = (
                []
                if out["status"] == "reached"
                else forecaster.monthly(
                    self.ledger.user_id,
                    goal.saved_cents / 100,
                    float(out["share"]),
                    goal.months_left(today),
                )
            )
            fields["monthly"] = [
                {
                    "month": m["month"],
                    "median": money(m["median"]),
                    "low": money(m["low"]),
                    "high": money(m["high"]),
                }
                for m in thin(series)
            ]
        return fields

    # Helpers

    def _range(self, start_date: str, end_date: str) -> tuple[date, date]:
        try:
            start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        except (TypeError, ValueError) as error:
            raise ToolError(f"dates must be YYYY-MM-DD: {error}") from error
        if end < start:
            raise ToolError(f"end_date {end} is before start_date {start}")
        if start > self.ledger.as_of:
            raise ToolError(f"no data after {self.ledger.as_of}, the latest available day")
        return start, min(end, self.ledger.as_of)

    @staticmethod
    def _not_available(title: str, detail: str, requirement: str) -> ToolResult:
        data = {
            "status": "not_available",
            "message": (
                f"{title} isn't available yet: the model behind it ({requirement}) hasn't been "
                "released. Don't estimate it."
            ),
        }
        return ToolResult(data, Source(f"{title} · not available yet", detail))
