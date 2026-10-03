"""The tools the dashboard and the coach call (Technical Design, "Tools exposed to assistants").

No tool takes a `user_id`: a `Tools` is built around one user's `Ledger`, which the caller makes
from the signed-in session, so a tool argument can't name whose data is read. Every result is
JSON with amounts in the user's currency, so the coach quotes numbers without doing arithmetic.
Services that aren't built yet (anomalies, forecasts) return a typed "not available yet", never a
number (FR-14, Delivery Plan sync rule).

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
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Protocol

import pandas as pd

from smart_financial_coach.access.feedback import Correction, FeedbackError, FeedbackStore
from smart_financial_coach.access.ledger import INCOME, Ledger
from smart_financial_coach.access.review_items import item_id, open_review_items

CURRENCY = "USD"
DASH = "\u2013"  # en dash, for date ranges
MAX_TRANSACTIONS = 50
MAX_ITEMS = 25
LOW_CONFIDENCE = 0.5  # below this a review item's confidence band is "low", else "medium"

ToolSpec = dict[str, Any]

_DATE = {"type": "string", "format": "date", "description": "YYYY-MM-DD, inclusive"}
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
            "The signed-in user's savings goals: name, target amount and date, and the saved "
            "balance on the date it was last recorded."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "detect_anomalies",
        "description": "Unusual transactions and spending spikes in a date range.",
        "input_schema": {
            "type": "object",
            "properties": {"start_date": _DATE, "end_date": _DATE},
            "required": ["start_date", "end_date"],
            "additionalProperties": False,
        },
    },
    {
        "name": "forecast_goal",
        "description": "Whether the user is on track for a savings goal, the gap and its range.",
        "input_schema": {
            "type": "object",
            "properties": {"goal_name": {"type": "string"}},
            "required": ["goal_name"],
            "additionalProperties": False,
        },
    },
]
_SPECS = {spec["name"]: spec for spec in TOOL_SPECS}


class ToolError(ValueError):
    """A tool call the caller can fix (bad dates, unknown category): reported, not raised on."""


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


class Tools:
    def __init__(self, ledger: Ledger, feedback: Feedback | None = None) -> None:
        self.base = ledger  # the model's categories
        self.feedback = feedback
        self.ledger = ledger.seen_by(feedback.store, feedback.subject) if feedback else ledger
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

    def list_goals(self) -> ToolResult:
        goals = self.ledger.goals.sort_values("target_date")
        data = {
            "currency": CURRENCY,
            "goals": [
                {
                    "name": g["name"],
                    "target_amount": money(g["target_amount"]),
                    "target_date": g["target_date"],
                    "saved": money(g["current_balance"]),
                    "saved_as_of": g["as_of_date"],
                }
                for g in goals.to_dict("records")
            ],
            "forecast": "not_available",
        }
        return ToolResult(data, Source("Savings goals", f"{len(goals)} goals"))

    def detect_anomalies(self, start_date: str, end_date: str) -> ToolResult:
        start, end = self._range(start_date, end_date)
        return self._not_available(
            "Unusual-spending alerts", f"{span_label(start, end)}", "FR-7 and FR-8"
        )

    def forecast_goal(self, goal_name: str) -> ToolResult:
        return self._not_available("Goal forecast", goal_name, "FR-10 to FR-12")

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
