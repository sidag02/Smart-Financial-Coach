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
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Protocol

import pandas as pd

from smart_financial_coach.access.ledger import INCOME, REVIEW_BELOW, Ledger

CURRENCY = "USD"
DASH = "\u2013"  # en dash, for date ranges
MAX_TRANSACTIONS = 50

ToolSpec = dict[str, Any]

_DATE = {"type": "string", "format": "date", "description": "YYYY-MM-DD, inclusive"}
TOOL_SPECS: list[ToolSpec] = [
    {
        "name": "get_spending_summary",
        "description": (
            "Spending and income for the signed-in user over a date range: totals, spending by "
            "category, and spending and income by calendar month. Spending excludes Income; "
            "refunds reduce their category's spending."
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
            f"total as well as up to {MAX_TRANSACTIONS} rows. Amounts are negative for money out."
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


class ToolsUnavailableError(RuntimeError):
    """The tools can't be reached at all (the MCP server refused or failed): no answer possible."""


def _check_type(tool: str, key: str, value: Any, prop: dict[str, Any]) -> None:
    """Arguments come from the model, so check their JSON types before any code uses them."""
    kind = prop["type"]
    ok = (
        isinstance(value, str)
        if kind == "string"
        else (isinstance(value, int) and not isinstance(value, bool) if kind == "integer" else True)
    )
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


class Tools:
    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self.categories = sorted(set(ledger.transactions["category"]))
        self._handlers: dict[str, Callable[..., ToolResult]] = {
            "get_spending_summary": self.get_spending_summary,
            "get_transactions": self.get_transactions,
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
                    "needs_review": bool(r["confidence"] < REVIEW_BELOW),
                }
                for r in shown.to_dict("records")
            ],
        }
        return ToolResult(data, Source(title, f"{len(rows)} matched"))

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
