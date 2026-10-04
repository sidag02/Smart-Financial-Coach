"""Plain-language reasons for spending spikes (FR-8 §7, NFR-7).

A reason is rendered from the flag's stored evidence, never by a model or the LLM, so the
dashboard works without the LLM (NFR-6) and every number in it is one the coach can recompute from
the user's own spending summary (FR-14). Only the user's own numbers appear: season profiles and
income move scores, never the quoted "usual" (NFR-2). "Usual" is the user's average month in the
category over the previous 12 months, or as many as exist (FR-8 decision 7).

    reason({"category": "Dining", "period_start": "2026-08-01", "actual": 1853.0, "usual": 748.0,
            "usual_months": 12, "excess": 1105.0, "ratio": 2.477, "count": 48,
            "usual_count": 18.6})
    # -> "You spent $1,853 on Dining in August 2026, $1,105 more than your average month over the
    #     past year ($748). That came from 48 purchases, against about 19 in an average month."
"""

import json
import math
from collections.abc import Mapping
from datetime import date
from typing import Any

from smart_financial_coach.intelligence.spikes.contract import evidence_errors

KIND_LABEL = "Spending spike"
YEAR = 12


def whole(value: float) -> int:
    """Rounded half up (2.5 -> 3), not Python's half to even."""
    return math.floor(value + 0.5)


def dollars(value: int) -> str:
    """Whole dollars: a month's total reads as $1,853, not $1,853.27. Evidence keeps the cents."""
    return f"${value:,}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _span(months: int) -> str:
    return "the past year" if months >= YEAR else f"the past {months} months"


def reason(evidence: Mapping[str, Any] | str) -> str:
    """The reason line for one spike, from its evidence (a dict or the stored JSON)."""
    text = evidence if isinstance(evidence, str) else json.dumps(dict(evidence))
    if errors := evidence_errors(text):
        raise ValueError(f"can't render a reason: {'; '.join(errors)}")
    e = json.loads(text)
    month = date.fromisoformat(e["period_start"]).strftime("%B %Y")
    actual, usual = whole(e["actual"]), whole(e["usual"])
    usual_count = whole(e["usual_count"])
    about = "about " if usual_count != e["usual_count"] else ""
    # The excess shown is the rounded actual minus the rounded usual, so the sentence adds up
    return (
        f"You spent {dollars(actual)} on {e['category']} in {month}, "
        f"{dollars(actual - usual)} more than your average month over "
        f"{_span(e['usual_months'])} ({dollars(usual)}). That came from "
        f"{_plural(int(e['count']), 'purchase')}, against {about}{usual_count} in an average "
        "month."
    )
