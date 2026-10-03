"""Plain-language reasons for unusual-transaction flags (FR-7 §7, NFR-7).

A reason is rendered from the flag's stored evidence, never by a model or the LLM, so the
dashboard works without the LLM (NFR-6) and every number in it is one the coach can quote
(FR-14). Only the user's own numbers appear: merchant profiles move scores, but what other users
pay is never shown (NFR-2).

    reason("amount_unusual", {"usual_amount": 20.99, "ratio": 9.0, "prior_charges": 14})
    # -> "About 9× your usual charge here ($20.99, from 14 earlier charges)."
"""

import json
from collections.abc import Mapping
from datetime import date
from typing import Any

from smart_financial_coach.intelligence.anomaly.contract import evidence_errors

KIND_LABELS = {
    "duplicate": "Possible duplicate",
    "amount_unusual": "Larger than usual",
    "new_merchant": "New merchant, large amount",
}
# Wording only, not a flagging rule: below this many earlier charges, "your largest since …"
# says little, so a new-merchant reason says the history is short instead (FR-7 §8).
SHORT_HISTORY = 30
LARGEST = 0.99  # rank in history from which a charge reads as "your largest since …"
BELOW_USUAL = 0.5  # below the user's median charge, "larger than N% of your charges" misleads
# Per-category wording (FR-7 §7) when the serving side adds the charge's predicted `category`,
# its `date` and `category_largest_since` (the latest earlier charge in that category at least as
# large; None: the largest yet) to the evidence. Used when it says more than a recent date would.
CATEGORY_SINCE_DAYS = 180


def money(value: float) -> str:
    return f"${value:,.2f}"


def _times(ratio: float) -> str:
    """9.04 -> "9", 2.34 -> "2.3", 12.6 -> "13": whole multiples from 10 up."""
    if ratio >= 10:
        return f"{ratio:.0f}"
    return f"{ratio:.1f}".removesuffix(".0")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _month(ts: str) -> str:
    return date.fromisoformat(ts[:10]).strftime("%b %Y")


def reason(reason_code: str, evidence: Mapping[str, Any] | str) -> str:
    """The reason line for one flag, from its evidence (a dict or the stored JSON)."""
    text = evidence if isinstance(evidence, str) else json.dumps(dict(evidence))
    if reason_code not in KIND_LABELS:
        raise ValueError(f"unknown reason code {reason_code!r}")
    if errors := evidence_errors(reason_code, text):
        raise ValueError(f"can't render a reason: {'; '.join(errors)}")
    e = json.loads(text)
    if reason_code == "duplicate":
        minutes = round(e["minutes_apart"])
        gap = "less than a minute" if minutes < 1 else _plural(minutes, "minute")
        return f"Same amount at the same merchant, {gap} after an earlier charge."
    if reason_code == "amount_unusual":
        earlier = _plural(int(e["prior_charges"]), "earlier charge")
        return (
            f"About {_times(e['ratio'])}× your usual charge here "
            f"({money(e['usual_amount'])}, from {earlier})."
        )
    if e["prior_charges"] < SHORT_HISTORY or e["rank_in_history"] is None:
        return "First charge here, and one of your first charges, so there's little to compare."
    if e["rank_in_history"] < BELOW_USUAL and e.get("above_merchant_usual") is True:
        # Smaller than most of the user's charges, but above what the merchant usually
        # charges: flagged on the merchant's price, said without a number (NFR-2; owner, #40)
        return "First charge here, and more than this merchant usually charges."
    if (category := e.get("category")) and "category_largest_since" in e and e.get("date"):
        since = e["category_largest_since"]
        if since is None:
            return f"First charge here, and your largest {category} charge yet."
        days = (date.fromisoformat(e["date"][:10]) - date.fromisoformat(since[:10])).days
        if days >= CATEGORY_SINCE_DAYS:
            return f"First charge here, and your largest {category} charge since {_month(since)}."
    if e["largest_since"] is None:
        return "First charge here, and your largest charge yet."
    if e["rank_in_history"] >= LARGEST:
        return f"First charge here, and your largest charge since {_month(e['largest_since'])}."
    share = int(e["rank_in_history"] * 100)
    return f"First charge here, and larger than {share}% of your earlier charges."
