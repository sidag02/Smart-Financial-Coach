"""Review items (FR-5 §2): one per merchant string a user still has flagged transactions at.

Derived from the ledger as its subject sees it (`Ledger.seen_by`), not stored: flags come from the
batch (the model's review policy), and an override settles every flag it covers, so an item
closes as soon as it is confirmed or corrected and reopens if that is undone. Ranked by the
spend still unreviewed, then by recency.

    items = open_review_items(ledger.seen_by(store, subject).transactions)
"""

import hashlib

import pandas as pd

INCOME = "Income"
REASON_ORDER = ("new_merchant", "low_confidence")  # an item is "new" if any of its rows is


def item_id(user_id: str, merchant_key: str) -> str:
    """Stable for a user and merchant string, and opaque: it doesn't reveal the string."""
    return hashlib.sha256(f"{user_id}\x1f{merchant_key}".encode()).hexdigest()[:16]


def open_review_items(transactions: pd.DataFrame, user_id: str) -> pd.DataFrame:
    """One row per merchant string with flagged transactions, most unreviewed spend first.

    Columns: item_id, merchant_key, merchant, sample_description, transaction_count,
    flagged_count, total_spend, unreviewed_spend (money out, as positive amounts),
    suggested_category, confidence (the lowest among flagged rows), reason, first_seen, last_seen.
    """
    flagged = transactions[transactions["needs_review"]]
    columns = [
        "item_id",
        "merchant_key",
        "merchant",
        "sample_description",
        "transaction_count",
        "flagged_count",
        "total_spend",
        "unreviewed_spend",
        "suggested_category",
        "confidence",
        "reason",
        "first_seen",
        "last_seen",
    ]
    if flagged.empty:
        return pd.DataFrame(columns=columns)
    at_flagged = transactions[transactions["merchant_key"].isin(set(flagged["merchant_key"]))]
    # What resolving the item covers: a merchant override leaves rows predicted Income alone
    if "model_category" in at_flagged:
        at_flagged = at_flagged[at_flagged["model_category"] != "Income"]
    spend = (-at_flagged["amount"]).clip(lower=0)
    totals = (
        at_flagged.assign(spend=spend)
        .groupby("merchant_key")
        .agg(transaction_count=("transaction_id", "size"), total_spend=("spend", "sum"))
    )

    def summarize(rows: pd.DataFrame) -> pd.Series:
        newest = rows.sort_values("ts").iloc[-1]
        reasons = set(rows["review_reason"])
        return pd.Series(
            {
                "merchant": newest["merchant"],
                "sample_description": newest["merchant_raw"],
                "flagged_count": len(rows),
                "unreviewed_spend": float((-rows["amount"]).clip(lower=0).sum()),
                "suggested_category": rows["category"].mode().sort_values().iloc[0],
                "confidence": float(rows["confidence"].min()),
                "reason": next(r for r in REASON_ORDER if r in reasons),
                "first_seen": rows["ts"].min(),
                "last_seen": rows["ts"].max(),
            }
        )

    items = flagged.groupby("merchant_key")[list(flagged.columns)].apply(summarize)
    items = items.join(totals).reset_index()
    items["item_id"] = [item_id(user_id, k) for k in items["merchant_key"]]
    items = items.sort_values(
        ["unreviewed_spend", "last_seen", "merchant_key"], ascending=[False, False, True]
    )
    return items[columns].reset_index(drop=True)


def alternatives(
    transactions: pd.DataFrame, merchant_key: str, suggested: str, n: int
) -> list[str]:
    """Quick picks next to an item's suggestion (Web App UI, gap 2: the predictions keep only the
    top category). First the other categories the model gave this merchant's transactions (an
    ambiguous merchant such as a warehouse club), then the user's most frequent spending
    categories by transaction count: fixed bills are rarely the right alternative for a shop."""
    spending = transactions[transactions["model_category"] != INCOME]
    here = spending.loc[spending["merchant_key"] == merchant_key, "model_category"]
    ranked = list(here.value_counts().index) + list(spending["model_category"].value_counts().index)
    picks: list[str] = []
    for c in ranked:
        if c != suggested and c not in picks:
            picks.append(str(c))
    return picks[:n]
