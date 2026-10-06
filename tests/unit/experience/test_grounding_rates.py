"""How often the grounding check passes a wrong number, and fails a right one (decision 8).

Realistic tool results (the test users' summaries, alerts, goals, forecasts, transactions and
review items, from the real tools) and plausible numbers: each true value written as the coach
writes it: amounts (cents and whole dollars), counts (fields and list lengths) and percentages.
Wrong ones are each value moved by 3 to 50%, random amounts in the result's range and random whole
percentages. Targets: false accepts ≤ 1% for amounts and percentages, ≤ 5% for derived numbers
(sums and differences); false rejects ≤ 2% (FR-13 to FR-15 design, §3, decision 8). Counts are
reported on their own (see COUNT_FALSE_ACCEPTS).
"""

import random
from collections.abc import Iterator
from datetime import date
from itertools import pairwise
from typing import Any

import pytest

from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.tools import Tools
from smart_financial_coach.experience.grounding import _derived, check, values

DIRECT_FALSE_ACCEPTS = 0.01
DERIVED_FALSE_ACCEPTS = 0.05
# Counts are small whole numbers, so a wrong one often equals another count in the same result:
# 7.2% measured (review on #74). Only field-level citations would tell them apart (decision 8's
# follow-up), so this is a bound against regressions until the owner decides (asked on #74)
COUNT_FALSE_ACCEPTS = 0.10
FALSE_REJECTS = 0.02


def payloads(tools: Tools) -> Iterator[dict[str, Any]]:
    """One user's tool results, as the coach would see them."""
    end = tools.as_of
    for months_back in range(6):
        year, month = divmod(end.year * 12 + end.month - 1 - months_back, 12)
        start = date(year, month + 1, 1)
        yield tools.call(
            "get_spending_summary", {"start_date": start.isoformat(), "end_date": end.isoformat()}
        ).data
    year_ago = date(end.year - 1, end.month, 1).isoformat()
    yield tools.call("detect_anomalies", {"start_date": year_ago, "end_date": end.isoformat()}).data
    yield tools.call(
        "get_transactions",
        {"start_date": year_ago, "end_date": end.isoformat(), "sort": "largest", "limit": 10},
    ).data
    yield tools.call("list_review_items", {"limit": 10}).data
    goals = tools.call("list_goals", {}).data
    yield goals
    for goal in goals["goals"]:
        if goal["status"] in ("active", "reached"):
            yield tools.call("forecast_goal", {"goal_id": goal["goal_id"]}).data


def money(value: float, cents: bool) -> str:
    return f"${abs(value):,.2f}" if cents else f"${round(abs(value)):,}"


def moved(value: float, rng: random.Random) -> float:
    return value * (1 + rng.choice((-1, 1)) * rng.uniform(0.03, 0.5))


@pytest.fixture(scope="module")
def results(sources: DataSources, two_users: tuple[str, str]) -> list[dict[str, Any]]:
    return [
        {"source_id": "S1", **payload}
        for user in two_users
        for payload in payloads(Tools(Ledger.load(sources, user)))
    ]


def rate(passed: list[bool]) -> float:
    return sum(passed) / len(passed)


def test_the_check_rarely_passes_a_wrong_number_or_fails_a_right_one(
    results: list[dict[str, Any]],
) -> None:
    rng = random.Random(0)
    right: list[bool] = []
    wrong: dict[str, list[bool]] = {"amounts": [], "counts": [], "percentages": [], "derived": []}

    def says(text: str, cited: dict[str, Any]) -> bool:
        return check(f"It was {text} [S1].", cited).ok

    for payload in results:
        cited = {"S1": payload}
        found = [v for v in values("S1", payload) if v.kind != "text"]
        amounts = [v for v in found if v.money and abs(v.number) >= 1]
        counts = [v for v in found if v.unit == "count" and not v.row]
        shares = [v for v in found if v.unit in ("proportion", "ratio") and not v.row]
        for v in amounts:
            for cents in (True, False):
                good = money(v.number, cents)
                right.append(says(good, cited))
                bad = money(moved(v.number, rng), cents)
                if bad != good:
                    wrong["amounts"].append(says(bad, cited))
        if amounts:
            low, high = min(abs(v.number) for v in amounts), max(abs(v.number) for v in amounts)
            for _ in amounts:
                wrong["amounts"].append(
                    says(money(rng.uniform(low, high), rng.random() < 0.5), cited)
                )
        for v in counts:
            good = f"{round(v.number):,}"
            right.append(says(good, cited))
            bad = f"{round(moved(v.number, rng)):,}"
            if bad != good:  # a small count moved a little can round back to itself
                wrong["counts"].append(says(bad, cited))
        for v in shares:
            pct = abs(v.number - 1) * 100 if v.unit == "ratio" else v.number * 100
            good = f"{round(pct)}%"
            right.append(says(good, cited))
        if shares or amounts:  # a random whole percentage, cited to the result
            for _ in range(10):
                bad = f"{rng.randint(1, 100)}%"
                if not any(
                    f"{round(abs(v.number - 1) * 100 if v.unit == 'ratio' else v.number * 100)}%"
                    == bad
                    for v in shares
                ):
                    wrong["percentages"].append(says(bad, cited))
        summary = [v for v in amounts if not v.row]
        derived = sorted(d for d in _derived(summary) if d >= 1)
        for d in rng.sample(derived, min(len(derived), 50)):
            right.append(says(money(d, True), cited))
            wrong["derived"].append(says(money(moved(d, rng), rng.random() < 0.5), cited))

    print(  # the measured rates, for the results report
        f"\nfalse rejects {1 - rate(right):.2%} of {len(right)}; false accepts: "
        + ", ".join(f"{kind} {rate(passed):.2%} of {len(passed)}" for kind, passed in wrong.items())
    )
    assert all(len(passed) > 100 for passed in wrong.values())
    assert 1 - rate(right) <= FALSE_REJECTS
    assert rate(wrong["amounts"]) <= DIRECT_FALSE_ACCEPTS
    assert rate(wrong["percentages"]) <= DIRECT_FALSE_ACCEPTS
    assert rate(wrong["counts"]) <= COUNT_FALSE_ACCEPTS
    assert rate(wrong["derived"]) <= DERIVED_FALSE_ACCEPTS


def test_the_paragraph_fallback_rarely_passes_a_wrong_number(
    results: list[dict[str, Any]],
) -> None:
    """A list cited once, at its end, lends the paragraph's tags to its untagged lines (design §3,
    as changed on #75). Measured where it's loosest: a paragraph citing two results, and an
    untagged line under it, checked against both."""
    rng = random.Random(1)
    right: list[bool] = []
    wrong: list[bool] = []
    for first, second in pairwise(results):
        cited = {"S1": first, "S2": {**second, "source_id": "S2"}}
        amounts = [
            v
            for source, payload in cited.items()
            for v in values(source, payload)
            if v.kind == "field" and v.money and abs(v.number) >= 1
        ]
        if not amounts:
            continue
        low, high = min(abs(v.number) for v in amounts), max(abs(v.number) for v in amounts)

        def says(text: str, cited: dict[str, Any] = cited) -> bool:
            return check(f"Here's what I found:\n- {text}\nBoth from [S1, S2].", cited).ok

        for v in rng.sample(amounts, min(len(amounts), 20)):
            good = money(v.number, cents=True)
            right.append(says(good))
            bad = money(moved(v.number, rng), cents=rng.random() < 0.5)
            if bad not in (good, money(v.number, cents=False)):
                wrong.append(says(bad))
            wrong.append(says(money(rng.uniform(low, high), cents=rng.random() < 0.5)))

    print(
        f"\nparagraph fallback: false rejects {1 - rate(right):.2%} of {len(right)}, "
        f"false accepts {rate(wrong):.2%} of {len(wrong)}"
    )
    assert len(wrong) > 300
    assert 1 - rate(right) <= FALSE_REJECTS
    assert rate(wrong) <= DIRECT_FALSE_ACCEPTS
