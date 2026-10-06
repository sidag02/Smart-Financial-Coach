"""How often the grounding check passes a wrong number, and fails a right one (decision 8).

Realistic tool results (the test users' summaries, alerts, goals, forecasts, transactions and
review items, from the real tools) and plausible numbers: each true value written as the coach
writes money (cents and whole dollars), moved by 3 to 50% for the wrong ones, plus random amounts in
the result's range. Targets: false accepts ≤ 1% for direct numbers and ≤ 5% for derived ones
(sums and differences); false rejects ≤ 2% (FR-13 to FR-15 design, §3, decision 8).
"""

import random
from collections.abc import Iterator
from datetime import date
from typing import Any

import pytest

from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.tools import Tools
from smart_financial_coach.experience.grounding import _derived, check, values

DIRECT_FALSE_ACCEPTS = 0.01
DERIVED_FALSE_ACCEPTS = 0.05
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


def written(value: float, amount: bool, cents: bool) -> str:
    """A value as the coach writes it: an amount as money, a count as a plain number."""
    return money(value, cents) if amount else f"{round(abs(value)):,}"


def wrong(value: float, rng: random.Random) -> float:
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
    wrong_amounts: list[bool] = []
    wrong_counts: list[bool] = []
    wrong_derived: list[bool] = []
    for payload in results:
        cited = {"S1": payload}
        fields = [v for v in values("S1", payload) if v.kind == "field" and abs(v.number) >= 1]
        if not fields:
            continue
        derived = sorted(d for d in _derived([v for v in fields if v.money]) if d >= 1)
        low, high = min(abs(v.number) for v in fields), max(abs(v.number) for v in fields)
        for v in fields:
            for cents in (True, False) if v.money else (False,):
                good = written(v.number, v.money, cents)
                right.append(check(f"It was {good} [S1].", cited).ok)
                bad = written(wrong(v.number, rng), v.money, cents)
                if bad != good:  # a small count moved a little can round back to itself
                    (wrong_amounts if v.money else wrong_counts).append(
                        check(f"It was {bad} [S1].", cited).ok
                    )
            bad = money(rng.uniform(low, high), rng.random() < 0.5)
            wrong_amounts.append(check(f"It was {bad} [S1].", cited).ok)
        for d in rng.sample(derived, min(len(derived), 50)):
            right.append(check(f"The difference is {money(d, True)} [S1].", cited).ok)
            bad = money(wrong(d, rng), rng.random() < 0.5)
            wrong_derived.append(check(f"The difference is {bad} [S1].", cited).ok)

    wrong_direct = wrong_amounts + wrong_counts
    print(  # the measured rates, for the results report
        f"\nfalse rejects {1 - rate(right):.2%} of {len(right)}; false accepts: direct "
        f"{rate(wrong_direct):.2%} of {len(wrong_direct)} (amounts {rate(wrong_amounts):.2%} of "
        f"{len(wrong_amounts)}, counts {rate(wrong_counts):.2%} of {len(wrong_counts)}), "
        f"derived {rate(wrong_derived):.2%} of {len(wrong_derived)}"
    )
    assert len(wrong_direct) > 500
    assert len(wrong_derived) > 100
    assert 1 - rate(right) <= FALSE_REJECTS
    assert rate(wrong_direct) <= DIRECT_FALSE_ACCEPTS
    assert rate(wrong_derived) <= DERIVED_FALSE_ACCEPTS
