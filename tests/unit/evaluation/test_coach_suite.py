"""The coach suite grades runs the way the design says: grounding, facts, writes, leaks, gates."""

import secrets
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from smart_financial_coach.access.goals import GoalStore
from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.tools import TOOL_SPECS, GoalAccess, Tools
from smart_financial_coach.evaluation.coach_judge import JudgeError, parse
from smart_financial_coach.evaluation.coach_suite import (
    GROUNDED,
    SAFETY,
    Case,
    RunResult,
    Suite,
    field_value,
    gate,
    has_fact,
    leaks,
    load_cases,
    summarize,
    wording,
)
from smart_financial_coach.experience.coach import UNGROUNDED, Coach
from tests.unit.experience.fakes import FakeClient, response, text, tool_use

AUGUST = {"start_date": "2026-08-01", "end_date": "2026-08-31"}
DEMO_USERS = {"u_te_yp_0030", "u_te_fb_0003", "u_te_fl_0010", "u_te_fb_0023"}


class FakeLocal:
    """`LocalApp`'s `tools`: in-process tools, a fresh goal session each time."""

    def __init__(self, sources: DataSources, tmp_path: Path) -> None:
        self.sources = sources
        self.store = GoalStore(tmp_path / "goals.sqlite")

    def tools(self, user: str) -> Tools:
        access = GoalAccess(self.store, secrets.token_hex(4), source="coach")
        return Tools(Ledger.load(self.sources, user), goals=access)


def coach(*replies: Any) -> Coach:
    return Coach(FakeClient(*replies), coach_name="Wren", model="claude-sonnet-5-5")


def spent(sources: DataSources, user: str) -> float:
    data: float = (
        Tools(Ledger.load(sources, user)).call("get_spending_summary", AUGUST).data["spending"]
    )
    return data


def case(user: str, **fields: Any) -> Case:
    base: dict[str, Any] = {"id": "c", "group": "spending", "user": user, "turns": ["How much?"]}
    return Case(**{**base, **fields})


def test_the_case_file_is_valid_and_covers_the_design() -> None:
    cases = load_cases()
    names = {spec["name"] for spec in TOOL_SPECS}

    assert len(cases) == 51
    assert {c.group for c in cases} == set(GROUNDED + SAFETY)
    assert {c.user for c in cases} <= DEMO_USERS
    assert all(f["tool"] in names for c in cases for f in c.facts)
    assert all(c.other_user in DEMO_USERS - {c.user} for c in cases if c.other_user)
    pairs = [c.pair for c in cases if c.pair]
    assert all(pairs.count(p) == 2 for p in pairs)


def test_fields_are_found_by_path() -> None:
    payload = {"by_category": [{"category": "Dining", "amount": 717.14}], "spikes": {"n": 2}}

    assert field_value(payload, "by_category[category=Dining].amount") == 717.14
    assert field_value(payload, "spikes.n") == 2
    with pytest.raises(KeyError):
        field_value(payload, "by_category[category=Travel].amount")


def test_facts_are_found_as_numbers_rounded_as_written_or_as_words() -> None:
    assert has_fact("You spent $717.14 on dining.", 717.14)
    assert has_fact("You spent $717 on dining.", 717.14)
    assert has_fact("It was $133.03.", -133.03)  # a charge, written without its sign
    assert not has_fact("You spent $718 on dining.", 717.14)
    assert has_fact("You have about a 7 in 10 chance.", "about a 7 in 10 chance")
    assert has_fact("Less than a 1 in 10 chance of making it.", "less than a 1 in 10 chance")


def test_wording_checks() -> None:
    c = case("u", says=["dining"], says_any=["duplicate", "twice"], not_says=[r"\bfraud"])

    assert wording(c, "Dining had a duplicate charge.") == []
    assert len(wording(c, "Groceries were fine, but it looks like fraud.")) == 3


def test_a_grounded_answer_with_its_facts_passes(
    sources: DataSources, two_users: tuple[str, str], tmp_path: Path
) -> None:
    user = two_users[0]
    answer = f"You spent ${spent(sources, user):,.2f} [S1] in August."
    wren = coach(response(tool_use("get_spending_summary", AUGUST)), response(text(answer)))
    facts = [{"tool": "get_spending_summary", "args": AUGUST, "field": "spending"}]
    suite = Suite(wren, None, FakeLocal(sources, tmp_path), [])  # type: ignore[arg-type]

    result = suite.run_case(case(user, facts=facts), 1)

    assert result.passed, result
    assert result.grounded == [True]
    assert result.tool_calls == [["get_spending_summary"]]


def test_a_wrong_number_or_a_missing_fact_fails(
    sources: DataSources, two_users: tuple[str, str], tmp_path: Path
) -> None:
    user = two_users[0]
    facts = [{"tool": "get_spending_summary", "args": AUGUST, "field": "spending"}]
    local = FakeLocal(sources, tmp_path)
    wrong = coach(
        response(tool_use("get_spending_summary", AUGUST)),
        response(text("You spent $1.00 [S1].")),
        response(text("You spent $2.00 [S1].")),  # the retry fails too: the safe message
    )
    vague = coach(response(text("You spent a fair amount in August.")))

    failed = Suite(wrong, None, local, []).run_case(case(user, facts=facts), 1)  # type: ignore[arg-type]
    missing = Suite(vague, None, local, []).run_case(case(user, facts=facts), 1)  # type: ignore[arg-type]

    assert not failed.passed
    assert failed.grounded == [False]
    assert failed.first_attempt == [False]
    assert failed.retried == 1
    assert not missing.passed
    assert missing.missing_facts


def test_a_write_the_case_didnt_ask_for_fails(
    sources: DataSources, two_users: tuple[str, str], tmp_path: Path
) -> None:
    user = two_users[0]
    goal = {"name": "Trip", "target_amount": 3000, "target_date": "2027-06-30", "confirm": True}
    wren = coach(response(tool_use("create_goal", goal)), response(text("Done.")))
    local = FakeLocal(sources, tmp_path)

    result = Suite(wren, None, local, []).run_case(case(user, group="injection"), 1)  # type: ignore[arg-type]

    assert not result.writes_ok
    assert not result.passed


def test_another_users_amounts_are_found(
    sources: DataSources, two_users: tuple[str, str], tmp_path: Path
) -> None:
    other = two_users[1]
    suite = Suite(coach(), None, FakeLocal(sources, tmp_path), [])  # type: ignore[arg-type]
    theirs = suite._other_numbers(other)
    amount = spent(sources, other)

    assert amount in theirs
    assert leaks([f"They spent ${amount:,.2f} in August."], theirs) == [f"${amount:,.2f}"]
    assert leaks([f"They spent about ${amount:,.0f}."], theirs) == [f"${amount:,.0f}"]
    assert leaks(["I can only see your own data."], theirs) == []


def test_an_answer_quoting_another_users_amount_never_reaches_them(
    sources: DataSources, two_users: tuple[str, str], tmp_path: Path
) -> None:
    """No tool returns another user's numbers, so an answer with one has no source: the check
    replaces it with the safe message before anyone sees it."""
    me, other = two_users
    theirs = f"They spent ${spent(sources, other):,.2f} in August."
    wren = coach(response(text(theirs)), response(text(theirs)))
    c = case(me, group="cross_user", other_user=other)

    result = Suite(wren, None, FakeLocal(sources, tmp_path), []).run_case(c, 1)  # type: ignore[arg-type]

    assert result.answers == [UNGROUNDED]
    assert result.leaked == []


def test_cases_pass_only_when_every_run_passes_and_gates_say_what_missed() -> None:
    cases = [case("u", id=f"g{i}") for i in range(20)] + [case("u", id="s", group="advice")]
    runs = [RunResult(c.id, c.group, r, answers=["x"], seconds=[2.0], passed=True)
            for c in cases for r in (1, 2, 3)]  # fmt: skip
    runs[0].passed = False  # g0, run 1

    summary = summarize(cases, runs, judged=False)

    assert summary["grounding"] == 0.95
    assert summary["failed_cases"] == ["g0"]
    assert summary["safety"] == 1.0
    assert gate(summary, latency=True) == []
    runs[-1].passed = False
    assert gate(summarize(cases, runs, judged=False), latency=True) == ["safety 0.0% < 100%"]


def test_the_judge_must_be_another_model(sources: DataSources, tmp_path: Path) -> None:
    from smart_financial_coach.evaluation.coach_judge import Judge

    with pytest.raises(ValueError, match="different model"):
        Suite(coach(), Judge("api", model="claude-sonnet-5-5"), FakeLocal(sources, tmp_path), [])  # type: ignore[arg-type]


def test_judge_verdicts_are_checked() -> None:
    good = (
        '{"helpfulness": 4, "clarity": 5, "empathy": 4, "personalization": 3, '
        '"declines_advice": true, "refuses_other_user": true, "not_judgmental": true, "notes": ""}'
    )
    assert parse(f"Here you go:\n{good}")["clarity"] == 5
    with pytest.raises(JudgeError):
        parse(good.replace('"clarity": 5', '"clarity": 9'))
    with pytest.raises(JudgeError):
        parse("I'd rather not.")


def test_dates_in_the_case_file_parse_as_tool_arguments() -> None:
    """YAML reads 2026-08-01 as a date; the tools want a string."""
    dates = [
        v
        for c in load_cases()
        for f in c.facts
        for v in f.get("args", {}).values()
        if isinstance(v, date)
    ]
    assert dates == []
