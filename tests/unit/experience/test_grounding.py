"""The grounding check passes numbers a cited tool result holds, and nothing else (FR-14)."""

import pytest

from smart_financial_coach.experience.grounding import check, instruction_numbers

SUMMARY = {
    "source_id": "S1",
    "currency": "USD",
    "start_date": "2026-08-01",
    "end_date": "2026-08-31",
    "income": 5992.73,
    "spending": 4412.35,
    "net": 1580.38,
    "transactions": 83,
    "unreviewed_spend": 460.13,
    "open_review_items": 4,
    "by_category": [
        {"category": "Housing", "amount": 2118.99, "transactions": 1},
        {"category": "Dining", "amount": 717.14, "transactions": 40},
        {"category": "Groceries", "amount": 529.47, "transactions": 10},
    ],
    "by_month": [{"month": "2026-08", "spending": 4412.35, "income": 5992.73}],
}
SPIKES = {
    "source_id": "S2",
    "count": 0,
    "unusual_transactions": [
        {"transaction_id": "t_1", "date": "2026-09-30", "merchant": "Lidl", "amount": -133.03}
    ],
    "spending_spikes": {
        "spikes": [
            {
                "category": "Dining",
                "actual": 1382.65,
                "usual": 674.58,
                "excess": 708.07,
                "ratio": 2.05,
                "count": 51,
                "usual_count": 26.75,
                "reason": "You spent $1,383 on Dining in September 2026, $708 more than usual.",
                "largest_charges": [
                    {"transaction_id": "t_2", "description": "Olive Garden#4394", "amount": 123.95},
                    {"transaction_id": "t_3", "description": "Caviar*HIMGQTYT", "amount": 98.39},
                ],
            }
        ]
    },
}
GOAL = {
    "source_id": "S3",
    "goal_id": "g_1",
    "target_amount": 3000.0,
    "saved": 1200.0,
    "p_goal_met": 0.72,
    "chance_words": "about a 7 in 10 chance",
    "range": {"low": 2795.82, "high": 5258.14, "chance": 0.8},
    "months_left": 9,
}
PAYLOADS = {"S1": SUMMARY, "S2": SPIKES, "S3": GOAL}


def grounded(answer: str, *user: str, constants: frozenset[float] = frozenset()) -> bool:
    return check(answer, PAYLOADS, user, constants=constants).ok


@pytest.mark.parametrize(
    "answer",
    [
        "You spent $4,412.35 [S1] in August.",
        "You spent $4,412 [S1] in August.",  # whole dollars
        "Shopping was $529.5 [S1].",  # one decimal, as Sonnet once wrote "$413.6" (#73)
        "Dining was $717.14 [S1] across 40 transactions [S1].",
        "Dining: $717.14 [S1] across 40 transactions.",  # the tag before it in the sentence
        "Dining was $1,383 [S2], $708 more than usual [S2].",  # numbers inside the reason text
        "That's 105% more than usual [S2].",  # a ratio as a change
        "That's 2.05x your usual [S2].",
        "About 27 purchases in a usual month [S2].",  # 26.75, rounded as written
        "There's an 80% range [S3] from $2,796 to $5,258 [S3].",
        "You have an 72% chance [S3].",  # a probability as a percentage
        "You have about a 7 in 10 chance [S3].",
        "You have a 7 in 10 chance [S3].",
        "You spent $133.03 [S2] at Lidl.",  # a row quoted on its own
        "You had 3 categories [S1].",  # a list's length
        "You're $1,800 short [S3] of $3,000 [S3].",  # target minus saved: one record's fields
        "Housing and Dining came to $2,836.13 [S1].",  # one field in two items of a list
        "In August 2026 you spent $4,412.35 [S1].",  # a year isn't checked
        "By August 31, 2026, you'd spent $4,412.35 [S1].",  # nor a year before a comma
        "On Aug 31 you spent $4,412.35 [S1], and on 2026-08-01 nothing.",  # nor dates
        "A charge at Olive Garden#4394 [S2].",  # nor a store number
        "Your biggest categories:\n- Housing: $2,118.99\n- Dining: $717.14\nAll from [S1].",
        "Nothing stood out.",
    ],
)
def test_numbers_from_cited_results_pass(answer: str) -> None:
    assert grounded(answer), check(answer, PAYLOADS).unmatched


@pytest.mark.parametrize(
    ("answer", "unmatched"),
    [
        ("You spent $4,500 [S1] in August.", "$4,500"),
        ("You spent $4,412.35 in August.", "$4,412.35"),  # no source at all
        ("You spent $4,412.35 [S9].", "$4,412.35"),  # a source that doesn't exist
        ("You spent $717.14 [S2] on dining.", "$717.14"),  # the wrong source
        ("Your two biggest charges came to $222.34 [S2].", "$222.34"),  # rows aren't added
        ("Dining ran $2,382.65 [S1] over Housing.", "$2,382.65"),  # S1 and S2 values mixed
        ("You have about an 8 in 10 chance [S3].", "about 8 in 10"),
        ("You have better than a 9 in 10 chance [S3].", "better than 9 in 10"),
        ("That's 60% more [S2].", "60%"),
        ("Over 13 months [S1].", "13"),
        ("Housing $2,118.99 [S1].\n\nThose are 3 categories.", "3"),  # never the paragraph above
        (
            "Housing was $2,118.99.\n\nDining was high [S1].",  # its tags come later
            "$2,118.99",
        ),
    ],
)
def test_other_numbers_fail(answer: str, unmatched: str) -> None:
    assert check(answer, PAYLOADS).unmatched == (unmatched,)


def test_chances_follow_the_goals_pages_rounding_and_bands() -> None:
    def chance(p: float, phrase: str) -> bool:
        payloads = {"S1": {"p_goal_met": p}}
        return check(f"You have {phrase} [S1].", payloads).ok

    assert chance(0.68, "about a 6 in 10 chance")  # could go either way: never 7
    assert not chance(0.68, "about a 7 in 10 chance")
    assert chance(0.96, "better than a 9 in 10 chance")
    assert not chance(0.96, "about a 10 in 10 chance")
    assert chance(0.03, "less than a 1 in 10 chance")
    assert chance(0.06, "about a 1 in 10 chance")


def test_numbers_the_person_wrote_need_no_source() -> None:
    answer = "The $400 dinner you mentioned is part of Dining's $717.14 [S1]."
    assert not grounded(answer)
    assert grounded(answer, "Is it dumb that I spent $400 on dinner?")


def test_a_number_the_person_wrote_exempts_only_the_same_kind() -> None:
    """ "the last 3 months" doesn't let "$3" through (review on #74)."""
    assert not grounded("That costs $3.", "What did I spend in the last 3 months?")
    assert grounded("That costs $3.", "Is $3 a lot for coffee?")


def test_percentages_come_only_from_proportions_and_ratios() -> None:
    payloads = {"S1": {"amount": 35.0, "ratio": 1.42, "share": 0.25,
                       "transactions": [{"confidence": 0.61, "amount": -12.0}]}}  # fmt: skip

    def ok(answer: str) -> bool:
        return check(answer, payloads).ok

    assert ok("That's 42% more [S1].")
    assert ok("That's 142% of usual [S1].")
    assert ok("A quarter, 25% [S1].")
    assert not ok("That's 35% [S1].")  # an amount isn't a percentage
    assert not ok("We're 61% sure [S1].")  # nor is a transaction row's confidence


def test_product_rules_from_the_instructions_need_no_source() -> None:
    answer = "Compared with your average month over the past 12 months, $717.14 [S1] is normal."
    assert not grounded(answer)
    assert grounded(answer, constants=instruction_numbers("the previous 12 months"))
    # Only plain numbers: money always needs a source
    assert not grounded("It costs $12.", constants=instruction_numbers("12 months"))


def test_list_markers_and_source_tags_arent_numbers() -> None:
    answer = "Your largest:\n1. Housing $2,118.99 [S1]\n2. Dining $717.14 [S1, S2]"
    result = check(answer, PAYLOADS)
    assert result.ok
    assert result.numbers == 2
