"""FR-5/FR-6 §4: global labels only by agreement among distinct users, re-evaluated per vote."""

from dataclasses import dataclass

from smart_financial_coach.intelligence.categorization.agreement import (
    AgreementRule,
    Change,
    Vote,
    current_votes,
    global_labels,
    label_history,
    splits,
)


@dataclass(frozen=True)
class E:
    seq: int
    subject: str
    action: str
    to_category: str
    scope: str = "merchant"
    merchant_key: str = "netflix"
    undone_at: str | None = None

    @property
    def undone(self) -> bool:
        return self.undone_at is not None


def v(seq: int, subject: str, category: str, corrected: bool = True, key: str = "netflix") -> Vote:
    return Vote(subject, key, category, corrected, seq)


def test_three_distinct_users_agreeing_make_a_label() -> None:
    votes = [v(1, "a", "Entertainment"), v(2, "b", "Entertainment"), v(3, "c", "Subscriptions")]

    (label,) = global_labels(votes).values()

    assert (label.category, label.voters, label.agreeing, label.corrections) == (
        "Entertainment", 3, 2, 2,
    )  # fmt: skip
    assert global_labels(votes[:2]) == {}  # two users aren't enough, however sure


def test_a_label_needs_a_correction_not_just_confirmations() -> None:
    """Three habitual confirmations of a model error never become a label (automation bias)."""
    confirmed = [v(i, s, "Travel", corrected=False) for i, s in enumerate("abc")]
    assert global_labels(confirmed) == {}
    assert global_labels([*confirmed, v(9, "d", "Travel")])["netflix"].corrections == 1


def test_one_user_can_never_make_a_label_alone() -> None:
    events = [E(i, "a", "correct", "Entertainment") for i in range(10)]
    assert global_labels(current_votes(events)) == {}


def test_splits_stay_personal_and_are_reported() -> None:
    votes = [v(1, "a", "Groceries"), v(2, "b", "Shopping"), v(3, "c", "Groceries"),
             v(4, "d", "Shopping")]  # fmt: skip

    assert global_labels(votes) == {}  # 2-2: no majority
    assert splits(votes)["netflix"] == {"Groceries": 2, "Shopping": 2}


def test_labels_are_re_evaluated_and_revoked_as_votes_arrive() -> None:
    """With N = 3 the first three votes always have a 2-of-3 side; later votes can revoke it."""
    cats = ["Entertainment"] * 2 + ["Subscriptions"] * 4
    votes = [v(i + 1, f"s{i}", c) for i, c in enumerate(cats)]

    assert label_history(votes) == [
        Change(3, "netflix", "Entertainment", 3),  # 2 of 3
        Change(4, "netflix", None, 4),  # 2-2: revoked; at 5 votes, 3 of 5 is short of two thirds
        Change(6, "netflix", "Subscriptions", 6),  # 4 of 6
    ]


def test_votes_are_each_subjects_standing_merchant_feedback() -> None:
    events = [
        E(1, "a", "correct", "Travel"),
        E(2, "a", "correct", "Dining"),  # changed their mind: Dining is the vote
        E(3, "b", "correct", "Dining", undone_at="t"),  # undone: no vote
        E(4, "c", "correct", "Dining", scope="transaction"),  # one transaction: an exception
        E(5, "d", "confirm", "Dining", merchant_key="other"),
    ]

    votes = current_votes(events)

    assert [(x.subject, x.merchant_key, x.category, x.corrected) for x in votes] == [
        ("a", "netflix", "Dining", True),
        ("d", "other", "Dining", False),
    ]


def test_the_rule_is_a_setting() -> None:
    votes = [v(1, "a", "Entertainment"), v(2, "b", "Entertainment")]
    assert global_labels(votes, AgreementRule(n=2))["netflix"].category == "Entertainment"
