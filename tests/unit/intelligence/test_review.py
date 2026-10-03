"""FR-5 §1: the review policy's rule, its flags, and that it belongs to the promoted model."""

from pathlib import Path

import numpy as np
import pytest

from smart_financial_coach.intelligence.categorization.review import (
    POLICY_FILE,
    PolicyError,
    ReviewPolicy,
    ReviewRule,
    derive_review_policy,
    load_review_policy,
    save_review_policy,
)
from smart_financial_coach.intelligence.models.artifact import POINTER_FILE


def rows(
    familiar: list[tuple[float, bool]], unfamiliar: list[tuple[float, bool]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(confidence, familiar, wrong) arrays from (confidence, wrong) pairs per group."""
    pairs = [(c, True, w) for c, w in familiar] + [(c, False, w) for c, w in unfamiliar]
    conf, known, wrong = zip(*pairs, strict=True)
    return np.array(conf), np.array(known), np.array(wrong)


def test_the_rule_catches_enough_unfamiliar_errors_and_flags_familiar_ones_worth_a_glance() -> None:
    conf, known, wrong = rows(
        # Familiar: one error at 0.62, and correct rows above it. Below 0.65 one flag, an error;
        # below 0.95 four flags, one an error (25%); higher thresholds are past the cap
        familiar=[(0.62, True), (0.7, False), (0.8, False), (0.9, False), (0.99, False)],
        # Unfamiliar: errors at 0.3, 0.5 and 0.85 (and one confident at 0.97). Below 0.55 two of
        # four (50%); below 0.9, three (75%): the lowest threshold catching 60% is 0.9
        unfamiliar=[(0.3, True), (0.5, True), (0.6, False), (0.85, True), (0.97, True)],
    )

    policy = derive_review_policy("v1", conf, known, wrong)

    assert policy.unfamiliar_threshold == 0.9
    assert policy.familiar_threshold == 0.95
    assert policy.evidence["unfamiliar"]["rows"] == 5
    assert policy.evidence["familiar"]["error_rate"] == pytest.approx(0.2)


def test_familiar_flags_never_reach_certainty() -> None:
    """Every familiar threshold's flags are all errors: the cap stops the rule at 0.95."""
    conf, known, wrong = rows(familiar=[(0.3, True), (0.99, False)], unfamiliar=[(0.1, True)])
    assert derive_review_policy("v1", conf, known, wrong).familiar_threshold == 0.95


def test_no_familiar_flags_when_none_would_be_worth_a_glance() -> None:
    conf, known, wrong = rows(
        familiar=[(c, False) for c in (0.2, 0.5, 0.8)], unfamiliar=[(0.1, True)]
    )
    assert derive_review_policy("v1", conf, known, wrong).familiar_threshold == 0.0


def test_an_unreachable_unfamiliar_target_is_the_owners_call() -> None:
    """Confident errors can't be caught by a threshold; the rule isn't relaxed in code."""
    conf, known, wrong = rows(familiar=[(0.5, True)], unfamiliar=[(0.1, True), (0.99, True)])

    with pytest.raises(PolicyError, match="at most 50%"):
        derive_review_policy("v1", conf, known, wrong)
    lower = ReviewRule(unfamiliar_errors_caught=0.5)
    assert derive_review_policy("v1", conf, known, wrong, lower).unfamiliar_threshold == 0.15


def test_flags_and_reasons() -> None:
    policy = ReviewPolicy("v1", familiar_threshold=0.95, unfamiliar_threshold=0.8)

    needs, reason = policy.flag([0.9, 0.97, 0.7, 0.85], [True, True, False, False])

    assert needs.tolist() == [True, False, True, False]
    assert reason.tolist() == ["low_confidence", "", "new_merchant", ""]


def test_the_promoted_models_policy_is_loaded_and_must_be_its_own(tmp_path: Path) -> None:
    root = tmp_path / "categorization"
    (root / "v2").mkdir(parents=True)
    (root / POINTER_FILE).write_text("v2\n")

    with pytest.raises(PolicyError, match=f"no {POLICY_FILE}"):
        load_review_policy(tmp_path)

    save_review_policy(ReviewPolicy("v1", 0.95, 0.8), root / "v2")
    with pytest.raises(PolicyError, match="for model v1, not v2"):
        load_review_policy(tmp_path)

    conf, known, wrong = rows(familiar=[(0.5, True)], unfamiliar=[(0.1, True)])
    policy = derive_review_policy("v2", conf, known, wrong, source={"run": "r1"})
    save_review_policy(policy, root / "v2")
    assert load_review_policy(tmp_path) == policy
