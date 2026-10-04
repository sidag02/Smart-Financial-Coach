"""From corrections to global training labels: the agreement rule (FR-5 and FR-6 design, §4).

A correction means one of two things: the model was wrong (the fix should reach everyone), or the
user sees it differently (it should stay theirs). The rule tells them apart by agreement across
distinct users:

- a merchant string's category becomes a **global label** when at least `n` distinct subjects
  (default 3) hold a vote on it and at least `majority` of them (two thirds) agree;
- at least one of the agreeing votes must be a **correction**: a confirmation accepts the model's
  own suggestion (automation bias), so confirmations alone never create a label;
- **re-evaluated with every vote**: a label whose agreement falls below the majority is revoked,
  and the next retraining drops it. `label_history` records when each label appeared and when it
  was revoked, for the replay's per-remap report (§7).

Only a subject's current merchant-wide feedback votes: their latest merchant-scope confirmation or
correction that isn't undone. Single-transaction changes are exceptions, not a view of the
merchant. The rule counts subjects, never corrections, so a string seen by fewer than `n` people
(a landlord's name, a person-to-person payment) can never leave its user (privacy, §4).

N, the majority and the down-weighting of users who often disagree with consensus are provisional
until the replay (owner, Oct 3, 2026); down-weighting isn't built yet.

    votes = current_votes(events)
    labels = global_labels(votes)  # merchant_key -> Label
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol


class Event(Protocol):
    """What the rule reads from a feedback event (`access.feedback.Correction` has these)."""

    @property
    def seq(self) -> int: ...
    @property
    def subject(self) -> str: ...
    @property
    def action(self) -> str: ...  # "confirm" or "correct"
    @property
    def scope(self) -> str: ...  # "merchant" or "transaction"
    @property
    def merchant_key(self) -> str: ...
    @property
    def to_category(self) -> str: ...
    @property
    def undone(self) -> bool: ...


@dataclass(frozen=True)
class AgreementRule:
    n: int = 3  # distinct subjects holding a vote
    majority: float = 2 / 3  # share of them agreeing on one category
    corrections: int = 1  # agreeing votes that must be corrections, not confirmations


@dataclass(frozen=True)
class Vote:
    subject: str
    merchant_key: str
    category: str
    corrected: bool  # a choice the subject made, not an accepted suggestion
    seq: int  # when it was cast (the event's order)


@dataclass(frozen=True)
class Label:
    merchant_key: str
    category: str
    voters: int  # distinct subjects with a vote on this string
    agreeing: int
    corrections: int  # agreeing votes that were corrections


@dataclass(frozen=True)
class Change:
    """A label appearing, changing category or being revoked (category None), at a vote."""

    seq: int
    merchant_key: str
    category: str | None
    voters: int


def current_votes(events: Iterable[Event]) -> list[Vote]:
    """Each subject's current merchant-wide vote per string, in the order they were cast."""
    latest: dict[tuple[str, str], Event] = {}
    for e in sorted(events, key=lambda e: e.seq):
        if e.scope != "merchant":
            continue
        key = (e.subject, e.merchant_key)
        if e.undone:
            # Skipped, so the subject's vote is their latest standing one: what they see (§3)
            continue
        latest[key] = e
    votes = [
        Vote(e.subject, e.merchant_key, e.to_category, e.action == "correct", e.seq)
        for e in latest.values()
    ]
    return sorted(votes, key=lambda v: v.seq)


def _label(merchant_key: str, votes: list[Vote], rule: AgreementRule) -> Label | None:
    if len(votes) < rule.n:
        return None
    counts = Counter(v.category for v in votes)
    (top, agreeing), *rest = counts.most_common()
    if rest and rest[0][1] == agreeing:  # a tie has no majority
        return None
    if agreeing < rule.majority * len(votes):
        return None
    corrections = sum(v.corrected for v in votes if v.category == top)
    if corrections < rule.corrections:
        return None
    return Label(merchant_key, top, len(votes), agreeing, corrections)


def global_labels(votes: Iterable[Vote], rule: AgreementRule | None = None) -> dict[str, Label]:
    """The global labels the votes support now: merchant_key -> Label."""
    rule = rule or AgreementRule()
    by_string: dict[str, list[Vote]] = {}
    for v in votes:
        by_string.setdefault(v.merchant_key, []).append(v)
    labels = {key: _label(key, vs, rule) for key, vs in by_string.items()}
    return {key: label for key, label in labels.items() if label is not None}


def label_history(votes: Iterable[Vote], rule: AgreementRule | None = None) -> list[Change]:
    """Every label change as votes arrive in order: re-evaluation at each vote (§4)."""
    rule = rule or AgreementRule()
    standing: dict[str, dict[str, Vote]] = {}  # merchant_key -> subject -> vote
    current: dict[str, str] = {}
    changes = []
    for v in sorted(votes, key=lambda v: v.seq):
        standing.setdefault(v.merchant_key, {})[v.subject] = v
        voters = list(standing[v.merchant_key].values())
        label = _label(v.merchant_key, voters, rule)
        now = label.category if label else None
        if now != current.get(v.merchant_key):
            changes.append(Change(v.seq, v.merchant_key, now, len(voters)))
            if now is None:
                current.pop(v.merchant_key, None)
            else:
                current[v.merchant_key] = now
    return changes


def splits(votes: Iterable[Vote], rule: AgreementRule | None = None) -> Mapping[str, Counter[str]]:
    """Strings with enough voters but no label for want of a majority: product signal (§4), a
    taxonomy question rather than a model fix."""
    rule = rule or AgreementRule()
    by_string: dict[str, list[Vote]] = {}
    for v in votes:
        by_string.setdefault(v.merchant_key, []).append(v)
    labelled = global_labels((v for vs in by_string.values() for v in vs), rule)
    return {
        key: Counter(v.category for v in vs)
        for key, vs in by_string.items()
        if len(vs) >= rule.n and key not in labelled
    }
