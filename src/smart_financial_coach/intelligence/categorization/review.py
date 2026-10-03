"""The review policy (FR-5 §1): which categories a user is asked to check, and why.

One threshold per familiarity group, because the two groups' confidence behaves differently: a
single threshold either floods users with correct familiar items or misses most unfamiliar
errors. Thresholds are chosen per model, on its validation predictions, by a written rule:

- unfamiliar: the lowest threshold that catches at least 60% of unfamiliar-string errors;
- familiar: the highest threshold at which at least 25% of flags are real errors, capped at 0.95.

The model is a cold start good enough to trust (FR-3, FR-4), so review flags only what it is
unsure about; its confident errors are left to corrections and retraining (owner, Oct 3, 2026).

The policy travels with the model: the experiment runner derives it from each run's pooled
validation predictions (`sfc-experiment`), and promotion copies it next to the promoted model.

    policy = load_review_policy()
    needs_review, reason = policy.flag(out["confidence"], out["familiar"])
"""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from smart_financial_coach.config import get_settings
from smart_financial_coach.intelligence.models.artifact import promoted_version

POLICY_FILE = "review_policy.json"
SERVICE = "categorization"
# Candidate thresholds: a coarse grid, so the choice is reproducible and easy to explain
GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05 .. 0.95
REASONS = ("new_merchant", "low_confidence")  # unfamiliar, familiar


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class ReviewRule:
    unfamiliar_errors_caught: float = 0.60
    familiar_flags_wrong: float = 0.25
    familiar_cap: float = 0.95


@dataclass(frozen=True)
class ReviewPolicy:
    model_version: str
    familiar_threshold: float  # flag a familiar row whose confidence is below this
    unfamiliar_threshold: float
    rule: ReviewRule = ReviewRule()
    # How it was chosen: per group, the rows, error rate and each grid threshold's
    # flagged / errors-caught / flags-wrong shares; and where the predictions came from
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def flag(
        self, confidence: Any, familiar: Any
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.str_]]:
        """Whether each row needs review, and why: "new_merchant", "low_confidence" or ""."""
        conf = np.asarray(confidence, dtype=float)
        known = np.asarray(familiar, dtype=bool)
        threshold = np.where(known, self.familiar_threshold, self.unfamiliar_threshold)
        needs = conf < threshold
        reason = np.where(needs, np.where(known, REASONS[1], REASONS[0]), "")
        return needs, reason

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "ReviewPolicy":
        raw = json.loads(text)
        return cls(
            model_version=str(raw["model_version"]),
            familiar_threshold=float(raw["familiar_threshold"]),
            unfamiliar_threshold=float(raw["unfamiliar_threshold"]),
            rule=ReviewRule(**raw.get("rule", {})),
            evidence=raw.get("evidence", {}),
        )


@dataclass(frozen=True)
class Row:
    threshold: float
    flagged: float  # share of the group's rows below the threshold
    errors_caught: float  # share of the group's errors among them
    flags_wrong: float | None  # share of flags that are errors; None when nothing is flagged


def _table(confidence: npt.NDArray[np.float64], wrong: npt.NDArray[np.bool_]) -> list[Row]:
    errors = max(int(wrong.sum()), 1)
    rows = []
    for t in GRID:
        flag = confidence < t
        flagged, caught = int(flag.sum()), int((flag & wrong).sum())
        share = flagged / len(flag) if len(flag) else 0.0
        rows.append(Row(t, share, caught / errors, caught / flagged if flagged else None))
    return rows


def derive_review_policy(
    model_version: str,
    confidence: Any,
    familiar: Any,
    wrong: Any,
    rule: ReviewRule | None = None,
    source: Mapping[str, Any] | None = None,
) -> ReviewPolicy:
    """Choose the thresholds by `rule` from validation rows (confidence, familiarity, error).

    Raises `PolicyError` when no threshold meets the unfamiliar target: the rule is the owner's
    to change (FR-5 handoff, working practice), not something to relax in code.
    """
    rule = rule or ReviewRule()
    conf = np.asarray(confidence, dtype=float)
    known = np.asarray(familiar, dtype=bool)
    bad = np.asarray(wrong, dtype=bool)
    groups = {"familiar": known, "unfamiliar": ~known}
    tables = {name: _table(conf[mask], bad[mask]) for name, mask in groups.items()}

    catching = [
        r.threshold
        for r in tables["unfamiliar"]
        if r.errors_caught >= rule.unfamiliar_errors_caught
    ]
    if not catching:
        best = max(r.errors_caught for r in tables["unfamiliar"])
        raise PolicyError(
            f"no threshold catches {rule.unfamiliar_errors_caught:.0%} of unfamiliar errors "
            f"(at most {best:.0%}); the review rule needs the owner's decision"
        )
    # When no familiar threshold's flags are worth a glance, none are flagged
    worth = [
        r.threshold
        for r in tables["familiar"]
        if r.threshold <= rule.familiar_cap
        and r.flags_wrong is not None
        and r.flags_wrong >= rule.familiar_flags_wrong
    ]
    evidence = {
        name: {
            "rows": int(mask.sum()),
            "error_rate": float(bad[mask].mean()) if mask.any() else None,
            "table": [asdict(r) for r in tables[name]],
        }
        for name, mask in groups.items()
    }
    return ReviewPolicy(
        model_version=model_version,
        familiar_threshold=max(worth, default=0.0),
        unfamiliar_threshold=min(catching),
        rule=rule,
        evidence={**evidence, "source": dict(source or {})},
    )


def save_review_policy(policy: ReviewPolicy, folder: Path) -> Path:
    path = folder / POLICY_FILE
    path.write_text(policy.to_json(), encoding="utf-8")
    return path


def read_review_policy(folder: Path) -> ReviewPolicy:
    return ReviewPolicy.from_json((folder / POLICY_FILE).read_text(encoding="utf-8"))


def load_review_policy(artifacts_dir: Path | None = None) -> ReviewPolicy:
    """The promoted categorizer's review policy, checked to belong to that model."""
    root = (artifacts_dir or get_settings().artifacts_dir) / SERVICE
    version = promoted_version(root)
    path = root / version / POLICY_FILE
    if not path.exists():
        raise PolicyError(f"the promoted categorizer {version} has no {POLICY_FILE}")
    policy = read_review_policy(root / version)
    if policy.model_version != version:
        raise PolicyError(f"{path} is for model {policy.model_version}, not {version}")
    return policy
