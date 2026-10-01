"""Committed promotions are consistent on every clone: PROMOTED matches the promotion log."""

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.intelligence.models.artifact import promotion_errors

ARTIFACTS = PROJECT_ROOT / "artifacts"


def test_promoted_matches_promotion_log() -> None:
    services = sorted(p for p in ARTIFACTS.iterdir() if p.is_dir())

    assert [e for service in services for e in promotion_errors(service)] == []
