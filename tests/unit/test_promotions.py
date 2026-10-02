"""Committed promotions are consistent on every clone: PROMOTED matches the promotion log."""

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.intelligence.models.artifact import (
    URL_KEY,
    promotion_errors,
    promotions,
)

ARTIFACTS = PROJECT_ROOT / "artifacts"


def test_promoted_matches_promotion_log() -> None:
    services = sorted(p for p in ARTIFACTS.iterdir() if p.is_dir())

    assert [e for service in services for e in promotion_errors(service)] == []


def test_committed_promotions_are_downloadable() -> None:
    """Model files aren't committed, so a promotion without a URL can't be loaded elsewhere."""
    services = sorted(p for p in ARTIFACTS.iterdir() if p.is_dir())
    unpublished = [
        f"{service.name} {entry['version']}"
        for service in services
        for entry in promotions(service)
        if not entry.get(URL_KEY)
    ]

    assert unpublished == []
