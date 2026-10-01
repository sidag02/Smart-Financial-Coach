from datetime import date

import numpy as np
import pytest

from smart_financial_coach.data.generator.timeline import Timeline


@pytest.fixture
def tl() -> Timeline:
    return Timeline(date(2025, 1, 15), date(2025, 12, 31))


def test_days_and_months(tl: Timeline) -> None:
    assert tl.n_days == 351
    assert tl.n_months == 12
    assert tl.date_of(0) == date(2025, 1, 15)
    assert tl.dow[(np.datetime64("2025-08-01") - np.datetime64("2025-01-15")).astype(int)] == 4


def test_bank_holidays_and_weekends_are_not_business_days(tl: Timeline) -> None:
    def idx(d: str) -> int:
        return int((np.datetime64(d) - np.datetime64("2025-01-15")).astype(int))

    assert not tl.is_business[idx("2025-07-04")]  # Independence Day, a Friday
    assert not tl.is_business[idx("2025-08-02")]  # Saturday
    assert tl.is_business[idx("2025-08-01")]
    shifted = tl.previous_business_day(np.array([idx("2025-07-04"), idx("2025-08-03")]))
    assert [str(tl.date_of(int(d))) for d in shifted] == ["2025-07-03", "2025-08-01"]


def test_month_day_clamps_and_respects_partial_first_month(tl: Timeline) -> None:
    feb = 1
    assert tl.date_of(tl.month_day(feb, 30)) == date(2025, 2, 28)
    assert tl.month_day(0, 1) == -1  # Jan 1 is before the timeline starts
    assert tl.date_of(tl.month_day(0, 20)) == date(2025, 1, 20)


def test_end_must_follow_start() -> None:
    with pytest.raises(ValueError, match="after start"):
        Timeline(date(2025, 1, 1), date(2025, 1, 1))
