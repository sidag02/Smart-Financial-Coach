"""The fixed calendar a dataset covers: day arrays, months, business days and US bank holidays."""

from dataclasses import dataclass, field
from datetime import date

import holidays
import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Timeline:
    """Every day from `start` to `end` inclusive, addressed by integer index from `start`."""

    start: date
    end: date
    days: npt.NDArray[np.datetime64] = field(init=False, repr=False)
    month_idx: IntArray = field(init=False, repr=False)  # 0 = month containing `start`
    moy: IntArray = field(init=False, repr=False)  # month of year, 1..12
    dom: IntArray = field(init=False, repr=False)  # day of month, 1..31
    dow: IntArray = field(init=False, repr=False)  # 0 = Monday
    is_business: BoolArray = field(init=False, repr=False)
    month_first: IntArray = field(init=False, repr=False)  # first day index of each month
    month_last: IntArray = field(init=False, repr=False)  # last day index of each month

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("timeline end must be after start")
        days = np.arange(
            np.datetime64(self.start, "D"), np.datetime64(self.end, "D") + 1, dtype="datetime64[D]"
        )
        months = days.astype("datetime64[M]")
        month_idx = (months - months[0]).astype(np.int64)
        moy = months.astype(np.int64) % 12 + 1
        dom = (days - months.astype("datetime64[D]")).astype(np.int64) + 1
        dow = (days.astype(np.int64) + 3) % 7  # 1970-01-01 was a Thursday
        # US federal holidays (with observed dates) are the days the Federal Reserve is closed.
        bank_holidays = holidays.country_holidays(
            "US", years=range(self.start.year, self.end.year + 1)
        )
        holiday_days = np.array(sorted(bank_holidays), dtype="datetime64[D]")
        is_business = (dow < 5) & ~np.isin(days, holiday_days)
        n_months = int(month_idx[-1]) + 1
        month_first = np.searchsorted(month_idx, np.arange(n_months), side="left").astype(np.int64)
        month_last = np.searchsorted(month_idx, np.arange(n_months), side="right").astype(np.int64)
        for name, value in (
            ("days", days),
            ("month_idx", month_idx),
            ("moy", moy),
            ("dom", dom),
            ("dow", dow),
            ("is_business", is_business),
            ("month_first", month_first),
            ("month_last", month_last - 1),
        ):
            object.__setattr__(self, name, value)

    @property
    def n_days(self) -> int:
        return len(self.days)

    @property
    def n_months(self) -> int:
        return len(self.month_first)

    @property
    def years(self) -> float:
        return self.n_days / 365.25

    def date_of(self, day: int) -> date:
        result: date = self.days[day].astype(object)
        return result

    def previous_business_day(self, day_idx: IntArray) -> IntArray:
        """Move each day back to the nearest business day on or before it (clamped at day 0)."""
        business = np.flatnonzero(self.is_business)
        pos = np.searchsorted(business, day_idx, side="right") - 1
        return np.where(pos >= 0, business[np.clip(pos, 0, None)], day_idx).astype(np.int64)

    def month_day(self, month: int, day_of_month: int) -> int:
        """Day index of `day_of_month` in `month`, clamped to the month end; -1 if before start."""
        first = int(self.month_first[month])
        last = int(self.month_last[month])
        offset = day_of_month - int(self.dom[first])
        if offset < 0:
            return -1
        return min(first + offset, last)

    def last_business_day(self, month: int) -> int:
        return int(self.previous_business_day(np.array([self.month_last[month]]))[0])
