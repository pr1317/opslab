"""Working-time calendar.

Back-office SLAs are quoted in *working* hours, not wall-clock hours: a case
that arrives at 16:30 on a Friday and is closed at 09:30 on the Monday has
consumed one working hour, not 65.  Every duration in this toolkit is measured
through a calendar so that control charts and survival models see the same
clock the service credits are measured against.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Iterable, Set

__all__ = ["BusinessCalendar"]


class BusinessCalendar:
    """A Monday-to-Friday working calendar with configurable hours and holidays.

    Parameters
    ----------
    start_hour, end_hour:
        Bounds of the working day, as whole or fractional hours.
    working_weekdays:
        ``date.weekday()`` values that count as working days (default Mon-Fri).
    holidays:
        Dates excluded regardless of weekday.
    """

    def __init__(
        self,
        start_hour: float = 9.0,
        end_hour: float = 17.0,
        working_weekdays: Iterable[int] = (0, 1, 2, 3, 4),
        holidays: Iterable[date] = (),
    ) -> None:
        if not 0.0 <= start_hour < end_hour <= 24.0:
            raise ValueError("require 0 <= start_hour < end_hour <= 24")
        self.start_hour = float(start_hour)
        self.end_hour = float(end_hour)
        self.working_weekdays: Set[int] = set(working_weekdays)
        if not self.working_weekdays:
            raise ValueError("at least one working weekday is required")
        self.holidays: Set[date] = set(holidays)

    # -- day helpers -------------------------------------------------------
    @property
    def hours_per_day(self) -> float:
        """Length of a full working day, in hours."""
        return self.end_hour - self.start_hour

    def is_working_day(self, day: date) -> bool:
        """True when ``day`` is a working weekday and not a holiday."""
        return day.weekday() in self.working_weekdays and day not in self.holidays

    def _day_start(self, day: date) -> datetime:
        return datetime.combine(day, time()) + timedelta(hours=self.start_hour)

    def _day_end(self, day: date) -> datetime:
        return datetime.combine(day, time()) + timedelta(hours=self.end_hour)

    def next_working_moment(self, moment: datetime) -> datetime:
        """Snap ``moment`` forward to the next instant inside working hours."""
        current = moment
        for _ in range(4000):
            day = current.date()
            if self.is_working_day(day):
                if current < self._day_start(day):
                    return self._day_start(day)
                if current < self._day_end(day):
                    return current
            current = datetime.combine(day + timedelta(days=1), time())
        raise RuntimeError("no working day found within 4000 days")

    # -- arithmetic --------------------------------------------------------
    def add_working_hours(self, moment: datetime, hours: float) -> datetime:
        """Advance ``moment`` by ``hours`` of working time."""
        if hours < 0:
            raise ValueError("hours must be non-negative")
        current = self.next_working_moment(moment)
        remaining = float(hours)
        for _ in range(20000):
            if remaining <= 1e-12:
                return current
            day_end = self._day_end(current.date())
            available = (day_end - current).total_seconds() / 3600.0
            if remaining < available:
                return current + timedelta(hours=remaining)
            remaining -= available
            current = self.next_working_moment(
                datetime.combine(current.date() + timedelta(days=1), time())
            )
        raise RuntimeError("add_working_hours() exceeded its iteration budget")

    def working_hours_between(self, start: datetime, end: datetime) -> float:
        """Working hours elapsed between two instants (0.0 when ``end <= start``)."""
        if end <= start:
            return 0.0
        total = 0.0
        day = start.date()
        last_day = end.date()
        while day <= last_day:
            if self.is_working_day(day):
                window_start = max(start, self._day_start(day))
                window_end = min(end, self._day_end(day))
                if window_end > window_start:
                    total += (window_end - window_start).total_seconds() / 3600.0
            day += timedelta(days=1)
        return total
