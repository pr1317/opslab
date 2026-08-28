from datetime import date, datetime

import pytest

from opslab.calendar import BusinessCalendar


@pytest.fixture
def calendar():
    return BusinessCalendar(start_hour=9.0, end_hour=17.0)


def test_weekend_is_not_working_time(calendar):
    # Friday 16:30 to Monday 09:30 is one working hour, not 65.
    assert calendar.working_hours_between(
        datetime(2026, 8, 28, 16, 30), datetime(2026, 8, 31, 9, 30)
    ) == pytest.approx(1.0)


def test_a_full_working_week_is_forty_hours(calendar):
    assert calendar.working_hours_between(
        datetime(2026, 8, 24, 9, 0), datetime(2026, 8, 28, 17, 0)
    ) == pytest.approx(40.0)


def test_add_working_hours_skips_the_weekend(calendar):
    assert calendar.add_working_hours(datetime(2026, 8, 28, 16, 30), 1.0) == datetime(
        2026, 8, 31, 9, 30
    )


def test_add_and_measure_are_inverse(calendar):
    start = datetime(2026, 3, 4, 11, 15)
    for hours in (0.5, 3.0, 7.75, 40.0, 123.5):
        end = calendar.add_working_hours(start, hours)
        assert calendar.working_hours_between(start, end) == pytest.approx(hours, abs=1e-9)


def test_out_of_hours_start_snaps_forward(calendar):
    assert calendar.next_working_moment(datetime(2026, 8, 29, 3, 0)) == datetime(
        2026, 8, 31, 9, 0
    )


def test_holidays_are_excluded():
    calendar = BusinessCalendar(holidays=[date(2026, 8, 31)])
    assert not calendar.is_working_day(date(2026, 8, 31))
    assert calendar.working_hours_between(
        datetime(2026, 8, 28, 16, 30), datetime(2026, 9, 1, 9, 30)
    ) == pytest.approx(1.0)


def test_backwards_interval_is_zero(calendar):
    assert calendar.working_hours_between(
        datetime(2026, 8, 28, 16, 0), datetime(2026, 8, 28, 10, 0)
    ) == 0.0


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        BusinessCalendar(start_hour=17.0, end_hour=9.0)
    with pytest.raises(ValueError):
        BusinessCalendar(working_weekdays=[])
