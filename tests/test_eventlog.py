"""Event log and case table round-trips."""

from datetime import datetime

import pytest

from opslab.eventlog import Case, CaseTable, Event, EventLog, parse_timestamp


def test_events_are_kept_in_timestamp_order():
    log = EventLog([
        Event("1", "B", datetime(2026, 1, 2, 9)),
        Event("1", "A", datetime(2026, 1, 1, 9)),
    ])
    assert [e.activity for e in log] == ["A", "B"]


def test_traces_group_activities_per_case():
    log = EventLog([
        Event("1", "A", datetime(2026, 1, 1, 9)),
        Event("2", "A", datetime(2026, 1, 1, 10)),
        Event("1", "B", datetime(2026, 1, 1, 11)),
    ])
    assert log.traces() == {"1": ("A", "B"), "2": ("A",)}


def test_filter_cases_narrows_the_log():
    log = EventLog([
        Event("1", "A", datetime(2026, 1, 1, 9)),
        Event("2", "A", datetime(2026, 1, 1, 10)),
    ])
    assert len(log.filter_cases(["1"])) == 1


def test_event_log_csv_round_trip(tmp_path):
    original = EventLog([
        Event("1", "A", datetime(2026, 1, 1, 9), "alice", {"team": "Ops", "priority": "High"}),
        Event("1", "B", datetime(2026, 1, 1, 10), "bob", {"team": "Ops", "priority": "High"}),
    ])
    path = str(tmp_path / "events.csv")
    original.to_csv(path)
    restored = EventLog.from_csv(path)
    assert len(restored) == 2
    assert restored[0].resource == "alice"
    assert restored[0].attributes["team"] == "Ops"
    assert restored[1].timestamp == datetime(2026, 1, 1, 10)


def test_case_table_csv_round_trip(tmp_path):
    original = CaseTable([
        Case("1", datetime(2026, 1, 1, 9), 12.5, True, 40.0, {"channel": "Post"}),
        Case("2", datetime(2026, 1, 2, 9), 8.0, False, 40.0, {"channel": "Portal"}),
    ])
    path = str(tmp_path / "cases.csv")
    original.to_csv(path)
    restored = CaseTable.from_csv(path)
    assert len(restored) == 2
    assert restored.get("1").resolved is True
    assert restored.get("2").resolved is False
    assert restored.get("1").duration_hours == pytest.approx(12.5)
    assert restored.get("2").attributes["channel"] == "Portal"


def test_from_csv_reports_a_missing_column(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("case_id,activity\n1,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="timestamp"):
        EventLog.from_csv(str(path))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2026-01-02 03:04:05", datetime(2026, 1, 2, 3, 4, 5)),
        ("2026-01-02T03:04:05", datetime(2026, 1, 2, 3, 4, 5)),
        ("2026-01-02 03:04", datetime(2026, 1, 2, 3, 4)),
        ("2026-01-02", datetime(2026, 1, 2)),
    ],
)
def test_timestamp_formats(text, expected):
    assert parse_timestamp(text) == expected


def test_an_unparseable_timestamp_is_rejected():
    with pytest.raises(ValueError, match="unrecognised timestamp"):
        parse_timestamp("02/01/2026")


def test_an_open_case_within_its_target_has_an_unknown_outcome():
    """The distinction the whole SLA module rests on."""
    still_running = Case("1", datetime(2026, 1, 1, 9), 10.0, False, 40.0)
    assert still_running.breached is None

    already_late = Case("2", datetime(2026, 1, 1, 9), 50.0, False, 40.0)
    assert already_late.breached is True

    delivered_on_time = Case("3", datetime(2026, 1, 1, 9), 10.0, True, 40.0)
    assert delivered_on_time.breached is False


def test_resolved_view_drops_open_cases():
    table = CaseTable([
        Case("1", datetime(2026, 1, 1, 9), 10.0, True, 40.0),
        Case("2", datetime(2026, 1, 1, 9), 10.0, False, 40.0),
    ])
    assert len(table.resolved()) == 1
