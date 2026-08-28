"""Event log and case table - the shared data model for the whole toolkit.

An :class:`EventLog` is the record of *what happened*: one row per activity
completion.  A :class:`CaseTable` is the record of *how each case ended*:
one row per case, carrying the attributes the survival models treat as
covariates and the censoring flag the SLA analysis depends on.

Both read and write plain CSV so the same files can be dropped straight into
Power BI, Excel or a database without an intermediate format.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

__all__ = ["Event", "EventLog", "Case", "CaseTable", "parse_timestamp"]

_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
)


def parse_timestamp(text: str) -> datetime:
    """Parse the timestamp formats the CSV readers accept."""
    raw = text.strip()
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    raise ValueError("unrecognised timestamp: %r" % text)


@dataclass(frozen=True)
class Event:
    """A single completed activity within a case."""

    case_id: str
    activity: str
    timestamp: datetime
    resource: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class Case:
    """One case, with its outcome and the attributes used as covariates."""

    case_id: str
    arrived: datetime
    #: Working hours consumed; for a censored case, hours observed so far.
    duration_hours: float
    #: True when the case reached a terminal activity inside the extract window.
    resolved: bool
    #: Working-hour SLA target the case was measured against.
    sla_hours: float
    attributes: Dict[str, str] = field(default_factory=dict)

    @property
    def breached(self) -> Optional[bool]:
        """Whether the SLA was breached, or ``None`` while still undecidable.

        An open case that has not yet passed its target has an unknown
        outcome - reporting it as compliant is the single most common way
        operational dashboards flatter themselves.
        """
        if self.duration_hours > self.sla_hours:
            return True
        return False if self.resolved else None


class EventLog:
    """An ordered collection of :class:`Event` records."""

    def __init__(self, events: Iterable[Event] = ()) -> None:
        self._events: List[Event] = sorted(events, key=lambda e: (e.timestamp, e.case_id))

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __getitem__(self, index: int) -> Event:
        return self._events[index]

    @property
    def events(self) -> List[Event]:
        """The events, in timestamp order."""
        return list(self._events)

    def add(self, event: Event) -> None:
        """Append an event, keeping the log sorted."""
        self._events.append(event)
        self._events.sort(key=lambda e: (e.timestamp, e.case_id))

    # -- views -------------------------------------------------------------
    def by_case(self) -> Dict[str, List[Event]]:
        """Group events by case, each group in timestamp order."""
        grouped: Dict[str, List[Event]] = defaultdict(list)
        for event in self._events:
            grouped[event.case_id].append(event)
        for events in grouped.values():
            events.sort(key=lambda e: e.timestamp)
        return dict(grouped)

    def traces(self) -> Dict[str, Tuple[str, ...]]:
        """Map each case to its activity sequence."""
        return {
            case_id: tuple(e.activity for e in events)
            for case_id, events in self.by_case().items()
        }

    def activities(self) -> List[str]:
        """Distinct activity names, sorted."""
        return sorted({e.activity for e in self._events})

    def resources(self) -> List[str]:
        """Distinct resource names, sorted."""
        return sorted({e.resource for e in self._events if e.resource})

    def filter_cases(self, case_ids: Iterable[str]) -> "EventLog":
        """A new log restricted to the given cases."""
        wanted = set(case_ids)
        return EventLog(e for e in self._events if e.case_id in wanted)

    # -- persistence -------------------------------------------------------
    #: Columns written before any extra attribute columns.
    CORE_COLUMNS: Sequence[str] = ("case_id", "activity", "timestamp", "resource")

    def to_csv(self, path: str) -> None:
        """Write the log to ``path``, flattening attributes into columns."""
        extra = sorted({key for e in self._events for key in e.attributes})
        columns = list(self.CORE_COLUMNS) + extra
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for event in self._events:
                row = {
                    "case_id": event.case_id,
                    "activity": event.activity,
                    "timestamp": event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "resource": event.resource,
                }
                row.update(event.attributes)
                writer.writerow(row)

    @classmethod
    def from_csv(
        cls,
        path: str,
        case_id: str = "case_id",
        activity: str = "activity",
        timestamp: str = "timestamp",
        resource: str = "resource",
    ) -> "EventLog":
        """Read a log from CSV; unmapped columns become event attributes."""
        events: List[Event] = []
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = {case_id, activity, timestamp} - set(reader.fieldnames or ())
            if missing:
                raise ValueError("missing required column(s): %s" % ", ".join(sorted(missing)))
            mapped = {case_id, activity, timestamp, resource}
            for row in reader:
                events.append(
                    Event(
                        case_id=row[case_id],
                        activity=row[activity],
                        timestamp=parse_timestamp(row[timestamp]),
                        resource=(row.get(resource) or ""),
                        attributes={k: v for k, v in row.items() if k not in mapped},
                    )
                )
        return cls(events)


class CaseTable:
    """A collection of :class:`Case` records keyed by case id."""

    def __init__(self, cases: Iterable[Case] = ()) -> None:
        self._cases: List[Case] = sorted(cases, key=lambda c: (c.arrived, c.case_id))

    def __len__(self) -> int:
        return len(self._cases)

    def __iter__(self) -> Iterator[Case]:
        return iter(self._cases)

    @property
    def cases(self) -> List[Case]:
        """The cases, in arrival order."""
        return list(self._cases)

    def get(self, case_id: str) -> Optional[Case]:
        """Look up a single case by id."""
        for case in self._cases:
            if case.case_id == case_id:
                return case
        return None

    def resolved(self) -> "CaseTable":
        """The subset of cases that completed inside the extract window."""
        return CaseTable(c for c in self._cases if c.resolved)

    CORE_COLUMNS: Sequence[str] = (
        "case_id", "arrived", "duration_hours", "resolved", "sla_hours",
    )

    def to_csv(self, path: str) -> None:
        """Write the case table to ``path``."""
        extra = sorted({key for c in self._cases for key in c.attributes})
        columns = list(self.CORE_COLUMNS) + extra
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for case in self._cases:
                row = {
                    "case_id": case.case_id,
                    "arrived": case.arrived.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_hours": "%.4f" % case.duration_hours,
                    "resolved": "1" if case.resolved else "0",
                    "sla_hours": "%.4f" % case.sla_hours,
                }
                row.update(case.attributes)
                writer.writerow(row)

    @classmethod
    def from_csv(cls, path: str) -> "CaseTable":
        """Read a case table written by :meth:`to_csv`."""
        cases: List[Case] = []
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(cls.CORE_COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError("missing required column(s): %s" % ", ".join(sorted(missing)))
            for row in reader:
                cases.append(
                    Case(
                        case_id=row["case_id"],
                        arrived=parse_timestamp(row["arrived"]),
                        duration_hours=float(row["duration_hours"]),
                        resolved=row["resolved"].strip() in {"1", "true", "True", "yes"},
                        sla_hours=float(row["sla_hours"]),
                        attributes={
                            k: v for k, v in row.items() if k not in set(cls.CORE_COLUMNS)
                        },
                    )
                )
        return cls(cases)
