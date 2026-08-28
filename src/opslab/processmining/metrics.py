"""Operational metrics derived from an event log.

The questions these answer are the ones a process owner actually asks: where
does the time go, which steps get redone, and where does work change hands.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from opslab.calendar import BusinessCalendar
from opslab.eventlog import CaseTable, EventLog
from opslab.numeric import mean, quantile
from opslab.processmining.discovery import DirectlyFollowsGraph, discover_dfg

__all__ = [
    "ActivityStats",
    "activity_statistics",
    "rework_statistics",
    "ReworkStats",
    "bottlenecks",
    "HandoverEdge",
    "handover_network",
]


@dataclass
class ActivityStats:
    """Timing and volume for a single activity."""

    activity: str
    occurrences: int = 0
    cases: int = 0
    durations: List[float] = field(default_factory=list)

    @property
    def mean_hours(self) -> float:
        """Mean working hours from the previous event to this one."""
        return mean(self.durations) if self.durations else 0.0

    @property
    def median_hours(self) -> float:
        """Median working hours for this step."""
        return quantile(self.durations, 0.5) if self.durations else 0.0

    @property
    def p90_hours(self) -> float:
        """90th percentile working hours for this step."""
        return quantile(self.durations, 0.9) if self.durations else 0.0

    @property
    def total_hours(self) -> float:
        """Total working hours consumed by this activity across all cases."""
        return sum(self.durations)


def activity_statistics(
    log: EventLog,
    calendar: Optional[BusinessCalendar] = None,
    cases: Optional[CaseTable] = None,
) -> Dict[str, ActivityStats]:
    """Per-activity volumes and step durations.

    The duration charged to an activity is the time from the previous event in
    the case to that activity's completion - i.e. queueing plus handling, which
    is what the customer experiences.  When ``cases`` is supplied, the first
    activity in each case is measured from the case's arrival rather than
    treated as instantaneous.
    """
    stats: Dict[str, ActivityStats] = {}
    case_lookup = {c.case_id: c for c in cases} if cases is not None else {}

    def hours_between(start, end) -> float:
        if calendar is not None:
            return calendar.working_hours_between(start, end)
        return max(0.0, (end - start).total_seconds() / 3600.0)

    for case_id, events in log.by_case().items():
        seen: Counter = Counter()
        previous_time = None
        case = case_lookup.get(case_id)
        if case is not None:
            previous_time = case.arrived
        for event in events:
            entry = stats.setdefault(event.activity, ActivityStats(event.activity))
            entry.occurrences += 1
            seen[event.activity] += 1
            if previous_time is not None:
                entry.durations.append(hours_between(previous_time, event.timestamp))
            previous_time = event.timestamp
        for activity in seen:
            stats[activity].cases += 1
    return stats


@dataclass
class ReworkStats:
    """How often an activity is repeated within the same case."""

    activity: str
    cases_with_activity: int
    cases_with_repeat: int
    total_occurrences: int
    self_loops: int

    @property
    def rework_rate(self) -> float:
        """Share of cases in which the activity happened more than once."""
        if self.cases_with_activity == 0:
            return 0.0
        return self.cases_with_repeat / self.cases_with_activity

    @property
    def first_time_right(self) -> float:
        """Complement of the rework rate - the usual quality headline."""
        return 1.0 - self.rework_rate


def rework_statistics(log: EventLog) -> Dict[str, ReworkStats]:
    """Rework and first-time-right rates per activity."""
    occurrences: Counter = Counter()
    with_activity: Counter = Counter()
    with_repeat: Counter = Counter()
    self_loops: Counter = Counter()

    for events in log.by_case().values():
        counts = Counter(e.activity for e in events)
        for activity, count in counts.items():
            occurrences[activity] += count
            with_activity[activity] += 1
            if count > 1:
                with_repeat[activity] += 1
        for previous, current in zip(events, events[1:]):
            if previous.activity == current.activity:
                self_loops[current.activity] += 1

    return {
        activity: ReworkStats(
            activity=activity,
            cases_with_activity=with_activity[activity],
            cases_with_repeat=with_repeat[activity],
            total_occurrences=occurrences[activity],
            self_loops=self_loops[activity],
        )
        for activity in sorted(occurrences)
    }


def bottlenecks(
    log: EventLog,
    calendar: Optional[BusinessCalendar] = None,
    limit: int = 10,
    graph: Optional[DirectlyFollowsGraph] = None,
) -> List[Tuple[str, str, float, float, int]]:
    """Transitions ranked by the total working time they consume.

    Ranking by *total* rather than mean time is the point: a two-hour wait that
    happens on every case costs the operation far more than a two-week wait
    that happens twice, and only the first is worth a project.

    Returns tuples of ``(source, target, total_hours, mean_hours, frequency)``.
    """
    model = graph if graph is not None else discover_dfg(log, calendar)
    ranked = sorted(model.edges.values(), key=lambda e: e.total_hours, reverse=True)
    return [
        (e.source, e.target, e.total_hours, e.mean_hours, e.frequency)
        for e in ranked[:limit]
    ]


@dataclass
class HandoverEdge:
    """A transfer of work between two resources."""

    source: str
    target: str
    count: int = 0

    @property
    def is_self_handover(self) -> bool:
        """True when the same person did both steps."""
        return self.source == self.target


def handover_network(
    log: EventLog,
    include_self: bool = False,
) -> List[HandoverEdge]:
    """Count handovers of work between resources, busiest first.

    A dense handover network is a common and expensive finding: every transfer
    is a queue, a context reload and an opportunity to lose the thread.
    """
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for events in log.by_case().values():
        for previous, current in zip(events, events[1:]):
            if not previous.resource or not current.resource:
                continue
            if not include_self and previous.resource == current.resource:
                continue
            counts[(previous.resource, current.resource)] += 1
    edges = [HandoverEdge(s, t, c) for (s, t), c in counts.items()]
    return sorted(edges, key=lambda e: (-e.count, e.source, e.target))
