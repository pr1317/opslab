"""Directly-follows discovery and variant analysis.

A directly-follows graph is the honest first model of a case-management
process: it claims only what the log actually shows, one activity following
another, and it carries the timing on the edges where the waiting really
happens.

Edge durations are measured through a :class:`~opslab.calendar.BusinessCalendar`
when one is supplied, so an overnight gap does not masquerade as a bottleneck.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from opslab.calendar import BusinessCalendar
from opslab.eventlog import EventLog
from opslab.numeric import mean, quantile

__all__ = [
    "NodeStats",
    "EdgeStats",
    "DirectlyFollowsGraph",
    "discover_dfg",
    "Variant",
    "variants",
]


@dataclass
class NodeStats:
    """How often an activity occurred, and in how many distinct cases."""

    activity: str
    frequency: int = 0
    case_frequency: int = 0

    @property
    def repetitions_per_case(self) -> float:
        """Mean occurrences per case that contains the activity."""
        if self.case_frequency == 0:
            return 0.0
        return self.frequency / self.case_frequency


@dataclass
class EdgeStats:
    """A directly-follows relation and the time spent traversing it."""

    source: str
    target: str
    frequency: int = 0
    durations: List[float] = field(default_factory=list)

    @property
    def mean_hours(self) -> float:
        """Mean working hours between the two activities."""
        return mean(self.durations) if self.durations else 0.0

    @property
    def median_hours(self) -> float:
        """Median working hours between the two activities."""
        return quantile(self.durations, 0.5) if self.durations else 0.0

    @property
    def p90_hours(self) -> float:
        """90th percentile working hours - where SLA damage accumulates."""
        return quantile(self.durations, 0.9) if self.durations else 0.0

    @property
    def total_hours(self) -> float:
        """Total working hours the process spent on this transition."""
        return sum(self.durations)


@dataclass
class DirectlyFollowsGraph:
    """The discovered model: activities, transitions, and their timings."""

    nodes: Dict[str, NodeStats] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], EdgeStats] = field(default_factory=dict)
    start_activities: Counter = field(default_factory=Counter)
    end_activities: Counter = field(default_factory=Counter)
    case_count: int = 0

    def filter_edges(self, min_frequency: int) -> "DirectlyFollowsGraph":
        """Drop rare transitions, keeping every node.

        Real logs have a long tail of one-off paths; hiding them is how a
        process map becomes readable without misstating the common flow.
        """
        kept = {k: v for k, v in self.edges.items() if v.frequency >= min_frequency}
        return DirectlyFollowsGraph(
            nodes=dict(self.nodes),
            edges=kept,
            start_activities=Counter(self.start_activities),
            end_activities=Counter(self.end_activities),
            case_count=self.case_count,
        )

    def successors(self, activity: str) -> List[str]:
        """Activities observed directly after ``activity``."""
        return sorted(t for (s, t) in self.edges if s == activity)

    def self_loops(self) -> List[EdgeStats]:
        """Transitions from an activity back to itself."""
        return [e for (s, t), e in self.edges.items() if s == t]

    def busiest_edges(self, limit: int = 10) -> List[EdgeStats]:
        """Edges ranked by total working hours consumed."""
        return sorted(self.edges.values(), key=lambda e: e.total_hours, reverse=True)[:limit]


def discover_dfg(
    log: EventLog,
    calendar: Optional[BusinessCalendar] = None,
) -> DirectlyFollowsGraph:
    """Build a directly-follows graph from an event log.

    When ``calendar`` is given, edge durations are working hours; otherwise
    they are elapsed wall-clock hours.
    """
    graph = DirectlyFollowsGraph()
    grouped = log.by_case()
    graph.case_count = len(grouped)

    node_cases: Dict[str, set] = defaultdict(set)
    for case_id, events in grouped.items():
        if not events:
            continue
        graph.start_activities[events[0].activity] += 1
        graph.end_activities[events[-1].activity] += 1

        for event in events:
            stats = graph.nodes.setdefault(event.activity, NodeStats(event.activity))
            stats.frequency += 1
            node_cases[event.activity].add(case_id)

        for previous, current in zip(events, events[1:]):
            key = (previous.activity, current.activity)
            edge = graph.edges.setdefault(key, EdgeStats(key[0], key[1]))
            edge.frequency += 1
            if calendar is not None:
                hours = calendar.working_hours_between(previous.timestamp, current.timestamp)
            else:
                hours = (current.timestamp - previous.timestamp).total_seconds() / 3600.0
            edge.durations.append(hours)

    for activity, cases in node_cases.items():
        graph.nodes[activity].case_frequency = len(cases)
    return graph


@dataclass
class Variant:
    """One distinct activity sequence and the cases that followed it."""

    trace: Tuple[str, ...]
    case_ids: List[str]

    @property
    def count(self) -> int:
        """How many cases followed this exact path."""
        return len(self.case_ids)

    @property
    def length(self) -> int:
        """Number of activities in the path."""
        return len(self.trace)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return " -> ".join(self.trace)


def variants(log: EventLog, limit: Optional[int] = None) -> List[Variant]:
    """Distinct traces, most frequent first.

    Variant analysis is usually the fastest route to a process problem: a
    handful of paths cover most cases, and the tail is where the rework,
    escalations and one-off workarounds live.
    """
    grouped: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for case_id, trace in log.traces().items():
        grouped[trace].append(case_id)
    ordered = sorted(
        (Variant(trace, sorted(cases)) for trace, cases in grouped.items()),
        key=lambda v: (-v.count, v.trace),
    )
    return ordered[:limit] if limit else ordered


def variant_coverage(variant_list: Sequence[Variant]) -> List[float]:
    """Cumulative share of cases covered as variants are added in order."""
    total = sum(v.count for v in variant_list)
    if total == 0:
        return []
    running = 0
    coverage = []
    for variant in variant_list:
        running += variant.count
        coverage.append(running / total)
    return coverage
