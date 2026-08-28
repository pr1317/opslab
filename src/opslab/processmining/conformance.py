"""Declarative conformance checking.

Rather than replay tokens against a hand-drawn Petri net - which few operations
teams have, and fewer keep current - conformance here is expressed as a set of
rules the process is *supposed* to satisfy.  Each rule is independently
checkable, names the cases that break it, and maps directly onto a control an
auditor or regulator would recognise.

:class:`SegregationOfDuties` is the one that tends to matter most in financial
services: it catches cases where the person who calculated a benefit also
signed it off.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from opslab.eventlog import Event, EventLog

__all__ = [
    "Violation",
    "ProcessRule",
    "RequiredActivities",
    "ForbiddenTransition",
    "Ordering",
    "MaxOccurrences",
    "StartActivities",
    "EndActivities",
    "SegregationOfDuties",
    "ConformanceReport",
    "check_conformance",
]


@dataclass(frozen=True)
class Violation:
    """One rule broken by one case."""

    case_id: str
    rule: str
    detail: str


class ProcessRule:
    """Base class for conformance rules."""

    #: Short stable identifier used in reports.
    code = "RULE"

    @property
    def description(self) -> str:
        """Human-readable statement of what the rule requires."""
        raise NotImplementedError

    def check(self, case_id: str, events: Sequence[Event]) -> List[Violation]:
        """Return the violations this case commits against the rule."""
        raise NotImplementedError


@dataclass
class RequiredActivities(ProcessRule):
    """Every case must contain each of the named activities."""

    activities: Tuple[str, ...]
    code: str = "REQUIRED"

    @property
    def description(self) -> str:
        return "every case must include: %s" % ", ".join(self.activities)

    def check(self, case_id: str, events: Sequence[Event]) -> List[Violation]:
        present = {e.activity for e in events}
        missing = [a for a in self.activities if a not in present]
        if not missing:
            return []
        return [Violation(case_id, self.code, "missing %s" % ", ".join(missing))]


@dataclass
class ForbiddenTransition(ProcessRule):
    """A given activity must never be directly followed by another."""

    source: str
    target: str
    code: str = "FORBIDDEN"

    @property
    def description(self) -> str:
        return "'%s' must never be directly followed by '%s'" % (self.source, self.target)

    def check(self, case_id: str, events: Sequence[Event]) -> List[Violation]:
        violations = []
        for index, (previous, current) in enumerate(zip(events, events[1:])):
            if previous.activity == self.source and current.activity == self.target:
                violations.append(
                    Violation(
                        case_id,
                        self.code,
                        "'%s' -> '%s' at position %d" % (self.source, self.target, index + 1),
                    )
                )
        return violations


@dataclass
class Ordering(ProcessRule):
    """One activity must not occur before another has occurred."""

    before: str
    after: str
    code: str = "ORDERING"

    @property
    def description(self) -> str:
        return "'%s' must not occur before '%s'" % (self.after, self.before)

    def check(self, case_id: str, events: Sequence[Event]) -> List[Violation]:
        seen_before = False
        for index, event in enumerate(events):
            if event.activity == self.before:
                seen_before = True
            elif event.activity == self.after and not seen_before:
                return [
                    Violation(
                        case_id,
                        self.code,
                        "'%s' occurred at position %d without a preceding '%s'"
                        % (self.after, index, self.before),
                    )
                ]
        return []


@dataclass
class MaxOccurrences(ProcessRule):
    """An activity may occur at most ``limit`` times in a case."""

    activity: str
    limit: int
    code: str = "MAX_OCCURRENCES"

    @property
    def description(self) -> str:
        return "'%s' may occur at most %d time(s) per case" % (self.activity, self.limit)

    def check(self, case_id: str, events: Sequence[Event]) -> List[Violation]:
        count = sum(1 for e in events if e.activity == self.activity)
        if count <= self.limit:
            return []
        return [
            Violation(
                case_id, self.code, "'%s' occurred %d times" % (self.activity, count)
            )
        ]


@dataclass
class StartActivities(ProcessRule):
    """Cases must start with one of the permitted activities."""

    allowed: Tuple[str, ...]
    code: str = "START"

    @property
    def description(self) -> str:
        return "cases must start with one of: %s" % ", ".join(self.allowed)

    def check(self, case_id: str, events: Sequence[Event]) -> List[Violation]:
        if not events or events[0].activity in self.allowed:
            return []
        return [Violation(case_id, self.code, "started with '%s'" % events[0].activity)]


@dataclass
class EndActivities(ProcessRule):
    """Completed cases must end with one of the permitted activities."""

    allowed: Tuple[str, ...]
    code: str = "END"

    @property
    def description(self) -> str:
        return "cases must end with one of: %s" % ", ".join(self.allowed)

    def check(self, case_id: str, events: Sequence[Event]) -> List[Violation]:
        if not events or events[-1].activity in self.allowed:
            return []
        return [Violation(case_id, self.code, "ended with '%s'" % events[-1].activity)]


@dataclass
class SegregationOfDuties(ProcessRule):
    """The checker of a step must not be the person who performed it.

    The four-eyes principle, checked against the log rather than against
    policy documentation.  ``checker`` is compared to the resource on the most
    recent preceding ``maker`` activity, which is how the control is actually
    defined: the reviewer must differ from whoever did the work being reviewed.
    """

    maker_activities: Tuple[str, ...]
    checker_activity: str
    code: str = "FOUR_EYES"

    @property
    def description(self) -> str:
        return "'%s' must be performed by someone other than whoever performed %s" % (
            self.checker_activity,
            " or ".join("'%s'" % a for a in self.maker_activities),
        )

    def check(self, case_id: str, events: Sequence[Event]) -> List[Violation]:
        violations: List[Violation] = []
        last_maker = None
        for event in events:
            if event.activity in self.maker_activities:
                last_maker = event
            elif event.activity == self.checker_activity and last_maker is not None:
                if event.resource and event.resource == last_maker.resource:
                    violations.append(
                        Violation(
                            case_id,
                            self.code,
                            "%s both performed '%s' and checked it"
                            % (event.resource, last_maker.activity),
                        )
                    )
        return violations


@dataclass
class ConformanceReport:
    """The outcome of checking a log against a rule set."""

    total_cases: int
    violations: List[Violation] = field(default_factory=list)
    rules: List[str] = field(default_factory=list)

    @property
    def violating_cases(self) -> int:
        """Number of distinct cases with at least one violation."""
        return len({v.case_id for v in self.violations})

    @property
    def fitness(self) -> float:
        """Share of cases that satisfied every rule."""
        if self.total_cases == 0:
            return 1.0
        return 1.0 - self.violating_cases / self.total_cases

    def by_rule(self) -> Dict[str, int]:
        """Violation counts per rule code."""
        return dict(Counter(v.rule for v in self.violations))

    def cases_for(self, rule_code: str) -> List[str]:
        """Distinct case ids that broke a given rule."""
        return sorted({v.case_id for v in self.violations if v.rule == rule_code})

    def to_text(self, examples: int = 3) -> str:
        """A readable summary with a few example violations per rule."""
        lines = [
            "Conformance: %.2f%% of %d cases fully conformant"
            % (100.0 * self.fitness, self.total_cases)
        ]
        counts = self.by_rule()
        if not counts:
            lines.append("  no violations")
            return "\n".join(lines)
        for code in sorted(counts, key=lambda c: -counts[c]):
            affected = len(self.cases_for(code))
            lines.append(
                "  %-16s %5d violation(s) across %d case(s)" % (code, counts[code], affected)
            )
            shown = [v for v in self.violations if v.rule == code][:examples]
            for violation in shown:
                lines.append("      %s: %s" % (violation.case_id, violation.detail))
        return "\n".join(lines)


def check_conformance(log: EventLog, rules: Iterable[ProcessRule]) -> ConformanceReport:
    """Check every case in ``log`` against every rule."""
    rule_list = list(rules)
    grouped = log.by_case()
    violations: List[Violation] = []
    for case_id, events in sorted(grouped.items()):
        for rule in rule_list:
            violations.extend(rule.check(case_id, events))
    return ConformanceReport(
        total_cases=len(grouped),
        violations=violations,
        rules=[r.description for r in rule_list],
    )
