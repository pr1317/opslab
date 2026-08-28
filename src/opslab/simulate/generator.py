"""Generate a life-and-pensions back-office event log with a known ground truth.

Real operational logs cannot be published, and a toolkit whose outputs nobody
can check is worth very little.  This generator sidesteps both problems: case
durations are drawn from a Weibull proportional-hazards model whose
coefficients are stated in :data:`GROUND_TRUTH_BETA`, so the estimates the
:mod:`opslab.sla` module produces can be compared against the values that
actually generated the data.  The same mechanism injects a dated special-cause
shift, which gives :mod:`opslab.spc` something real to detect.

The generated log deliberately contains the awkward features that make
operational mining interesting and that clean textbook logs lack: rework
loops, four-eyes violations, out-of-hours gaps, and cases still open at the
extract date (right censoring).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Sequence, Tuple

from opslab.calendar import BusinessCalendar
from opslab.eventlog import Case, CaseTable, Event, EventLog

__all__ = [
    "GROUND_TRUTH_BETA",
    "SimulationConfig",
    "SimulationResult",
    "simulate",
]


#: Coefficients of the data-generating Weibull proportional-hazards model.
#:
#: These act on the hazard of *resolution*, so a positive coefficient means the
#: case clears faster.  A Cox model fitted to the generated data should recover
#: each of these within roughly two standard errors.
GROUND_TRUTH_BETA: Dict[str, float] = {
    "priority_urgent": 0.55,
    "channel_post": -0.45,
    "complexity": -1.10,
    "backlog_index": -0.85,
    "awaiting_third_party": -1.90,
}

#: Weibull shape of the generating model (>1 means hazard rises with age).
GROUND_TRUTH_SHAPE = 1.15

#: Weibull scale of the generating model, in working hours.
GROUND_TRUTH_SCALE = 24.0

#: Activities whose presence in a trace means the case is waiting on someone
#: outside the operation.  This is an *observed* covariate rather than hidden
#: heterogeneity, which is what keeps the proportional-hazards fit honest: a
#: latent long-runner class would bias the Cox coefficients towards zero.
THIRD_PARTY_ACTIVITIES = ("Await Customer", "Escalate to Technical")


@dataclass(frozen=True)
class _ReworkRule:
    """A loop inserted after an activity with some probability."""

    activities: Tuple[str, ...]
    probability: float
    max_repeats: int = 2


@dataclass(frozen=True)
class _ProcessSpec:
    """The reference process for one case type."""

    name: str
    happy_path: Tuple[str, ...]
    sla_hours: float
    #: Relative service-time weight per activity; unlisted activities weigh 1.0.
    weights: Dict[str, float] = field(default_factory=dict)
    rework: Dict[str, _ReworkRule] = field(default_factory=dict)
    share: float = 1.0

    def weight_of(self, activity: str) -> float:
        """Service-time weight for one activity."""
        return self.weights.get(activity, 1.0)


_PROCESSES: Tuple[_ProcessSpec, ...] = (
    _ProcessSpec(
        name="Retirement Quote",
        happy_path=(
            "Receive Request", "Validate Details", "Index Documents",
            "Calculate Benefit", "Peer Check", "Issue Quote", "Close Case",
        ),
        sla_hours=64.0,
        weights={"Calculate Benefit": 2.4, "Peer Check": 0.8, "Close Case": 0.2},
        rework={
            "Validate Details": _ReworkRule(("Request Information", "Await Customer"), 0.22),
            "Peer Check": _ReworkRule(("Rework Calculation", "Peer Check"), 0.15, 2),
        },
        share=0.34,
    ),
    _ProcessSpec(
        name="Transfer Out",
        happy_path=(
            "Receive Request", "Validate Details", "Index Documents",
            "Verify Receiving Scheme", "Calculate Benefit", "Peer Check",
            "Release Payment", "Close Case",
        ),
        sla_hours=96.0,
        weights={
            "Verify Receiving Scheme": 2.0,
            "Calculate Benefit": 2.2,
            "Release Payment": 1.4,
            "Close Case": 0.2,
        },
        rework={
            "Validate Details": _ReworkRule(("Request Information", "Await Customer"), 0.30),
            "Verify Receiving Scheme": _ReworkRule(("Escalate to Technical",), 0.18, 1),
            "Peer Check": _ReworkRule(("Rework Calculation", "Peer Check"), 0.20, 2),
        },
        share=0.22,
    ),
    _ProcessSpec(
        name="Death Claim",
        happy_path=(
            "Receive Request", "Validate Details", "Index Documents",
            "Verify Documents", "Calculate Benefit", "Peer Check",
            "Release Payment", "Close Case",
        ),
        sla_hours=128.0,
        weights={"Verify Documents": 2.6, "Calculate Benefit": 2.0, "Close Case": 0.2},
        rework={
            "Verify Documents": _ReworkRule(("Request Information", "Await Customer"), 0.35, 2),
            "Peer Check": _ReworkRule(("Rework Calculation", "Peer Check"), 0.12, 1),
        },
        share=0.14,
    ),
    _ProcessSpec(
        name="Contribution Correction",
        happy_path=(
            "Receive Request", "Validate Details", "Reconcile Contributions",
            "Apply Adjustment", "Peer Check", "Close Case",
        ),
        sla_hours=52.0,
        weights={"Reconcile Contributions": 2.8, "Close Case": 0.2},
        rework={
            "Reconcile Contributions": _ReworkRule(("Escalate to Technical",), 0.16, 1),
            "Peer Check": _ReworkRule(("Rework Calculation", "Peer Check"), 0.18, 2),
        },
        share=0.20,
    ),
    _ProcessSpec(
        name="Complaint",
        happy_path=(
            "Receive Request", "Acknowledge Complaint", "Investigate",
            "Draft Response", "Peer Check", "Issue Response", "Close Case",
        ),
        sla_hours=76.0,
        weights={"Investigate": 3.0, "Close Case": 0.2},
        rework={
            "Investigate": _ReworkRule(("Request Information", "Await Customer"), 0.28, 2),
        },
        share=0.10,
    ),
)

_TEAMS: Dict[str, Tuple[str, ...]] = {
    "Benefits": tuple("BEN-%02d" % i for i in range(1, 13)),
    "Transfers": tuple("TRF-%02d" % i for i in range(1, 10)),
    "Claims": tuple("CLM-%02d" % i for i in range(1, 8)),
    "Contributions": tuple("CTR-%02d" % i for i in range(1, 9)),
    "Complaints": tuple("CMP-%02d" % i for i in range(1, 6)),
}

_PROCESS_TEAM = {
    "Retirement Quote": "Benefits",
    "Transfer Out": "Transfers",
    "Death Claim": "Claims",
    "Contribution Correction": "Contributions",
    "Complaint": "Complaints",
}

#: Activities that must be performed by someone other than the calculator.
FOUR_EYES_ACTIVITY = "Peer Check"
FOUR_EYES_PRECEDING = ("Calculate Benefit", "Apply Adjustment", "Draft Response")


@dataclass
class SimulationConfig:
    """Knobs for :func:`simulate`."""

    n_cases: int = 1200
    seed: int = 20260828
    start_date: date = date(2026, 1, 5)
    #: Cases still open at this instant are right-censored.
    extract_date: date = date(2026, 6, 30)
    #: Date a backlog surge begins; drives the special cause SPC should find.
    shift_date: date = date(2026, 4, 13)
    #: Multiplier applied to the backlog index from ``shift_date`` onwards.
    shift_strength: float = 1.8
    #: Probability that a peer check is done by the same person who calculated.
    four_eyes_violation_rate: float = 0.04
    calendar: BusinessCalendar = field(default_factory=BusinessCalendar)


@dataclass
class SimulationResult:
    """Everything one simulation run produced."""

    log: EventLog
    cases: CaseTable
    config: SimulationConfig

    @property
    def censoring_rate(self) -> float:
        """Fraction of cases still open at the extract date."""
        if not len(self.cases):
            return 0.0
        open_cases = sum(1 for c in self.cases if not c.resolved)
        return open_cases / len(self.cases)


def _pick_process(rng: random.Random) -> _ProcessSpec:
    """Draw a case type according to the configured mix."""
    total = math.fsum(spec.share for spec in _PROCESSES)
    threshold = rng.random() * total
    cumulative = 0.0
    for spec in _PROCESSES:
        cumulative += spec.share
        if threshold <= cumulative:
            return spec
    return _PROCESSES[-1]


def _build_trace(rng: random.Random, spec: _ProcessSpec) -> List[str]:
    """Expand the happy path into a concrete trace, inserting rework loops."""
    trace: List[str] = []
    for activity in spec.happy_path:
        trace.append(activity)
        rule = spec.rework.get(activity)
        if rule is not None and rng.random() < rule.probability:
            repeats = rng.randint(1, max(1, rule.max_repeats))
            for _ in range(repeats):
                trace.extend(rule.activities)
    return trace


def _backlog_index(day: date, config: SimulationConfig, rng: random.Random) -> float:
    """A 0-1 load signal that steps up on the configured shift date.

    Written so a single mechanism drives both the survival covariate and the
    special cause the control charts are meant to flag.
    """
    seasonal = 0.30 + 0.10 * math.sin(day.timetuple().tm_yday / 58.0)
    noise = rng.gauss(0.0, 0.05)
    level = seasonal + noise
    if day >= config.shift_date:
        level *= config.shift_strength
    return max(0.0, min(1.0, level))


def _weibull_duration(rng: random.Random, linear_predictor: float) -> float:
    """Inverse-transform sample from the ground-truth Weibull PH model."""
    uniform = rng.random()
    # S(t) = exp(-(t / scale)^shape * exp(lp))  =>  invert for t.
    numerator = -math.log(max(uniform, 1e-12)) / math.exp(linear_predictor)
    return GROUND_TRUTH_SCALE * numerator ** (1.0 / GROUND_TRUTH_SHAPE)


def _assign_resources(
    rng: random.Random, trace: Sequence[str], team: str, violation_rate: float
) -> List[str]:
    """Pick a worker per activity, honouring four eyes most of the time."""
    pool = _TEAMS[team]
    resources: List[str] = []
    last_maker = ""
    for activity in trace:
        if activity == FOUR_EYES_ACTIVITY and last_maker:
            if rng.random() < violation_rate:
                chosen = last_maker
            else:
                alternatives = [r for r in pool if r != last_maker]
                chosen = rng.choice(alternatives) if alternatives else last_maker
        else:
            chosen = rng.choice(pool)
        if activity in FOUR_EYES_PRECEDING:
            last_maker = chosen
        resources.append(chosen)
    return resources


def simulate(config: SimulationConfig = None) -> SimulationResult:
    """Generate an event log and matching case table.

    The returned durations are working hours measured through
    ``config.calendar``, and cases unfinished at ``config.extract_date`` are
    marked unresolved with the time observed so far - exactly the censoring
    pattern a live operational extract has.
    """
    config = config or SimulationConfig()
    if config.n_cases <= 0:
        raise ValueError("n_cases must be positive")
    if config.extract_date <= config.start_date:
        raise ValueError("extract_date must fall after start_date")

    rng = random.Random(config.seed)
    calendar = config.calendar
    extract_moment = datetime.combine(config.extract_date, datetime.min.time()) + timedelta(
        hours=calendar.end_hour
    )

    events: List[Event] = []
    cases: List[Case] = []
    span_days = (config.extract_date - config.start_date).days

    for index in range(config.n_cases):
        case_id = "C%05d" % (index + 1)
        spec = _pick_process(rng)

        # -- arrival, snapped into working hours ---------------------------
        offset_days = rng.uniform(0.0, float(span_days))
        arrival_day = config.start_date + timedelta(days=int(offset_days))
        arrival = calendar.next_working_moment(
            datetime.combine(arrival_day, datetime.min.time())
            + timedelta(hours=calendar.start_hour + rng.uniform(0.0, calendar.hours_per_day))
        )

        # -- covariates ----------------------------------------------------
        priority_urgent = 1 if rng.random() < 0.23 else 0
        channel = rng.choices(("Portal", "Post", "Phone"), weights=(0.52, 0.28, 0.20))[0]
        complexity = round(min(1.0, max(0.0, rng.betavariate(2.2, 3.0))), 4)
        backlog = round(_backlog_index(arrival.date(), config, rng), 4)

        # The trace is drawn first: whether the case ends up waiting on a third
        # party is a property of the path it takes, and it is also one of the
        # covariates driving how long the case runs.
        trace = _build_trace(rng, spec)
        third_party = 1 if any(a in trace for a in THIRD_PARTY_ACTIVITIES) else 0

        linear_predictor = (
            GROUND_TRUTH_BETA["priority_urgent"] * priority_urgent
            + GROUND_TRUTH_BETA["channel_post"] * (1 if channel == "Post" else 0)
            + GROUND_TRUTH_BETA["complexity"] * complexity
            + GROUND_TRUTH_BETA["backlog_index"] * backlog
            + GROUND_TRUTH_BETA["awaiting_third_party"] * third_party
        )
        true_duration = _weibull_duration(rng, linear_predictor)

        team = _PROCESS_TEAM[spec.name]
        resources = _assign_resources(
            rng, trace, team, config.four_eyes_violation_rate
        )
        weights = [spec.weight_of(a) * rng.uniform(0.6, 1.4) for a in trace]
        weight_total = math.fsum(weights) or 1.0

        attributes = {
            "case_type": spec.name,
            "team": team,
            "priority": "Urgent" if priority_urgent else "Standard",
            "channel": channel,
            "complexity": "%.4f" % complexity,
            "backlog_index": "%.4f" % backlog,
            "awaiting_third_party": str(third_party),
        }

        # Walk the trace, stopping at the extract date if the case runs past it.
        cursor = arrival
        resolved = True
        case_events: List[Event] = []
        for activity, resource, weight in zip(trace, resources, weights):
            step_hours = true_duration * (weight / weight_total)
            completion = calendar.add_working_hours(cursor, step_hours)
            if completion > extract_moment:
                resolved = False
                break
            case_events.append(
                Event(
                    case_id=case_id,
                    activity=activity,
                    timestamp=completion,
                    resource=resource,
                    attributes=dict(attributes),
                )
            )
            cursor = completion

        if not case_events:
            # Not even the first activity landed before the extract; the case
            # exists in the source system but has no rows in the event log.
            continue

        # A censored case contributes the time observed so far, not its
        # unobserved total - the distinction the whole SLA module turns on.
        elapsed = (
            true_duration
            if resolved
            else calendar.working_hours_between(arrival, extract_moment)
        )
        events.extend(case_events)

        cases.append(
            Case(
                case_id=case_id,
                arrived=arrival,
                duration_hours=round(elapsed, 4),
                resolved=resolved,
                sla_hours=spec.sla_hours * (0.75 if priority_urgent else 1.0),
                attributes=attributes,
            )
        )

    return SimulationResult(log=EventLog(events), cases=CaseTable(cases), config=config)
