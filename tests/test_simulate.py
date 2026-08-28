"""The generator's guarantees: reproducibility, consistency and censoring."""

from datetime import date, datetime

import pytest

from opslab.simulate import GROUND_TRUTH_BETA, SimulationConfig, simulate
from opslab.simulate.generator import THIRD_PARTY_ACTIVITIES


@pytest.fixture(scope="module")
def result():
    return simulate(SimulationConfig(n_cases=400, seed=99))


def test_the_same_seed_reproduces_the_same_log():
    first = simulate(SimulationConfig(n_cases=120, seed=5))
    second = simulate(SimulationConfig(n_cases=120, seed=5))
    assert [e.activity for e in first.log] == [e.activity for e in second.log]
    assert [c.duration_hours for c in first.cases] == [c.duration_hours for c in second.cases]


def test_different_seeds_produce_different_logs():
    first = simulate(SimulationConfig(n_cases=120, seed=5))
    second = simulate(SimulationConfig(n_cases=120, seed=6))
    assert [c.duration_hours for c in first.cases] != [c.duration_hours for c in second.cases]


def test_every_case_in_the_table_appears_in_the_log(result):
    logged = set(result.log.traces())
    assert {c.case_id for c in result.cases} == logged


def test_some_cases_are_still_open_at_the_extract(result):
    assert 0.0 < result.censoring_rate < 0.5
    assert any(not c.resolved for c in result.cases)


def test_events_stay_inside_working_hours(result):
    calendar = result.config.calendar
    for event in result.log:
        assert calendar.is_working_day(event.timestamp.date())
        hour = event.timestamp.hour + event.timestamp.minute / 60.0
        assert calendar.start_hour <= hour <= calendar.end_hour


def test_no_event_lands_after_the_extract_date(result):
    limit = datetime.combine(result.config.extract_date, datetime.min.time())
    for event in result.log:
        assert event.timestamp.date() <= limit.date()


def test_every_case_starts_by_being_received(result):
    for trace in result.log.traces().values():
        assert trace[0] == "Receive Request"


def test_the_third_party_flag_matches_the_trace(result):
    """The covariate must be observable in the log, not hidden heterogeneity."""
    traces = result.log.traces()
    for case in result.cases:
        expected = any(a in traces[case.case_id] for a in THIRD_PARTY_ACTIVITIES)
        # A censored case may not have reached its waiting step yet.
        if case.resolved:
            assert (case.attributes["awaiting_third_party"] == "1") == expected


def _mean(values):
    return sum(values) / len(values)


def test_the_backlog_surge_lengthens_cases():
    """Measured by completion date, the injected special cause is visible."""
    result = simulate(SimulationConfig(n_cases=1500, seed=11))
    calendar = result.config.calendar
    shift = result.config.shift_date

    before, after = [], []
    for case in result.cases:
        if not case.resolved:
            continue
        completed = calendar.add_working_hours(case.arrived, case.duration_hours).date()
        (after if completed >= shift else before).append(case.duration_hours)

    assert _mean(after) > _mean(before)


def test_bucketing_resolved_cases_by_arrival_hides_the_surge():
    """The reporting trap the SLA module exists to avoid.

    Selecting only cases that have finished and grouping them by *arrival*
    date drops the slow post-shift cases, because those are precisely the ones
    still open at the extract. The measured average then improves exactly when
    the process got worse.
    """
    result = simulate(SimulationConfig(n_cases=1500, seed=11))
    shift = result.config.shift_date
    before = [c.duration_hours for c in result.cases if c.resolved and c.arrived.date() < shift]
    after = [c.duration_hours for c in result.cases if c.resolved and c.arrived.date() >= shift]

    # The naive view reports an improvement that did not happen.
    assert _mean(after) < _mean(before)

    # The Kaplan-Meier estimator, which keeps the open cases, does not.
    from opslab.sla import kaplan_meier

    post = [c for c in result.cases if c.arrived.date() >= shift]
    pre = [c for c in result.cases if c.arrived.date() < shift]
    post_curve = kaplan_meier([c.duration_hours for c in post],
                              [1 if c.resolved else 0 for c in post])
    pre_curve = kaplan_meier([c.duration_hours for c in pre],
                             [1 if c.resolved else 0 for c in pre])
    horizon = 120.0
    assert post_curve.survival_at(horizon) > pre_curve.survival_at(horizon)


def test_four_eyes_violations_occur_at_roughly_the_configured_rate():
    from opslab.processmining import SegregationOfDuties, check_conformance

    config = SimulationConfig(n_cases=1500, seed=3, four_eyes_violation_rate=0.10)
    result = simulate(config)
    report = check_conformance(result.log, [
        SegregationOfDuties(("Calculate Benefit", "Apply Adjustment", "Draft Response"), "Peer Check")
    ])
    observed = len(report.cases_for("FOUR_EYES")) / len(result.cases)
    assert 0.05 < observed < 0.16


def test_urgent_cases_get_a_tighter_target(result):
    urgent = [c for c in result.cases if c.attributes["priority"] == "Urgent"]
    standard = [c for c in result.cases if c.attributes["priority"] == "Standard"]
    same_type = lambda cases, name: [c for c in cases if c.attributes["case_type"] == name]
    urgent_quotes = same_type(urgent, "Retirement Quote")
    standard_quotes = same_type(standard, "Retirement Quote")
    if urgent_quotes and standard_quotes:
        assert urgent_quotes[0].sla_hours < standard_quotes[0].sla_hours


def test_ground_truth_covers_every_design_column(result):
    from opslab.sla import design_matrix

    matrix = design_matrix(
        result.cases,
        numeric=["complexity", "backlog_index", "awaiting_third_party"],
        binary={"priority": "Urgent", "channel": "Post"},
    )
    assert set(matrix.names) == set(GROUND_TRUTH_BETA)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_cases": 0},
        {"extract_date": date(2025, 1, 1)},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        simulate(SimulationConfig(**kwargs))
