"""Discovery, metrics and conformance checking."""

from datetime import datetime

import pytest

from opslab.calendar import BusinessCalendar
from opslab.eventlog import Event, EventLog
from opslab.processmining import (
    EndActivities,
    ForbiddenTransition,
    MaxOccurrences,
    Ordering,
    RequiredActivities,
    SegregationOfDuties,
    StartActivities,
    activity_statistics,
    bottlenecks,
    check_conformance,
    discover_dfg,
    handover_network,
    rework_statistics,
    to_dot,
    to_mermaid,
    variants,
)


def _event(case_id, activity, hour, resource=""):
    return Event(case_id, activity, datetime(2026, 3, 2, hour, 0), resource)


@pytest.fixture
def log():
    """Two cases on the happy path, one with a rework loop."""
    return EventLog([
        _event("1", "Receive", 9, "alice"),
        _event("1", "Assess", 10, "bob"),
        _event("1", "Check", 11, "carol"),
        _event("2", "Receive", 9, "alice"),
        _event("2", "Assess", 12, "bob"),
        _event("2", "Check", 13, "carol"),
        _event("3", "Receive", 9, "dave"),
        _event("3", "Assess", 10, "dave"),
        _event("3", "Rework", 11, "dave"),
        _event("3", "Assess", 12, "dave"),
        _event("3", "Check", 13, "dave"),
    ])


def test_discovery_counts_activities_and_transitions(log):
    graph = discover_dfg(log)
    assert graph.case_count == 3
    assert graph.nodes["Assess"].frequency == 4
    assert graph.nodes["Assess"].case_frequency == 3
    assert graph.edges[("Receive", "Assess")].frequency == 3
    assert graph.edges[("Rework", "Assess")].frequency == 1


def test_start_and_end_activities_are_recorded(log):
    graph = discover_dfg(log)
    assert dict(graph.start_activities) == {"Receive": 3}
    assert dict(graph.end_activities) == {"Check": 3}


def test_edge_durations_respect_the_working_calendar():
    calendar = BusinessCalendar(start_hour=9.0, end_hour=17.0)
    log = EventLog([
        Event("1", "A", datetime(2026, 8, 28, 16, 30)),
        Event("1", "B", datetime(2026, 8, 31, 9, 30)),
    ])
    with_calendar = discover_dfg(log, calendar).edges[("A", "B")].mean_hours
    without_calendar = discover_dfg(log).edges[("A", "B")].mean_hours
    assert with_calendar == pytest.approx(1.0)
    assert without_calendar == pytest.approx(65.0)


def test_filter_edges_drops_rare_transitions(log):
    graph = discover_dfg(log).filter_edges(min_frequency=3)
    assert ("Receive", "Assess") in graph.edges
    assert ("Rework", "Assess") not in graph.edges
    # Nodes survive so the map keeps its vocabulary.
    assert "Rework" in graph.nodes


def test_variants_are_ordered_by_frequency(log):
    result = variants(log)
    assert result[0].count == 2
    assert result[0].trace == ("Receive", "Assess", "Check")
    assert result[1].count == 1
    assert result[1].length == 5


def test_rework_statistics_identify_the_repeated_step(log):
    stats = rework_statistics(log)
    assert stats["Assess"].cases_with_repeat == 1
    assert stats["Assess"].rework_rate == pytest.approx(1 / 3)
    assert stats["Assess"].first_time_right == pytest.approx(2 / 3)
    assert stats["Receive"].rework_rate == 0.0


def test_self_loops_are_counted_separately():
    log = EventLog([_event("1", "A", 9), _event("1", "A", 10), _event("1", "B", 11)])
    assert rework_statistics(log)["A"].self_loops == 1


def test_bottlenecks_rank_by_total_not_mean_time(log):
    # Receive->Assess is 1h on two cases and 3h on one: 5h total over 3 traversals.
    ranked = bottlenecks(log, limit=1)
    source, target, total, mean_hours, frequency = ranked[0]
    assert (source, target) == ("Receive", "Assess")
    assert total == pytest.approx(5.0)
    assert frequency == 3


def test_activity_statistics_measure_the_step_into_each_activity(log):
    stats = activity_statistics(log)
    assert stats["Assess"].occurrences == 4
    assert stats["Assess"].cases == 3
    assert stats["Check"].mean_hours == pytest.approx(1.0)


def test_handover_network_excludes_self_handovers_by_default(log):
    edges = handover_network(log)
    assert all(not e.is_self_handover for e in edges)
    assert edges[0].source == "alice" and edges[0].target == "bob"
    assert edges[0].count == 2


def test_handover_network_can_include_self_handovers(log):
    edges = handover_network(log, include_self=True)
    assert any(e.is_self_handover for e in edges)


# -- conformance -----------------------------------------------------------
def test_required_activities_flags_the_missing_case(log):
    report = check_conformance(log, [RequiredActivities(("Receive", "Approve"))])
    assert report.violating_cases == 3
    assert report.fitness == pytest.approx(0.0)


def test_a_satisfied_rule_gives_full_fitness(log):
    report = check_conformance(log, [RequiredActivities(("Receive", "Check"))])
    assert report.fitness == pytest.approx(1.0)
    assert report.violations == []


def test_ordering_rule_catches_an_out_of_sequence_activity():
    log = EventLog([_event("1", "Check", 9), _event("1", "Assess", 10)])
    report = check_conformance(log, [Ordering("Assess", "Check")])
    assert report.by_rule() == {"ORDERING": 1}


def test_forbidden_transition_is_detected(log):
    report = check_conformance(log, [ForbiddenTransition("Rework", "Assess")])
    assert report.cases_for("FORBIDDEN") == ["3"]


def test_max_occurrences_counts_repeats(log):
    report = check_conformance(log, [MaxOccurrences("Assess", 1)])
    assert report.cases_for("MAX_OCCURRENCES") == ["3"]


def test_start_and_end_rules(log):
    report = check_conformance(
        log, [StartActivities(("Receive",)), EndActivities(("Close",))]
    )
    assert report.by_rule() == {"END": 3}


def test_segregation_of_duties_catches_the_self_check(log):
    rule = SegregationOfDuties(("Assess",), "Check")
    report = check_conformance(log, [rule])
    # Case 3 is worked end to end by dave; cases 1 and 2 pass to carol.
    assert report.cases_for("FOUR_EYES") == ["3"]
    assert "dave" in report.violations[0].detail


def test_segregation_of_duties_ignores_missing_resources():
    log = EventLog([_event("1", "Assess", 9), _event("1", "Check", 10)])
    report = check_conformance(log, [SegregationOfDuties(("Assess",), "Check")])
    assert report.violations == []


def test_conformance_report_renders_examples(log):
    report = check_conformance(log, [RequiredActivities(("Approve",))])
    text = report.to_text()
    assert "REQUIRED" in text and "missing Approve" in text


# -- rendering -------------------------------------------------------------
def test_dot_output_is_well_formed(log):
    dot = to_dot(discover_dfg(log))
    assert dot.startswith("digraph process {") and dot.rstrip().endswith("}")
    assert "Receive" in dot
    assert dot.count("->") >= len(discover_dfg(log).edges)


def test_mermaid_output_declares_a_flowchart(log):
    mermaid = to_mermaid(discover_dfg(log))
    assert mermaid.splitlines()[0] == "flowchart TD"
    assert "-->" in mermaid


def test_renderers_honour_the_edge_frequency_filter(log):
    sparse = to_mermaid(discover_dfg(log), min_edge_frequency=3)
    assert "Rework" in sparse  # the node survives even though its edges do not
    # Only Receive->Assess and Assess->Check occur on all three cases.
    assert sparse.count("-->") == 2
    assert "n3 -->" not in sparse and "--> n3" not in sparse
