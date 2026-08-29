"""The bundled sample, the SVG process map, and the HTML report over both."""

import os
import re
import xml.etree.ElementTree as ElementTree

import pytest

from opslab import data
from opslab.calendar import BusinessCalendar
from opslab.cli import main
from opslab.eventlog import CaseTable, EventLog
from opslab.processmining import discover_dfg, to_svg
from opslab.processmining.render import _back_edges, _layers, _wrap
from opslab.report import build_report


# -- the bundled sample ----------------------------------------------------
def test_every_sample_file_ships_with_the_package():
    for path in data.paths().values():
        assert os.path.exists(path), path


def test_the_sample_is_the_committed_output_of_the_documented_command():
    """The README quotes these figures, so a regenerated sample must not move them."""
    log = EventLog.from_csv(data.events_path())
    cases = CaseTable.from_csv(data.cases_path())
    assert len(cases) == 1483
    assert len(log) == 11460
    open_cases = [c for c in cases if not c.resolved]
    assert 0.09 < len(open_cases) / len(cases) < 0.12


def test_the_sample_still_carries_the_censoring_the_survival_module_needs():
    """Without open cases the Kaplan-Meier and naive medians would coincide."""
    cases = CaseTable.from_csv(data.cases_path())
    assert any(not case.resolved for case in cases)
    assert all(case.duration_hours > 0 for case in cases)


# -- process map layout ----------------------------------------------------
def test_back_edges_are_the_ones_that_close_a_loop():
    nodes = ["a", "b", "c"]
    edges = {("a", "b"), ("b", "c"), ("c", "a")}
    back = _back_edges(nodes, edges)
    assert len(back) == 1
    assert back < edges


def test_an_acyclic_graph_has_no_back_edges():
    assert _back_edges(["a", "b", "c"], {("a", "b"), ("b", "c"), ("a", "c")}) == set()


def test_layering_puts_every_node_below_its_inputs():
    nodes = ["a", "b", "c", "d"]
    edges = {("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")}
    layer = _layers(nodes, edges, back=set())
    assert layer["a"] == 0
    assert layer["b"] == layer["c"] == 1
    assert layer["d"] == 2


def test_layering_terminates_on_a_cyclic_graph_once_the_cycle_is_broken():
    nodes = ["a", "b", "c"]
    edges = {("a", "b"), ("b", "c"), ("c", "b")}
    back = _back_edges(nodes, edges)
    layer = _layers(nodes, edges, back)
    assert max(layer.values()) < len(nodes)


def test_long_activity_names_wrap_onto_two_lines_at_a_space():
    assert _wrap("Peer Check") == ["Peer Check"]
    assert _wrap("Reconcile Contributions") == ["Reconcile", "Contributions"]
    assert _wrap("Verify Receiving Scheme") == ["Verify Receiving", "Scheme"]
    # No space to break on, so it falls back to a hard split rather than overflowing.
    assert len(_wrap("A" * 40)) == 2


@pytest.fixture(scope="module")
def sample_graph():
    return discover_dfg(EventLog.from_csv(data.events_path()), BusinessCalendar())


def test_the_map_is_well_formed_xml(sample_graph):
    root = ElementTree.fromstring(to_svg(sample_graph, min_edge_frequency=5))
    assert root.tag.endswith("svg")


def test_the_map_draws_one_box_per_activity(sample_graph):
    svg = to_svg(sample_graph, min_edge_frequency=5)
    assert svg.count('class="mp-node') == len(sample_graph.nodes)


def test_only_the_activities_that_really_start_or_end_cases_are_marked(sample_graph):
    """Nearly every activity ends some case; marking them all would say nothing."""
    svg = to_svg(sample_graph, min_edge_frequency=5)
    assert svg.count('class="mp-node start"') == 1
    assert svg.count('class="mp-node end"') < len(sample_graph.nodes) / 2


def test_raising_the_edge_threshold_drops_transitions_but_keeps_activities(sample_graph):
    sparse = to_svg(sample_graph, min_edge_frequency=200)
    dense = to_svg(sample_graph, min_edge_frequency=1)
    assert sparse.count("mp-edge") < dense.count("mp-edge")
    assert sparse.count('class="mp-node') == dense.count('class="mp-node')


def test_an_empty_graph_renders_rather_than_raising():
    from opslab.processmining.discovery import DirectlyFollowsGraph

    assert "svg" in to_svg(DirectlyFollowsGraph())


# -- the HTML report -------------------------------------------------------
@pytest.fixture(scope="module")
def sample_report():
    return build_report(
        data.events_path(), data.cases_path(), data.model_path(),
        ground_truth={"complexity": -1.10},
    )


def test_the_report_covers_all_four_modules(sample_report):
    for anchor in ("mining", "spc", "sla", "daxlint"):
        assert 'id="%s"' % anchor in sample_report


def test_every_chart_in_the_report_is_well_formed_xml(sample_report):
    charts = re.findall(r"<svg.*?</svg>", sample_report, re.DOTALL)
    assert len(charts) >= 4
    for chart in charts:
        ElementTree.fromstring(chart)


def test_the_report_needs_no_network_to_render(sample_report):
    """A locked-down laptop opening this from a file:// URL must see everything."""
    for tag in ("<script", "<link", "<iframe", "@import", "url(http"):
        assert tag not in sample_report
    # The only URLs allowed are the XML namespace, which is an identifier rather
    # than something the browser fetches, and the link home in the footer.
    urls = set(re.findall(r'https?://[^"\s<)]+', sample_report))
    assert urls <= {"http://www.w3.org/2000/svg", "https://github.com/pr1317/opslab"}, urls


def test_the_report_states_the_generating_coefficient_when_it_is_known(sample_report):
    assert "-1.1000" in sample_report
    assert "generating value" in sample_report


def test_the_report_omits_the_truth_column_for_data_that_has_no_ground_truth():
    plain = build_report(
        data.events_path(), data.cases_path(), data.model_path(), ground_truth=None
    )
    assert "generating values fall inside" not in plain


def test_the_report_contrasts_the_naive_median_with_the_censored_one(sample_report):
    assert "median, closed cases only" in sample_report
    assert "median, Kaplan-Meier" in sample_report


# -- the try command -------------------------------------------------------
def test_try_writes_a_report_and_the_data_behind_it(tmp_path):
    out = tmp_path / "try"
    assert main(["try", "--out", str(out)]) == 0
    assert (out / "report.html").exists()
    assert (out / "sample_events.csv").exists()
    assert (out / "sample_cases.csv").exists()


def test_try_runs_against_your_own_extract(tmp_path):
    extract = tmp_path / "mine"
    assert main(["simulate", "--cases", "200", "--seed", "7", "--out", str(extract)]) == 0
    out = tmp_path / "report"
    assert main([
        "try", "--out", str(out),
        "--events", str(extract / "events.csv"),
        "--cases", str(extract / "cases.csv"),
    ]) == 0
    report = (out / "report.html").read_text(encoding="utf-8")
    assert "your own extract" in report
    # Nothing was copied, because the data was not ours to copy.
    assert not (out / "events.csv").exists()


def test_mine_writes_a_drawable_map(tmp_path):
    out = tmp_path / "mined"
    assert main(["mine", "--events", data.events_path(), "--out", str(out)]) == 0
    ElementTree.fromstring((out / "process_map.svg").read_text(encoding="utf-8"))
