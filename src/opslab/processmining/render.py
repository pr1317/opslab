"""Render a discovered process map to Graphviz DOT or Mermaid.

Both formats are text, which keeps the whole toolkit dependency-free: DOT for
anyone with Graphviz, Mermaid for anything that renders Markdown - GitHub,
Confluence, a wiki - which is usually where a process map needs to end up.
"""

from __future__ import annotations

from typing import Optional

from opslab.processmining.discovery import DirectlyFollowsGraph

__all__ = ["to_dot", "to_mermaid"]


def _escape_dot(text: str) -> str:
    return text.replace('"', '\\"')


def _node_id(activity: str, index: int) -> str:
    return "n%d" % index


def _edge_label(hours: float, frequency: int, show_time: bool) -> str:
    if not show_time:
        return str(frequency)
    return "%d\\n%.1fh" % (frequency, hours)


def to_dot(
    graph: DirectlyFollowsGraph,
    show_time: bool = True,
    min_edge_frequency: int = 1,
    title: Optional[str] = None,
) -> str:
    """Render the graph as Graphviz DOT.

    Edge thickness scales with frequency so the dominant path is visible
    without reading a single number.
    """
    model = graph.filter_edges(min_edge_frequency)
    order = sorted(model.nodes)
    ids = {activity: _node_id(activity, i) for i, activity in enumerate(order)}

    lines = ["digraph process {", "  rankdir=TB;", '  node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];', '  edge [fontname="Helvetica", fontsize=9];']
    if title:
        lines.append('  labelloc="t"; label="%s";' % _escape_dot(title))

    lines.append('  __start [shape=circle, label="", fillcolor="#2e7d32", width=0.3];')
    lines.append('  __end [shape=doublecircle, label="", fillcolor="#c62828", width=0.3];')

    for activity in order:
        stats = model.nodes[activity]
        lines.append(
            '  %s [label="%s\\n%d (%d cases)"];'
            % (ids[activity], _escape_dot(activity), stats.frequency, stats.case_frequency)
        )

    max_frequency = max((e.frequency for e in model.edges.values()), default=1)
    for (source, target), edge in sorted(model.edges.items()):
        if source not in ids or target not in ids:
            continue
        width = 1.0 + 4.0 * (edge.frequency / max_frequency)
        lines.append(
            '  %s -> %s [label="%s", penwidth=%.2f];'
            % (
                ids[source],
                ids[target],
                _edge_label(edge.mean_hours, edge.frequency, show_time),
                width,
            )
        )

    for activity, count in sorted(model.start_activities.items()):
        if activity in ids:
            lines.append('  __start -> %s [label="%d", style=dashed];' % (ids[activity], count))
    for activity, count in sorted(model.end_activities.items()):
        if activity in ids:
            lines.append('  %s -> __end [label="%d", style=dashed];' % (ids[activity], count))

    lines.append("}")
    return "\n".join(lines)


def to_mermaid(
    graph: DirectlyFollowsGraph,
    show_time: bool = True,
    min_edge_frequency: int = 1,
) -> str:
    """Render the graph as a Mermaid flowchart."""
    model = graph.filter_edges(min_edge_frequency)
    order = sorted(model.nodes)
    ids = {activity: _node_id(activity, i) for i, activity in enumerate(order)}

    lines = ["flowchart TD"]
    for activity in order:
        stats = model.nodes[activity]
        label = "%s<br/>%d (%d cases)" % (activity, stats.frequency, stats.case_frequency)
        lines.append('    %s["%s"]' % (ids[activity], label))

    for (source, target), edge in sorted(model.edges.items()):
        if source not in ids or target not in ids:
            continue
        label = _edge_label(edge.mean_hours, edge.frequency, show_time).replace("\\n", " / ")
        lines.append("    %s -->|%s| %s" % (ids[source], label, ids[target]))
    return "\n".join(lines)
