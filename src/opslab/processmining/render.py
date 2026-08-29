"""Render a discovered process map to Graphviz DOT or Mermaid.

Both formats are text, which keeps the whole toolkit dependency-free: DOT for
anyone with Graphviz, Mermaid for anything that renders Markdown - GitHub,
Confluence, a wiki - which is usually where a process map needs to end up.
"""

from __future__ import annotations

from typing import Optional

from opslab.processmining.discovery import DirectlyFollowsGraph

__all__ = ["to_dot", "to_mermaid", "to_svg"]


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


# --------------------------------------------------------------------------
# SVG
# --------------------------------------------------------------------------
#: Layout constants, in SVG user units.
_NODE_W = 172.0
_NODE_H = 50.0
_GAP_X = 24.0
_GAP_Y = 68.0
_MARGIN = 24.0
_GUTTER = 52.0  # left channel that rework edges are routed through


def _successors(edges) -> dict:
    following: dict = {}
    for source, target in edges:
        following.setdefault(source, set()).add(target)
    return following


def _back_edges(nodes, edges) -> set:
    """Edges that close a cycle, found by a depth-first sweep.

    A directly-follows graph is almost never acyclic - rework loops are the
    point of drawing one - so the layering below needs the cycles broken first.
    These are the edges to break, and they are also the ones worth drawing
    differently, because a backwards edge in a case-management process *is* the
    rework.
    """
    following = _successors(edges)
    state = {activity: 0 for activity in nodes}  # 0 unseen, 1 on the stack, 2 done
    back = set()

    for root in sorted(nodes):
        if state[root] != 0:
            continue
        state[root] = 1
        stack = [(root, iter(sorted(following.get(root, ()))))]
        while stack:
            activity, remaining = stack[-1]
            descended = False
            for target in remaining:
                if state.get(target, 0) == 1:
                    back.add((activity, target))
                elif state.get(target, 0) == 0:
                    state[target] = 1
                    stack.append((target, iter(sorted(following.get(target, ())))))
                    descended = True
                    break
            if not descended:
                state[activity] = 2
                stack.pop()
    return back


def _layers(nodes, edges, back) -> dict:
    """Longest-path layering over the acyclic part: every node sits below its inputs."""
    forward = [(s, t) for (s, t) in edges if s != t and (s, t) not in back]
    layer = {activity: 0 for activity in nodes}
    for _ in range(len(nodes) + 1):
        changed = False
        for source, target in forward:
            if layer[target] < layer[source] + 1:
                layer[target] = layer[source] + 1
                changed = True
        if not changed:
            break
    return layer


def _order_within_layers(layer, edges, back) -> dict:
    """Reduce edge crossings by repeatedly sorting each layer on its neighbours' mean position."""
    rows: dict = {}
    for activity, depth in layer.items():
        rows.setdefault(depth, []).append(activity)
    for depth in rows:
        rows[depth].sort()

    position = {a: float(i) for depth in rows for i, a in enumerate(rows[depth])}
    forward = [(s, t) for (s, t) in edges if s != t and (s, t) not in back]
    incoming: dict = {}
    outgoing: dict = {}
    for source, target in forward:
        incoming.setdefault(target, []).append(source)
        outgoing.setdefault(source, []).append(target)

    depths = sorted(rows)
    for sweep in range(4):
        order = depths if sweep % 2 == 0 else list(reversed(depths))
        for depth in order:
            neighbours = incoming if sweep % 2 == 0 else outgoing
            def barycentre(activity: str) -> tuple:
                linked = neighbours.get(activity, [])
                if not linked:
                    return (position[activity], activity)
                return (sum(position[n] for n in linked) / len(linked), activity)
            rows[depth].sort(key=barycentre)
            for index, activity in enumerate(rows[depth]):
                position[activity] = float(index)
    return rows


def _wrap(text: str, limit: int = 22) -> list:
    """Split an activity name over at most two lines, breaking on a space."""
    if len(text) <= limit:
        return [text]
    words = text.split(" ")
    first: list = []
    while words and len(" ".join(first + words[:1])) <= limit:
        first.append(words.pop(0))
    if not first:
        return [text[:limit], text[limit:limit * 2]]
    return [" ".join(first), " ".join(words)]


def to_svg(
    graph: DirectlyFollowsGraph,
    min_edge_frequency: int = 1,
    show_time: bool = True,
    label_top_edges: int = 8,
) -> str:
    """Draw the process map as standalone SVG, laid out top to bottom.

    Graphviz would do this better, but it is a system package, and the point of
    this toolkit is that it runs where nothing can be installed. The layout is
    the usual three steps - break cycles, layer by longest path, then reduce
    crossings by sorting each layer on its neighbours' average position - which
    is enough for a case-management process, where the honest shape is mostly a
    line with rework loops hanging off it.

    Colours come from CSS custom properties with literal fallbacks, so the
    diagram inherits a host page's theme but still renders on its own.
    """
    model = graph.filter_edges(min_edge_frequency)
    nodes = sorted(model.nodes)
    if not nodes:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="0" height="0"></svg>'

    edges = set(model.edges)
    back = _back_edges(nodes, edges)
    layer = _layers(nodes, edges, back)
    rows = _order_within_layers(layer, edges, back)

    widest = max(len(row) for row in rows.values())
    width = _MARGIN * 2 + _GUTTER + widest * _NODE_W + (widest - 1) * _GAP_X
    height = _MARGIN * 2 + len(rows) * _NODE_H + (len(rows) - 1) * _GAP_Y

    centre: dict = {}
    for depth in sorted(rows):
        row = rows[depth]
        span = len(row) * _NODE_W + (len(row) - 1) * _GAP_X
        left = _MARGIN + _GUTTER + (width - _MARGIN * 2 - _GUTTER - span) / 2.0
        y = _MARGIN + depth * (_NODE_H + _GAP_Y)
        for index, activity in enumerate(row):
            centre[activity] = (left + index * (_NODE_W + _GAP_X) + _NODE_W / 2.0, y)

    # Almost every activity ends *some* case, so colouring on the raw counter marks
    # the whole diagram. Only the activities that begin or finish a real share of
    # cases are worth calling out.
    threshold = 0.05 * max(1, model.case_count)
    starts = {a for a, n in model.start_activities.items() if n >= threshold}
    ends = {a for a, n in model.end_activities.items() if n >= threshold}

    frequencies = [e.frequency for e in model.edges.values()] or [1]
    busiest = max(frequencies)
    labelled = {
        (e.source, e.target)
        for e in sorted(model.edges.values(), key=lambda e: -e.frequency)[:label_top_edges]
    }

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %.0f %.0f" width="%.0f" '
        'height="%.0f" role="img" aria-label="Discovered process map">' % (width, height, width, height),
        "<defs>"
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="var(--map-edge,#4a5568)"/></marker>'
        '<marker id="arrow-back" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="var(--map-rework,#b4472e)"/></marker>'
        "</defs>",
        "<style>"
        ".mp-node{fill:var(--map-node,#eef3f1);stroke:var(--map-stroke,#c3d2cd);stroke-width:1}"
        ".mp-node.start{stroke:var(--map-start,#2e7d5b);stroke-width:2}"
        ".mp-node.end{stroke:var(--map-end,#b4472e);stroke-width:2}"
        ".mp-name{fill:var(--map-ink,#1b1d1f);font:600 11px ui-sans-serif,system-ui,sans-serif}"
        ".mp-meta{fill:var(--map-muted,#6a7480);font:10px ui-monospace,SFMono-Regular,Menlo,monospace}"
        ".mp-edge{fill:none;stroke:var(--map-edge,#4a5568);marker-end:url(#arrow)}"
        ".mp-edge.rework{stroke:var(--map-rework,#b4472e);stroke-dasharray:5 4;marker-end:url(#arrow-back)}"
        ".mp-label{fill:var(--map-muted,#6a7480);font:9.5px ui-monospace,SFMono-Regular,Menlo,monospace;"
        "stroke:var(--map-bg,#fbfbf9);stroke-width:3.5;paint-order:stroke;stroke-linejoin:round}"
        "</style>",
    ]

    labels: list = []

    def anchor(activity: str, at_bottom: bool):
        x, y = centre[activity]
        return x, (y + _NODE_H if at_bottom else y)

    for (source, target), edge in sorted(model.edges.items()):
        if source not in centre or target not in centre:
            continue
        stroke = 1.0 + 3.4 * (edge.frequency / busiest)
        title = "%s to %s: %d times, %.1fh mean" % (
            source, target, edge.frequency, edge.mean_hours
        )
        if source == target:
            x, y = centre[source]
            right = x + _NODE_W / 2.0
            path = ("M%.1f,%.1f C%.1f,%.1f %.1f,%.1f %.1f,%.1f"
                    % (right, y + 14, right + 34, y + 4, right + 34, y + _NODE_H - 4,
                       right, y + _NODE_H - 14))
        elif (source, target) in back:
            x_s, y_s = centre[source]
            x_t, y_t = centre[target]
            channel = _MARGIN + _GUTTER * 0.45
            path = ("M%.1f,%.1f C%.1f,%.1f %.1f,%.1f %.1f,%.1f"
                    % (x_s - _NODE_W / 2.0, y_s + _NODE_H / 2.0,
                       channel, y_s + _NODE_H / 2.0,
                       channel, y_t + _NODE_H / 2.0,
                       x_t - _NODE_W / 2.0, y_t + _NODE_H / 2.0))
        else:
            x_s, y_s = anchor(source, True)
            x_t, y_t = anchor(target, False)
            midpoint = (y_s + y_t) / 2.0
            # An edge that skips layers would otherwise run straight through the
            # nodes in between; bowing it sideways in proportion to the distance
            # skipped keeps it traceable.
            span = layer[target] - layer[source]
            bow = 0.0 if span <= 1 else min(_NODE_W * 0.6, (span - 1) * 26.0)
            path = ("M%.1f,%.1f C%.1f,%.1f %.1f,%.1f %.1f,%.1f"
                    % (x_s, y_s, x_s + bow, midpoint, x_t + bow, midpoint, x_t, y_t))
            if (source, target) in labelled:
                # Where the curve actually is at t=0.5, which is not the midpoint
                # of its endpoints once the control points have been bowed aside.
                label = ("%d / %.1fh" % (edge.frequency, edge.mean_hours)
                         if show_time else str(edge.frequency))
                labels.append(
                    '<text class="mp-label" x="%.1f" y="%.1f" text-anchor="middle">%s</text>'
                    % ((x_s + x_t) / 2.0 + 0.75 * bow, midpoint + 3.5, _escape_xml(label))
                )
        classes = "mp-edge rework" if (source, target) in back or source == target else "mp-edge"
        parts.append('<path class="%s" d="%s" stroke-width="%.2f"><title>%s</title></path>'
                     % (classes, path, stroke, _escape_xml(title)))

    for activity in nodes:
        if activity not in centre:
            continue
        x, y = centre[activity]
        stats = model.nodes[activity]
        role = ""
        if activity in starts:
            role = " start"
        elif activity in ends:
            role = " end"
        parts.append(
            '<rect class="mp-node%s" x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="7"/>'
            % (role, x - _NODE_W / 2.0, y, _NODE_W, _NODE_H)
        )
        lines = _wrap(activity)
        for index, line in enumerate(lines):
            parts.append(
                '<text class="mp-name" x="%.1f" y="%.1f" text-anchor="middle">%s</text>'
                % (x, y + 17 + index * 13, _escape_xml(line))
            )
        parts.append(
            '<text class="mp-meta" x="%.1f" y="%.1f" text-anchor="middle">%s</text>'
            % (x, y + _NODE_H - 8, _escape_xml("%d cases" % stats.case_frequency))
        )

    parts.extend(labels)
    parts.append("</svg>")
    return "".join(parts)


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )
