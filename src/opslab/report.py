"""A single self-contained HTML report over all four modules.

``opslab try`` writes one of these against the bundled sample. The point is to
give somebody who has never run the toolkit something to look at: charts, not
just the CSVs the other subcommands emit, and each one captioned with what it is
evidence *for*.

The file has no external references at all - the CSS is inline, every chart is
inline SVG drawn here from the fitted objects, and the process map falls back to
a table when the optional diagram renderer is unavailable. That matters for the
same reason the package has no dependencies: it has to open on a locked-down
laptop, from a file:// URL, with no network.
"""

from __future__ import annotations

import html
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from opslab import __version__

__all__ = ["build_report", "write_report"]


# --------------------------------------------------------------------------
# small formatting helpers
# --------------------------------------------------------------------------
def _escape(text: object) -> str:
    return html.escape(str(text), quote=True)


def _number(value: Optional[float], places: int = 1, suffix: str = "") -> str:
    """Format a number for display, rendering ``None`` as an em dash."""
    if value is None:
        return "&mdash;"
    return ("%.*f%s" % (places, value, suffix))


def _percent(fraction: Optional[float], places: int = 1) -> str:
    if fraction is None:
        return "&mdash;"
    return "%.*f%%" % (places, 100.0 * fraction)


def _tile(value: str, label: str, note: str = "") -> str:
    note_html = '<span class="tile-note">%s</span>' % _escape(note) if note else ""
    return (
        '<div class="tile"><span class="tile-value">%s</span>'
        '<span class="tile-label">%s</span>%s</div>'
        % (value, _escape(label), note_html)
    )


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    numeric_columns: Sequence[int] = (),
) -> str:
    """Render a table. Cells are pre-escaped by the caller where they contain markup."""
    numeric = set(numeric_columns)
    head = "".join(
        '<th class="num">%s</th>' % _escape(h) if i in numeric else "<th>%s</th>" % _escape(h)
        for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(
            '<td class="num">%s</td>' % cell if i in numeric else "<td>%s</td>" % cell
            for i, cell in enumerate(row)
        )
        body.append("<tr>%s</tr>" % cells)
    return (
        '<div class="scroll"><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>'
        % (head, "".join(body))
    )


def _section(anchor: str, number: str, title: str, lede: str, body: str) -> str:
    return (
        '<section id="%s"><h2><span class="num-badge">%s</span>%s</h2>'
        '<p class="lede">%s</p>%s</section>'
        % (_escape(anchor), _escape(number), _escape(title), lede, body)
    )


def _note(text: str) -> str:
    """A short interpretive aside under a chart or table."""
    return '<p class="note">%s</p>' % text


# --------------------------------------------------------------------------
# SVG primitives
# --------------------------------------------------------------------------
class _Frame:
    """Maps data coordinates onto a fixed SVG viewBox."""

    def __init__(
        self,
        width: float,
        height: float,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        margins: Tuple[float, float, float, float] = (18.0, 16.0, 40.0, 58.0),
    ) -> None:
        self.width = width
        self.height = height
        self.top, self.right, self.bottom, self.left = margins
        self.x_lo, self.x_hi = x_range
        self.y_lo, self.y_hi = y_range
        if self.x_hi <= self.x_lo:
            self.x_hi = self.x_lo + 1.0
        if self.y_hi <= self.y_lo:
            self.y_hi = self.y_lo + 1.0

    @property
    def plot_width(self) -> float:
        return self.width - self.left - self.right

    @property
    def plot_height(self) -> float:
        return self.height - self.top - self.bottom

    def x(self, value: float) -> float:
        span = self.x_hi - self.x_lo
        return self.left + (value - self.x_lo) / span * self.plot_width

    def y(self, value: float) -> float:
        span = self.y_hi - self.y_lo
        return self.top + (1.0 - (value - self.y_lo) / span) * self.plot_height


def _points(pairs: Sequence[Tuple[float, float]]) -> str:
    return " ".join("%.2f,%.2f" % pair for pair in pairs)


def _axes(frame: _Frame, y_ticks: Sequence[Tuple[float, str]], x_ticks: Sequence[Tuple[float, str]]) -> str:
    parts = []
    for value, label in y_ticks:
        y = frame.y(value)
        parts.append(
            '<line class="grid" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>'
            % (frame.left, y, frame.width - frame.right, y)
        )
        parts.append(
            '<text class="tick" x="%.2f" y="%.2f" text-anchor="end" dy="3.5">%s</text>'
            % (frame.left - 8, y, _escape(label))
        )
    for value, label in x_ticks:
        x = frame.x(value)
        parts.append(
            '<text class="tick" x="%.2f" y="%.2f" text-anchor="middle">%s</text>'
            % (x, frame.height - frame.bottom + 18, _escape(label))
        )
    parts.append(
        '<line class="axis" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>'
        % (frame.left, frame.height - frame.bottom, frame.width - frame.right,
           frame.height - frame.bottom)
    )
    return "".join(parts)


def _svg(width: float, height: float, body: str, label: str) -> str:
    return (
        '<div class="scroll"><svg viewBox="0 0 %.0f %.0f" width="100%%" '
        'preserveAspectRatio="xMidYMid meet" role="img" aria-label="%s" '
        'style="min-width:520px">%s</svg></div>'
        % (width, height, _escape(label), body)
    )


def _tick_values(lo: float, hi: float, count: int = 4) -> List[float]:
    """Evenly spaced tick positions across a range, endpoints included."""
    if count < 2:
        return [lo, hi]
    step = (hi - lo) / (count - 1)
    return [lo + step * i for i in range(count)]


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------
def control_chart_svg(chart, title: str, percent: bool = False) -> str:
    """A control chart: the series, its centre line, and its control limits.

    Limits are drawn per point rather than as one horizontal pair, because on a
    p-chart they breathe with the subgroup size - a quiet week has wider limits
    than a busy one, and flattening that is how a chart starts lying.
    """
    points = list(chart.points)
    if not points:
        return '<p class="note">No data to chart.</p>'

    lows = [min(p.lower, p.value) for p in points]
    highs = [max(p.upper, p.value) for p in points]
    lo, hi = min(lows), max(highs)
    pad = (hi - lo) * 0.12 or 1.0
    if percent:
        # A rate axis that runs below zero invites the reader to believe a
        # negative lower limit is meaningful. It is an artefact of the normal
        # approximation, and the p-chart already floors the limit itself.
        lo, hi, pad = max(0.0, lo - pad), min(1.0, hi + pad), 0.0
    frame = _Frame(760.0, 280.0, (0.0, float(len(points) - 1)), (lo - pad, hi + pad))

    def fmt(value: float) -> str:
        return "%.1f%%" % (100.0 * value) if percent else "%.4g" % value

    y_ticks = [(v, fmt(v)) for v in _tick_values(lo - pad, hi + pad, 5)]
    # Every label carries the same "2026-" prefix, which is 5 of its 8 characters
    # and the reason they collide; the axis title is not the place to repeat it.
    shown = [p.label for p in points]
    prefix = shown[0][:5]
    if len(prefix) == 5 and prefix.endswith("-") and all(l.startswith(prefix) for l in shown):
        shown = [l[5:] for l in shown]
    stride = max(1, len(points) // 9)
    x_ticks = [(float(i), shown[i]) for i in range(len(points)) if i % stride == 0]

    body = [_axes(frame, y_ticks, x_ticks)]
    body.append(
        '<polyline class="limit" points="%s"/>'
        % _points([(frame.x(i), frame.y(p.upper)) for i, p in enumerate(points)])
    )
    body.append(
        '<polyline class="limit" points="%s"/>'
        % _points([(frame.x(i), frame.y(p.lower)) for i, p in enumerate(points)])
    )
    body.append(
        '<polyline class="center" points="%s"/>'
        % _points([(frame.x(i), frame.y(p.center)) for i, p in enumerate(points)])
    )
    body.append(
        '<polyline class="series" points="%s"/>'
        % _points([(frame.x(i), frame.y(p.value)) for i, p in enumerate(points)])
    )
    for index, point in enumerate(points):
        signal = bool(point.violations)
        tip = "%s: %s" % (point.label, fmt(point.value))
        if signal:
            tip += " - " + ", ".join(point.violations)
        body.append(
            '<circle class="pt%s" cx="%.2f" cy="%.2f" r="%s"><title>%s</title></circle>'
            % (" sig" if signal else "", frame.x(index), frame.y(point.value),
               "4.2" if signal else "2.8", _escape(tip))
        )
    body.append(
        '<text class="tick" x="%.2f" y="%.2f" text-anchor="end">UCL</text>'
        % (frame.width - frame.right, frame.y(points[-1].upper) - 5)
    )
    return _svg(frame.width, frame.height, "".join(body), title)


def km_svg(curve, censor_marks: bool = True) -> str:
    """Kaplan-Meier survival with its confidence band, drawn as a step function."""
    if not curve.times:
        return '<p class="note">No survival data.</p>'

    horizon = max(curve.times)
    frame = _Frame(760.0, 300.0, (0.0, horizon), (0.0, 1.0))

    # A curve fitted to a few thousand cases has a step per distinct event time,
    # which is far more detail than 700 pixels can show. Thinning to a fixed
    # budget keeps every visible corner and takes the file from ~160KB to ~40KB;
    # the last step is always kept so the curve ends where the data ends.
    steps = list(zip(curve.times, curve.survival, curve.lower, curve.upper))
    stride = max(1, len(steps) // 400)
    if stride > 1:
        steps = steps[::stride] + [steps[-1]]

    forward: List[Tuple[float, float]] = []
    backward: List[Tuple[float, float]] = []
    series: List[Tuple[float, float]] = [(frame.x(0.0), frame.y(1.0))]
    previous_s, previous_lo, previous_hi = 1.0, 1.0, 1.0
    for time, survival, low, high in steps:
        x = frame.x(time)
        series.append((x, frame.y(previous_s)))
        series.append((x, frame.y(survival)))
        forward.append((x, frame.y(previous_hi)))
        forward.append((x, frame.y(high)))
        backward.append((x, frame.y(previous_lo)))
        backward.append((x, frame.y(low)))
        previous_s, previous_lo, previous_hi = survival, low, high

    band = forward + list(reversed(backward))
    body = [
        _axes(
            frame,
            [(v, "%.0f%%" % (100 * v)) for v in (0.0, 0.25, 0.5, 0.75, 1.0)],
            [(v, "%.0fh" % v) for v in _tick_values(0.0, horizon, 6)],
        ),
        '<polygon class="band" points="%s"/>' % _points(band),
        '<polyline class="series" points="%s"/>' % _points(series),
    ]

    median = curve.median
    if median is not None:
        x = frame.x(median)
        body.append(
            '<line class="ref" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>'
            % (x, frame.y(0.5), x, frame.height - frame.bottom)
        )
        body.append(
            '<line class="ref" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>'
            % (frame.left, frame.y(0.5), x, frame.y(0.5))
        )
        body.append(
            '<text class="tick anno" x="%.2f" y="%.2f">median %.0fh</text>'
            % (x + 6, frame.y(0.5) - 6, median)
        )

    if censor_marks:
        for time, survival, censored in zip(curve.times, curve.survival, curve.n_censored):
            if censored:
                x, y = frame.x(time), frame.y(survival)
                body.append(
                    '<line class="censor" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>'
                    % (x, y - 3.5, x, y + 3.5)
                )
    return _svg(frame.width, frame.height, "".join(body), "Kaplan-Meier survival curve")


def forest_svg(result, ground_truth: Optional[Dict[str, float]] = None) -> str:
    """Cox coefficients with 95% intervals, against the values that generated the data.

    Plotting the truth alongside the estimate is the whole reason the sample is
    synthetic: on a real extract you can only report a coefficient, but here the
    reader can check that the estimator recovered what was put in.
    """
    names = list(result.names)
    if not names:
        return '<p class="note">No covariates fitted.</p>'

    intervals = result.confidence_intervals()
    truth = ground_truth or {}
    values = [b for pair in intervals for b in pair] + list(result.coefficients)
    values += [truth[n] for n in names if n in truth]
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.14 or 0.5
    row_height = 34.0
    height = row_height * len(names) + 64.0
    frame = _Frame(760.0, height, (lo - pad, hi + pad), (0.0, float(len(names))),
                   margins=(20.0, 130.0, 40.0, 190.0))

    body = [
        _axes(frame, [], [(v, "%.1f" % v) for v in _tick_values(lo - pad, hi + pad, 5)]),
        '<line class="zero" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>'
        % (frame.x(0.0), frame.top, frame.x(0.0), frame.height - frame.bottom),
    ]
    for index, name in enumerate(names):
        y = frame.top + (index + 0.5) * frame.plot_height / len(names)
        low, high = intervals[index]
        coefficient = result.coefficients[index]
        body.append(
            '<text class="row-label" x="%.2f" y="%.2f" text-anchor="end" dy="4">%s</text>'
            % (frame.left - 12, y, _escape(name))
        )
        body.append(
            '<line class="whisker" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f"/>'
            % (frame.x(low), y, frame.x(high), y)
        )
        body.append(
            '<circle class="pt" cx="%.2f" cy="%.2f" r="4.5"><title>%s</title></circle>'
            % (frame.x(coefficient), y,
               _escape("%s: %.4f (95%% CI %.4f to %.4f)" % (name, coefficient, low, high)))
        )
        if name in truth:
            body.append(
                '<line class="truth" x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f">'
                '<title>%s</title></line>'
                % (frame.x(truth[name]), y - 9, frame.x(truth[name]), y + 9,
                   _escape("generating value %.4f" % truth[name]))
            )
        body.append(
            '<text class="row-value" x="%.2f" y="%.2f" dy="4">%.3f</text>'
            % (frame.width - frame.right + 10, y, coefficient)
        )
    body.append(
        '<text class="tick" x="%.2f" y="%.2f" text-anchor="middle">log hazard ratio</text>'
        % (frame.left + frame.plot_width / 2, frame.height - 6)
    )
    return _svg(frame.width, frame.height, "".join(body), "Cox coefficient forest plot")


# --------------------------------------------------------------------------
# report assembly
# --------------------------------------------------------------------------
_STYLE = """
:root{color-scheme:light dark;--bg:#fbfbf9;--panel:#fff;--ink:#1b1d1f;--muted:#5d6470;
--line:#e2e4e6;--accent:#1f6f5c;--accent-soft:#e6f1ed;--warn:#b4472e;--band:#c8dcd5;
--truth:#8a5a2b;--mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#141618;--panel:#1c1f22;--ink:#e8eaec;
--muted:#9aa3ae;--line:#2c3136;--accent:#5bbf9f;--accent-soft:#1d2c28;--warn:#e2795b;
--band:#2f4a43;--truth:#c9975c}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:40px 22px 90px}
header{border-bottom:2px solid var(--ink);padding-bottom:22px;margin-bottom:8px}
h1{font-size:30px;line-height:1.2;margin:0 0 8px}
h2{font-size:20px;margin:0 0 6px;display:flex;align-items:baseline;gap:10px}
h3{font-size:15px;margin:26px 0 6px;color:var(--muted);
text-transform:uppercase;letter-spacing:.07em}
.num-badge{font:600 12px/1 var(--mono);color:var(--accent);border:1px solid var(--accent);
border-radius:4px;padding:4px 6px;flex:none}
.sub{color:var(--muted);margin:0}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:22px 24px;margin:20px 0}
.lede{color:var(--muted);margin:0 0 18px;max-width:66ch}
.note{color:var(--muted);font-size:13.5px;margin:10px 0 0;max-width:72ch}
.note b,.note strong{color:var(--ink)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:0 0 6px}
.tile{background:var(--accent-soft);border-radius:8px;padding:14px 16px;display:flex;flex-direction:column;gap:2px}
.tile-value{font:600 24px/1.15 var(--mono);color:var(--ink)}
.tile-label{font-size:12.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.tile-note{font-size:12.5px;color:var(--muted);margin-top:4px}
.scroll{overflow-x:auto;margin:14px 0 0}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:7px 12px 7px 0;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.05em}
td.num,th.num{text-align:right;font-family:var(--mono);white-space:nowrap}
code,pre{font-family:var(--mono);font-size:13px}
pre{background:var(--accent-soft);padding:14px 16px;border-radius:8px;overflow-x:auto;margin:14px 0 0}
.sev{font:600 11px/1 var(--mono);padding:3px 6px;border-radius:4px;background:var(--accent-soft);color:var(--muted)}
.sev.error{background:var(--warn);color:#fff}
.sev.warning{border:1px solid var(--warn);color:var(--warn)}
.axis{stroke:var(--muted);stroke-width:1}
.grid{stroke:var(--line);stroke-width:1}
.tick{fill:var(--muted);font:11px var(--mono)}
.tick.anno{fill:var(--accent)}
.row-label{fill:var(--ink);font:12px var(--mono)}
.row-value{fill:var(--muted);font:11px var(--mono)}
.limit{fill:none;stroke:var(--warn);stroke-width:1.2;stroke-dasharray:5 4}
.center{fill:none;stroke:var(--muted);stroke-width:1;stroke-dasharray:2 3}
.series{fill:none;stroke:var(--accent);stroke-width:2;stroke-linejoin:round}
.pt{fill:var(--accent)}
.pt.sig{fill:var(--warn)}
.band{fill:var(--band);opacity:.5}
.censor{stroke:var(--accent);stroke-width:1.4;opacity:.75}
.ref{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}
.zero{stroke:var(--muted);stroke-width:1}
.whisker{stroke:var(--accent);stroke-width:2;stroke-linecap:round}
.truth{stroke:var(--truth);stroke-width:2.4}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:12.5px;color:var(--muted);margin:12px 0 0}
.legend span{display:flex;align-items:center;gap:6px}
.swatch{width:16px;height:3px;border-radius:2px;display:inline-block}
footer{color:var(--muted);font-size:13px;margin-top:34px;border-top:1px solid var(--line);padding-top:18px}
a{color:var(--accent)}
.map{--map-node:var(--accent-soft);--map-stroke:var(--line);--map-ink:var(--ink);
--map-muted:var(--muted);--map-edge:var(--muted);--map-rework:var(--warn);
--map-start:var(--accent);--map-end:var(--warn);--map-bg:var(--panel)}
details{margin:14px 0 0}
summary{cursor:pointer;color:var(--muted);font-size:13.5px}
.toc{display:flex;flex-wrap:wrap;gap:8px 18px;margin:16px 0 0;font-size:13.5px}
"""


def _legend(items: Sequence[Tuple[str, str]]) -> str:
    return '<div class="legend">%s</div>' % "".join(
        '<span><i class="swatch" style="background:%s"></i>%s</span>' % (colour, _escape(text))
        for text, colour in items
    )


def _mining_section(log, calendar, min_edge_frequency: int) -> str:
    from opslab.processmining import (
        bottlenecks,
        check_conformance,
        discover_dfg,
        rework_statistics,
        to_mermaid,
        to_svg,
        variants,
    )
    from opslab.cli import _default_rules

    graph = discover_dfg(log, calendar)
    variant_list = variants(log)
    top = variant_list[:6]
    covered = sum(v.count for v in top)

    tiles = '<div class="tiles">%s%s%s%s</div>' % (
        _tile(str(len(graph.nodes)), "activities"),
        _tile(str(len(graph.edges)), "transitions"),
        _tile(str(len(variant_list)), "distinct paths"),
        _tile(_percent(covered / max(1, graph.case_count)), "cases on the top 6"),
    )

    map_block = (
        "<h3>Discovered map</h3>"
        + _note(
            "The process as the log actually ran it, not as it was designed. Node "
            "outlines mark where cases start (green) and finish (red); dashed red "
            "edges are the ones that point backwards, which in a case-management "
            "process is exactly the rework. Transitions seen fewer than %d times are "
            "hidden so the common flow stays readable. Hover any edge for its counts."
            % min_edge_frequency
        )
        + '<div class="scroll map">%s</div>'
        % to_svg(graph, min_edge_frequency=min_edge_frequency)
        + "<details><summary>The same map as Mermaid source</summary><pre>%s</pre></details>"
        % _escape(to_mermaid(graph, min_edge_frequency=min_edge_frequency))
    )

    variant_rows = [
        [
            str(v.count),
            _percent(v.count / max(1, graph.case_count)),
            "<code>%s</code>" % " &rarr; ".join(_escape(a) for a in v.trace),
        ]
        for v in top
    ]
    variants_block = "<h3>Most common paths</h3>" + _table(
        ("cases", "share", "path"), variant_rows, numeric_columns=(0, 1)
    )

    bottleneck_rows = [
        [_escape(source), _escape(target), "%.0f" % total, "%.1f" % mean_hours, str(frequency)]
        for source, target, total, mean_hours, frequency in bottlenecks(log, calendar, 6, graph)
    ]
    bottleneck_block = (
        "<h3>Where the time goes</h3>"
        + _note(
            "Ranked by <b>total</b> working hours, not mean. A transition that is slow "
            "but rare costs the operation less than one that is mildly slow and happens "
            "to every case."
        )
        + _table(
            ("from", "to", "total hours", "mean hours", "n"),
            bottleneck_rows,
            numeric_columns=(2, 3, 4),
        )
    )

    rework = sorted(rework_statistics(log).values(), key=lambda s: -s.rework_rate)
    rework_rows = [
        [_escape(s.activity), _percent(s.rework_rate), _percent(s.first_time_right)]
        for s in rework[:6] if s.rework_rate > 0
    ]
    rework_block = "<h3>Rework</h3>" + _table(
        ("activity", "repeated in", "first-time-right"), rework_rows, numeric_columns=(1, 2)
    )

    report = check_conformance(log, _default_rules())
    counts = report.by_rule()
    conformance_rows = [
        [_escape(code), str(counts[code]), str(len(report.cases_for(code)))]
        for code in sorted(counts, key=lambda c: -counts[c])
    ]
    conformance_block = (
        "<h3>Conformance against declared controls</h3>"
        + _note(
            "%s of %d cases satisfied every rule. These are declarative checks - "
            "\"a peer check must precede release\", \"nobody may check their own work\" - "
            "rather than a Petri net, because that is the form operational controls are "
            "actually written in."
            % (_percent(report.fitness, 2), report.total_cases)
        )
        + _table(("rule", "violations", "cases affected"), conformance_rows, numeric_columns=(1, 2))
    )

    return _section(
        "mining", "1", "Process mining",
        "What the process does, reconstructed from timestamps rather than from a "
        "procedure document.",
        tiles + map_block + variants_block + bottleneck_block + rework_block + conformance_block,
    )


def _spc_section(cases, calendar) -> str:
    from opslab.cli import _weekly_breach_table
    from opslab.spc import capability, individuals_chart, p_chart

    labels, breaches, sizes, mean_hours = _weekly_breach_table(cases, calendar)
    if len(labels) < 2:
        return _section("spc", "2", "Statistical process control",
                        "Not enough completed weeks to chart.", "")

    breach_chart = p_chart(breaches, sizes, labels)
    hours_chart = individuals_chart(mean_hours, labels)
    signals = breach_chart.signals() + hours_chart.signals()

    tiles = '<div class="tiles">%s%s%s</div>' % (
        _tile(_percent(breach_chart.center), "mean breach rate", "centre line"),
        _tile(str(len(signals)), "special-cause signals"),
        _tile(
            min((p.label for p in signals), default="none"),
            "earliest signal",
            "week the process changed" if signals else "",
        ),
    )

    charts = (
        "<h3>Weekly SLA breach rate (p-chart)</h3>"
        + control_chart_svg(breach_chart, "Weekly SLA breach rate", percent=True)
        + _legend([("observed rate", "var(--accent)"), ("control limits", "var(--warn)"),
                   ("signal", "var(--warn)")])
        + _note(
            "Limits narrow in weeks with more cases and widen in quiet ones, because a "
            "rate from 40 cases is a noisier estimate than the same rate from 120. "
            "Red points broke at least one Nelson rule; hover for which."
        )
        + "<h3>Weekly mean handling hours (individuals chart)</h3>"
        + control_chart_svg(hours_chart, "Weekly mean handling hours")
    )

    tagged = (
        [("breach rate", p, True) for p in breach_chart.signals()]
        + [("handling hours", p, False) for p in hours_chart.signals()]
    )
    signal_rows = [
        [
            _escape(point.label),
            _escape(series),
            _percent(point.value) if as_rate else "%.1f" % point.value,
            "%+.2f" % point.standardised,
            _escape(", ".join(point.violations)),
        ]
        for series, point, as_rate in sorted(tagged, key=lambda row: (row[1].label, row[0]))
    ]
    signal_block = ""
    if signal_rows:
        signal_block = (
            "<h3>Signals</h3>"
            + _table(("week", "chart", "value", "sigma", "rules broken"),
                     signal_rows, numeric_columns=(2, 3))
            + _note(
                "Nelson rules 1 to 8, each flagged on the point that completes the "
                "pattern. Rule 1 is a single point beyond three sigma; rules 2, 5 and 6 "
                "catch a sustained shift that never breaks a limit, which is the failure "
                "mode a threshold dashboard misses entirely."
            )
        )

    resolved = [c for c in cases if c.resolved]
    capability_block = ""
    if resolved:
        target = max(c.sla_hours for c in resolved)
        report = capability([c.duration_hours for c in resolved], usl=target)
        capability_block = (
            "<h3>Capability against a %.0f-hour target</h3>" % target
            + '<div class="tiles">%s%s%s%s</div>' % (
                _tile(_number(report.ppk, 2), "Ppk", "spread and distance to the target"),
                _tile(_percent(report.observed_defective), "observed over target"),
                _tile("{:,.0f}".format(report.dpmo), "DPMO", "defects per million"),
                _tile(_number(report.sigma_level, 2), "sigma level", "with the 1.5 shift"),
            )
            + _note(
                "Ppk uses the overall standard deviation, so it describes what the customer "
                "actually received; Cp and Cpk use the within-subgroup estimate and describe "
                "what the process could deliver if it were stable, which is the usual way "
                "capability gets overstated for a process the charts above have just shown "
                "to be out of control. DPMO is higher than the observed rate because it "
                "comes from a fitted normal, and turnaround time is right-skewed - a reason "
                "to quote both numbers rather than pick the flattering one."
            )
        )

    lede = (
        "Whether this week's number is a change or just noise. Weeks are bucketed by the "
        "date a case <b>completed</b>, not the date it arrived: bucketing by arrival puts "
        "finished and unfinished work in the same point and hides a growing backlog."
    )
    return _section("spc", "2", "Statistical process control", lede,
                    tiles + charts + signal_block + capability_block)


def _sla_section(cases, ground_truth: Optional[Dict[str, float]]) -> str:
    from opslab.sla import concordance_index, design_matrix, fit_cox, kaplan_meier, logrank_test

    durations = [c.duration_hours for c in cases]
    events = [1 if c.resolved else 0 for c in cases]
    open_cases = len(cases) - sum(events)
    curve = kaplan_meier(durations, events)

    closed = sorted(c.duration_hours for c in cases if c.resolved)
    naive_median = closed[len(closed) // 2] if closed else 0.0
    median = curve.median

    tiles = '<div class="tiles">%s%s%s</div>' % (
        _tile(_percent(open_cases / max(1, len(cases))), "still open",
              "%d of %d cases" % (open_cases, len(cases))),
        _tile(_number(naive_median, 1, "h"), "median, closed cases only", "biased low"),
        _tile(_number(median, 1, "h") if median is not None else "not reached",
              "median, Kaplan-Meier", "uses the open cases too"),
    )

    km_block = (
        "<h3>Time to resolution</h3>"
        + km_svg(curve)
        + _legend([("survival", "var(--accent)"), ("95% confidence band", "var(--band)")])
        + _note(
            "The gap between the two medians above is the whole argument for this module. "
            "Dropping the %d open cases and taking a median of what is left conditions on "
            "the case having already finished, which systematically excludes the slow ones. "
            "Kaplan-Meier keeps them as evidence up to the point they were last seen. "
            "Small ticks mark censoring times." % open_cases
        )
    )

    logrank_block = ""
    groups: Dict[str, List] = {}
    for case in cases:
        groups.setdefault(case.attributes.get("priority", ""), []).append(case)
    if len(groups) == 2:
        (label_a, cases_a), (label_b, cases_b) = sorted(groups.items())
        result = logrank_test(
            [c.duration_hours for c in cases_a], [1 if c.resolved else 0 for c in cases_a],
            [c.duration_hours for c in cases_b], [1 if c.resolved else 0 for c in cases_b],
            label_a, label_b,
        )
        logrank_block = "<h3>Log-rank test by priority</h3><pre>%s</pre>" % _escape(result.to_text())

    matrix = design_matrix(
        cases,
        numeric=["complexity", "backlog_index", "awaiting_third_party"],
        binary={"priority": "Urgent", "channel": "Post"},
    )
    model = fit_cox(durations, events, matrix.rows, names=matrix.names)
    c_index, pairs = concordance_index(durations, events, model.risk_scores(matrix.rows))

    cox_rows = [
        [
            _escape(row["term"]),
            "%.4f" % row["coef"],
            "%.4f" % row["hazard_ratio"],
            "%.4f to %.4f" % (row["ci_lower"], row["ci_upper"]),
            "%.3g" % row["p_value"],
            ("%.4f" % ground_truth[row["term"]]) if ground_truth and row["term"] in ground_truth else "&mdash;",
        ]
        for row in model.result.to_rows()
    ]
    cox_block = (
        "<h3>Cox proportional hazards</h3>"
        + forest_svg(model.result, ground_truth)
        + _legend([("estimate and 95% CI", "var(--accent)"), ("generating value", "var(--truth)")])
        + _table(
            ("covariate", "coefficient", "hazard ratio", "95% CI", "p", "true value"),
            cox_rows, numeric_columns=(1, 2, 3, 4, 5),
        )
        + _note(
            "A hazard ratio above 1 means the case clears faster; below 1 means it lingers. "
            "The sample is synthetic precisely so the last column can exist: the data was "
            "generated from a Weibull proportional-hazards model with published "
            "coefficients, so the estimator can be checked rather than admired. %s "
            "Concordance index <b>%.4f</b> over %s comparable pairs."
            % (_recovery_note(model.result, ground_truth), c_index, "{:,}".format(pairs))
        )
    )

    sla_target = cases.cases[0].sla_hours if len(cases) else 40.0
    scored = sorted(
        ((model.breach_probability(row, sla_target), case_id)
         for row, case_id in zip(matrix.rows, matrix.case_ids)),
        reverse=True,
    )
    risk_block = (
        "<h3>Cases most likely to breach a %.0f-hour target</h3>" % sla_target
        + _table(
            ("case", "breach probability"),
            [[_escape(cid), _percent(p)] for p, cid in scored[:10]],
            numeric_columns=(1,),
        )
        + _note(
            "This is the output an operations manager can act on: a worklist ordered by "
            "risk, derived from the same fitted model rather than from a separate "
            "scoring heuristic."
        )
    )

    return _section(
        "sla", "3", "SLA survival analysis",
        "How long cases take, when a tenth of them have not finished yet - and the slow "
        "ones are exactly the ones still open.",
        tiles + km_block + logrank_block + cox_block + risk_block,
    )


def _recovery_note(result, ground_truth: Optional[Dict[str, float]]) -> str:
    """How many generating values the fitted intervals actually covered."""
    if not ground_truth:
        return ""
    intervals = result.confidence_intervals()
    checked = [
        (name, ground_truth[name], intervals[index])
        for index, name in enumerate(result.names)
        if name in ground_truth
    ]
    if not checked:
        return ""
    covered = sum(1 for _, truth, (low, high) in checked if low <= truth <= high)
    return (
        "<b>%d of %d</b> generating values fall inside their 95%% interval, which is "
        "what a correct estimator should do at this sample size - an occasional miss "
        "is the interval behaving as advertised, not a defect."
        % (covered, len(checked))
    )


def _lint_section(model_path: str) -> str:
    from opslab.daxlint import ALL_RULES, Severity, lint, load_model

    model = load_model(model_path)
    findings = lint(model)
    counts = {severity: 0 for severity in Severity}
    for finding in findings:
        counts[finding.severity] += 1

    tiles = '<div class="tiles">%s%s%s%s</div>' % (
        _tile(str(len(ALL_RULES)), "rules"),
        _tile(str(counts[Severity.ERROR]), "errors"),
        _tile(str(counts[Severity.WARNING]), "warnings"),
        _tile(str(len(model.measures())), "measures checked"),
    )

    rows = [
        [
            "<code>%s</code>" % _escape(f.code),
            '<span class="sev %s">%s</span>' % (_escape(f.severity.value), _escape(f.severity.value)),
            "<code>%s%s</code>" % (_escape(f.object_name), (":%d" % f.line) if f.line else ""),
            _escape(f.message),
        ]
        for f in findings[:24]
    ]
    table = _table(("rule", "severity", "object", "finding"), rows)
    more = ""
    if len(findings) > 24:
        more = _note("%d further findings omitted; <code>opslab daxlint</code> prints them all."
                     % (len(findings) - 24))

    return _section(
        "daxlint", "4", "Power BI model lint",
        "The model that reports all of the above, checked the way code is checked. "
        "The sample <code>.bim</code> is written to fail every rule at least once, so "
        "the output is worth reading.",
        tiles + table + more
        + _note(
            "Findings split into model-level defects (MOD) - bidirectional relationships, "
            "unmarked date tables, identifier columns left summarising - and DAX-level ones "
            "(DAX) found by tokenising each expression, which is why a column reference "
            "inside a string literal does not trip a rule."
        ),
    )


def build_report(
    events_path: str,
    cases_path: str,
    model_path: str,
    *,
    calendar=None,
    min_edge_frequency: int = 5,
    ground_truth: Optional[Dict[str, float]] = None,
    source_note: str = "",
) -> str:
    """Run all four modules and return one self-contained HTML document."""
    from opslab.calendar import BusinessCalendar
    from opslab.eventlog import CaseTable, EventLog

    calendar = calendar or BusinessCalendar()
    log = EventLog.from_csv(events_path)
    cases = CaseTable.from_csv(cases_path)

    resolved = [c for c in cases if c.resolved]
    breached = sum(1 for c in resolved if c.breached)
    header_tiles = '<div class="tiles">%s%s%s%s</div>' % (
        _tile("{:,}".format(len(cases)), "cases"),
        _tile("{:,}".format(len(log)), "events"),
        _tile(_percent((len(cases) - len(resolved)) / max(1, len(cases))), "still open"),
        _tile(_percent(breached / max(1, len(resolved))), "breached", "of resolved cases"),
    )

    sections = [
        _mining_section(log, calendar, min_edge_frequency),
        _spc_section(cases, calendar),
        _sla_section(cases, ground_truth),
        _lint_section(model_path),
    ]

    toc = '<p class="toc">%s</p>' % "".join(
        '<a href="#%s">%s. %s</a>' % (anchor, number, title)
        for anchor, number, title in (
            ("mining", "1", "Process mining"),
            ("spc", "2", "Statistical process control"),
            ("sla", "3", "SLA survival analysis"),
            ("daxlint", "4", "Power BI model lint"),
        )
    )

    generated = datetime.now().strftime("%d %B %Y")
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        "<title>opslab &mdash; sample operations report</title>\n"
        "<style>%s</style>\n</head>\n<body>\n<div class=\"wrap\">\n"
        "<header>\n<h1>An operation, analysed four ways</h1>\n"
        '<p class="sub">Generated by <code>opslab try</code> (version %s) on %s. '
        "%s Every chart on this page was drawn from a model fitted in the standard "
        "library alone &mdash; no numpy, scipy, pandas or lifelines.</p>\n%s\n</header>\n"
        "%s\n%s\n"
        "<footer>opslab is open source under the MIT licence. "
        'Source and documentation: <a href="https://github.com/pr1317/opslab">'
        "github.com/pr1317/opslab</a>.</footer>\n"
        "</div>\n</body>\n</html>\n"
        % (_STYLE, _escape(__version__), generated, _escape(source_note), toc,
           header_tiles, "\n".join(sections))
    )


def write_report(path: str, **kwargs) -> str:
    """Build the report and write it to ``path``, returning the path."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(build_report(**kwargs))
    return path
