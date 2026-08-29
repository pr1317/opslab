"""Export everything the report shows as JSON, for a front end to draw.

`opslab report` renders HTML on the server side, which is the right answer when
the output is a document. It is the wrong answer when the output should respond
to a slider - a reader who wants to know what a *different* case's breach risk
would be needs the fitted model, not a picture of one.

So this writes the fitted objects themselves: the coefficients and the Breslow
baseline hazard (from which any covariate vector's survival curve follows by
arithmetic a browser can do), the chart points with their limits, the discovered
map already laid out at a range of thresholds, and the lint findings. The
`docs/` demo is built on exactly this file.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from opslab import __version__

__all__ = ["build_payload", "write_payload", "map_thresholds", "MAP_STEPS"]

#: How many cut-offs the map is pre-laid-out at, so the browser can offer the
#: filter without needing the layout algorithm ported to JavaScript.
MAP_STEPS = 8

#: How many points of the Kaplan-Meier curve to keep for drawing. The fitted step
#: function has one point per distinct event time, far more than a few hundred
#: pixels can show. The Cox baseline hazard is *not* thinned: it is arithmetic
#: the browser evaluates rather than a picture, and dropping steps from a
#: cumulative hazard biases every probability computed from it downwards.
CURVE_POINTS = 300


def _thin(rows: Sequence, budget: int) -> List:
    """Keep at most ``budget`` evenly spaced entries, always including the last."""
    rows = list(rows)
    if len(rows) <= budget:
        return rows
    stride = len(rows) // budget
    thinned = rows[::stride]
    if thinned[-1] is not rows[-1]:
        thinned.append(rows[-1])
    return thinned


def _round(value: float, places: int = 6) -> float:
    return round(float(value), places)


def map_thresholds(graph, steps: int = MAP_STEPS) -> List[int]:
    """Cut-offs spread over the observed edge frequencies, not over round numbers.

    A fixed ladder like 1, 10, 100 is useless on a log where every transition
    happens hundreds of times: the first several positions all show the same
    map. Taking quantiles of the frequencies themselves means each step of the
    slider actually removes something, whatever the volume of the process.
    """
    frequencies = sorted(edge.frequency for edge in graph.edges.values())
    if not frequencies:
        return [1]
    chosen = {1}
    for index in range(steps):
        position = int(index * (len(frequencies) - 1) / max(1, steps - 1))
        chosen.add(frequencies[position])
    return sorted(chosen)


def _chart_payload(chart) -> dict:
    return {
        "kind": chart.kind,
        "center": _round(chart.center),
        "points": [
            {
                "label": point.label,
                "value": _round(point.value),
                "center": _round(point.center),
                "lower": _round(point.lower),
                "upper": _round(point.upper),
                "sigma": _round(point.standardised, 4),
                "violations": list(point.violations),
            }
            for point in chart.points
        ],
    }


def _covariate_metadata(matrix, names: Sequence[str]) -> List[dict]:
    """Range and centre of each column, so a control can be built for it."""
    described = []
    for name in names:
        column = matrix.column(name)
        low, high = min(column), max(column)
        binary = set(column) <= {0.0, 1.0}
        # Widened to a round hundredth so a slider stepping by 0.01 lands on
        # values a reader recognises, rather than on min + k*0.01.
        described.append({
            "name": name,
            "binary": binary,
            "min": 0.0 if binary else math.floor(low * 100) / 100.0,
            "max": 1.0 if binary else math.ceil(high * 100) / 100.0,
            "mean": _round(sum(column) / len(column), 2),
        })
    return described


def build_payload(
    events_path: str,
    cases_path: str,
    model_path: str,
    *,
    calendar=None,
    ground_truth: Optional[Dict[str, float]] = None,
) -> dict:
    """Run every module and return a JSON-ready dictionary of the results."""
    from opslab.calendar import BusinessCalendar
    from opslab.cli import _default_rules, _weekly_breach_table
    from opslab.daxlint import ALL_RULES, lint, load_model
    from opslab.eventlog import CaseTable, EventLog
    from opslab.processmining import (
        bottlenecks,
        check_conformance,
        discover_dfg,
        rework_statistics,
        to_svg,
        variants,
    )
    from opslab.sla import (
        concordance_index,
        design_matrix,
        fit_cox,
        kaplan_meier,
    )
    from opslab.spc import capability, individuals_chart, p_chart

    calendar = calendar or BusinessCalendar()
    log = EventLog.from_csv(events_path)
    cases = CaseTable.from_csv(cases_path)

    # -- process mining ----------------------------------------------------
    graph = discover_dfg(log, calendar)
    thresholds = map_thresholds(graph)
    variant_list = variants(log)
    conformance = check_conformance(log, _default_rules())
    rework = sorted(rework_statistics(log).values(), key=lambda s: -s.rework_rate)

    mining = {
        "activities": len(graph.nodes),
        "transitions": len(graph.edges),
        "variants": len(variant_list),
        "maps": {
            str(threshold): to_svg(graph, min_edge_frequency=threshold)
            for threshold in thresholds
        },
        "thresholds": thresholds,
        "topVariants": [
            {
                "count": variant.count,
                "share": _round(variant.count / max(1, graph.case_count), 4),
                "trace": list(variant.trace),
            }
            for variant in variant_list[:8]
        ],
        "bottlenecks": [
            {
                "source": source,
                "target": target,
                "totalHours": _round(total, 1),
                "meanHours": _round(mean_hours, 2),
                "frequency": frequency,
            }
            for source, target, total, mean_hours, frequency
            in bottlenecks(log, calendar, 8, graph)
        ],
        "rework": [
            {
                "activity": stats.activity,
                "reworkRate": _round(stats.rework_rate, 4),
                "firstTimeRight": _round(stats.first_time_right, 4),
            }
            for stats in rework[:8] if stats.rework_rate > 0
        ],
        "conformance": {
            "fitness": _round(conformance.fitness, 4),
            "totalCases": conformance.total_cases,
            "byRule": [
                {"rule": code, "violations": count,
                 "cases": len(conformance.cases_for(code))}
                for code, count in sorted(
                    conformance.by_rule().items(), key=lambda item: -item[1]
                )
            ],
        },
    }

    # -- statistical process control ---------------------------------------
    labels, breaches, sizes, mean_hours = _weekly_breach_table(cases, calendar)
    breach_chart = p_chart(breaches, sizes, labels)
    hours_chart = individuals_chart(mean_hours, labels)
    resolved = [case for case in cases if case.resolved]
    target = max(case.sla_hours for case in resolved) if resolved else 0.0
    capability_report = (
        capability([case.duration_hours for case in resolved], usl=target)
        if resolved else None
    )

    spc = {
        "breachRate": _chart_payload(breach_chart),
        "handlingHours": _chart_payload(hours_chart),
        "weeklyVolume": [
            {"label": label, "resolved": size, "breached": breached}
            for label, size, breached in zip(labels, sizes, breaches)
        ],
        "capability": (
            {
                "target": _round(target, 1),
                "ppk": _round(capability_report.ppk, 4) if capability_report.ppk else None,
                "dpmo": _round(capability_report.dpmo, 1),
                "sigmaLevel": _round(capability_report.sigma_level, 3),
                "observedDefective": _round(capability_report.observed_defective, 4),
            }
            if capability_report else None
        ),
    }

    # -- survival ----------------------------------------------------------
    durations = [case.duration_hours for case in cases]
    events = [1 if case.resolved else 0 for case in cases]
    curve = kaplan_meier(durations, events)
    closed = sorted(case.duration_hours for case in cases if case.resolved)

    matrix = design_matrix(
        cases,
        numeric=["complexity", "backlog_index", "awaiting_third_party"],
        binary={"priority": "Urgent", "channel": "Post"},
    )
    model = fit_cox(durations, events, matrix.rows, names=matrix.names)
    c_index, pairs = concordance_index(durations, events, model.risk_scores(matrix.rows))

    km_points = _thin(
        list(zip(curve.times, curve.survival, curve.lower, curve.upper)),
        CURVE_POINTS,
    )

    sla = {
        "cases": len(cases),
        "open": events.count(0),
        "censoringRate": _round(events.count(0) / max(1, len(cases)), 4),
        "medianClosedOnly": _round(closed[len(closed) // 2], 2) if closed else None,
        "medianKaplanMeier": _round(curve.median, 2) if curve.median is not None else None,
        "km": {
            "times": [_round(t, 3) for t, _, _, _ in km_points],
            "survival": [_round(s, 6) for _, s, _, _ in km_points],
            "lower": [_round(lo, 6) for _, _, lo, _ in km_points],
            "upper": [_round(hi, 6) for _, _, _, hi in km_points],
        },
        "cox": {
            "names": list(model.result.names),
            # Full precision: these are evaluated in the browser rather than
            # displayed, and a parity test holds the two implementations to 1e-9.
            "coefficients": [_round(b, 12) for b in model.result.coefficients],
            "standardErrors": [_round(se) for se in model.result.standard_errors],
            "confidenceIntervals": [
                [_round(low), _round(high)]
                for low, high in model.result.confidence_intervals()
            ],
            "pValues": [float("%.6g" % p) for p in model.result.p_values],
            "groundTruth": dict(ground_truth or {}),
            "concordance": _round(c_index, 4),
            "comparablePairs": pairs,
            "covariates": _covariate_metadata(matrix, model.result.names),
            "baseline": {
                "times": [_round(t, 3) for t in model.baseline_times],
                "cumulativeHazard": [
                    _round(h, 12) for h in model.baseline_cumulative_hazard
                ],
            },
        },
        "slaTargets": sorted({_round(case.sla_hours, 1) for case in cases}),
    }

    # -- Power BI lint -----------------------------------------------------
    tabular = load_model(model_path)
    findings = lint(tabular)
    daxlint = {
        "model": tabular.name,
        "tables": len(tabular.tables),
        "measures": len(tabular.measures()),
        "relationships": len(tabular.relationships),
        "ruleCount": len(ALL_RULES),
        "rules": [
            {"code": rule.code, "severity": rule.severity.value, "summary": rule.summary}
            for rule in ALL_RULES
        ],
        "findings": [finding.to_dict() for finding in findings],
    }

    breached = sum(1 for case in resolved if case.breached)
    return {
        "version": __version__,
        "generated": datetime.now().strftime("%Y-%m-%d"),
        "summary": {
            "cases": len(cases),
            "events": len(log),
            "resolved": len(resolved),
            "open": len(cases) - len(resolved),
            "breachRate": _round(breached / max(1, len(resolved)), 4),
        },
        "mining": mining,
        "spc": spc,
        "sla": sla,
        "daxlint": daxlint,
    }


def write_payload(path: str, *, as_javascript: bool = False, **kwargs) -> str:
    """Write the payload to ``path``.

    With ``as_javascript`` the JSON is wrapped in an assignment to a global, so
    the demo page can load it with a plain ``<script src>`` and still work from
    a ``file://`` URL, where ``fetch`` is blocked by the same-origin policy.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = build_payload(**kwargs)
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    with open(path, "w", encoding="utf-8") as handle:
        if as_javascript:
            handle.write("window.OPSLAB_DATA = %s;\n" % body)
        else:
            handle.write(body + "\n")
    return path
