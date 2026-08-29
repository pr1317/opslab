"""Command line interface.

``opslab demo`` runs every module end to end against freshly generated data and
writes the outputs to a folder; the other subcommands run one module against
your own CSV extract.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Tuple

from opslab import __version__
from opslab.calendar import BusinessCalendar
from opslab.eventlog import Case, CaseTable, EventLog

__all__ = ["main"]


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _ensure_directory(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def _write_rows(path: str, rows: Sequence[dict]) -> None:
    """Write a list of uniform dictionaries to CSV."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text.rstrip("\n") + "\n")


def _completion_date(case: Case, calendar: BusinessCalendar) -> date:
    """The date a resolved case finished, from its arrival plus working hours."""
    return calendar.add_working_hours(case.arrived, case.duration_hours).date()


def _iso_week(day: date) -> str:
    year, week, _ = day.isocalendar()
    return "%04d-W%02d" % (year, week)


def _weekly_breach_table(
    cases: CaseTable, calendar: BusinessCalendar
) -> Tuple[List[str], List[int], List[int], List[float]]:
    """Weekly resolved volume, breach count and mean handling hours.

    Cases are bucketed by *completion* week, which is the week the operation
    actually delivered them; bucketing by arrival mixes finished and unfinished
    work in the same point.
    """
    buckets: Dict[str, List[Case]] = defaultdict(list)
    for case in cases:
        if not case.resolved:
            continue
        buckets[_iso_week(_completion_date(case, calendar))].append(case)

    labels = sorted(buckets)
    sizes = [len(buckets[w]) for w in labels]
    breaches = [sum(1 for c in buckets[w] if c.breached) for w in labels]
    mean_hours = [
        sum(c.duration_hours for c in buckets[w]) / len(buckets[w]) for w in labels
    ]
    return labels, breaches, sizes, mean_hours


def _default_rules():
    """The reference control set for the simulated pensions process."""
    from opslab.processmining import (
        MaxOccurrences,
        Ordering,
        RequiredActivities,
        SegregationOfDuties,
        StartActivities,
    )

    return [
        StartActivities(("Receive Request",)),
        RequiredActivities(("Receive Request",)),
        SegregationOfDuties(
            ("Calculate Benefit", "Apply Adjustment", "Draft Response"), "Peer Check"
        ),
        Ordering("Peer Check", "Release Payment"),
        MaxOccurrences("Rework Calculation", 1),
    ]


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------
def command_simulate(args: argparse.Namespace) -> int:
    """Generate a synthetic event log and case table."""
    from opslab.simulate import SimulationConfig, simulate

    config = SimulationConfig(n_cases=args.cases, seed=args.seed)
    result = simulate(config)
    out = _ensure_directory(args.out)
    result.log.to_csv(os.path.join(out, "events.csv"))
    result.cases.to_csv(os.path.join(out, "cases.csv"))
    print(
        "Wrote %d events across %d cases to %s (%.1f%% still open at the extract date)"
        % (len(result.log), len(result.cases), out, 100.0 * result.censoring_rate)
    )
    return 0


def command_mine(args: argparse.Namespace) -> int:
    """Discover a process model and report variants, rework and bottlenecks."""
    from opslab.processmining import (
        bottlenecks,
        check_conformance,
        discover_dfg,
        handover_network,
        rework_statistics,
        to_dot,
        to_mermaid,
        to_svg,
        variants,
    )

    log = EventLog.from_csv(args.events)
    calendar = BusinessCalendar()
    graph = discover_dfg(log, calendar)

    print("Discovered %d activities and %d transitions across %d cases"
          % (len(graph.nodes), len(graph.edges), graph.case_count))

    variant_list = variants(log)
    covered = sum(v.count for v in variant_list[: args.variants])
    print("\nTop %d of %d variants (%.1f%% of cases):"
          % (min(args.variants, len(variant_list)), len(variant_list),
             100.0 * covered / max(1, graph.case_count)))
    for variant in variant_list[: args.variants]:
        print("  %4d  %s" % (variant.count, " -> ".join(variant.trace)))

    print("\nBottlenecks by total working hours consumed:")
    for source, target, total, mean_hours, frequency in bottlenecks(log, calendar, args.top, graph):
        print("  %-26s -> %-26s %9.0fh total  %6.1fh mean  n=%d"
              % (source, target, total, mean_hours, frequency))

    print("\nRework:")
    rework = rework_statistics(log)
    ranked = sorted(rework.values(), key=lambda s: -s.rework_rate)
    for stats in ranked[: args.top]:
        if stats.rework_rate <= 0:
            break
        print("  %-26s repeated in %5.1f%% of its cases (first-time-right %5.1f%%)"
              % (stats.activity, 100 * stats.rework_rate, 100 * stats.first_time_right))

    report = check_conformance(log, _default_rules())
    print("\n" + report.to_text())

    handovers = handover_network(log)
    if handovers:
        print("\nBusiest handovers:")
        for edge in handovers[:5]:
            print("  %-10s -> %-10s %d" % (edge.source, edge.target, edge.count))

    if args.out:
        out = _ensure_directory(args.out)
        _write_text(os.path.join(out, "process_map.dot"),
                    to_dot(graph, min_edge_frequency=args.min_edge_frequency,
                           title="Discovered process"))
        _write_text(os.path.join(out, "process_map.mmd"),
                    to_mermaid(graph, min_edge_frequency=args.min_edge_frequency))
        _write_text(os.path.join(out, "process_map.svg"),
                    to_svg(graph, min_edge_frequency=args.min_edge_frequency))
        _write_rows(os.path.join(out, "variants.csv"),
                    [{"count": v.count, "length": v.length, "trace": " -> ".join(v.trace)}
                     for v in variant_list])
        _write_text(os.path.join(out, "conformance.txt"), report.to_text(examples=5))
        print("\nWrote process_map.svg (open it in a browser), process_map.dot, "
              "process_map.mmd, variants.csv and conformance.txt to %s" % out)
    return 0


def command_spc(args: argparse.Namespace) -> int:
    """Chart weekly SLA breach rate and handling time, and assess capability."""
    from opslab.spc import capability, individuals_chart, p_chart

    cases = CaseTable.from_csv(args.cases)
    calendar = BusinessCalendar()
    labels, breaches, sizes, mean_hours = _weekly_breach_table(cases, calendar)
    if len(labels) < 2:
        print("Not enough completed weeks to chart", file=sys.stderr)
        return 1

    breach_chart = p_chart(breaches, sizes, labels)
    print("Weekly SLA breach rate")
    print(breach_chart.to_text())
    for note in breach_chart.notes:
        print("  ! %s" % note)

    hours_chart = individuals_chart(mean_hours, labels)
    print("\nWeekly mean handling hours")
    print(hours_chart.to_text())

    signals = breach_chart.signals() + hours_chart.signals()
    if signals:
        print("\n%d special-cause signal(s); the earliest is week %s."
              % (len(signals), min(p.label for p in signals)))
    else:
        print("\nBoth charts are in statistical control: the variation seen is "
              "the process behaving normally, not something to investigate.")

    resolved = [c for c in cases if c.resolved]
    if resolved:
        target = args.sla_hours or max(c.sla_hours for c in resolved)
        report = capability([c.duration_hours for c in resolved], usl=target)
        print("\nCapability of case turnaround against a %.0f working-hour target"
              % target)
        print(report.to_text())

    if args.out:
        out = _ensure_directory(args.out)
        _write_rows(os.path.join(out, "spc_breach_rate.csv"), breach_chart.to_rows())
        _write_rows(os.path.join(out, "spc_handling_hours.csv"), hours_chart.to_rows())
        print("\nWrote spc_breach_rate.csv and spc_handling_hours.csv to %s" % out)
    return 0


def command_sla(args: argparse.Namespace) -> int:
    """Fit survival models to case turnaround, respecting open cases."""
    from opslab.sla import (
        concordance_index,
        design_matrix,
        fit_cox,
        kaplan_meier,
        logrank_test,
    )

    cases = CaseTable.from_csv(args.cases)
    durations = [c.duration_hours for c in cases]
    events = [1 if c.resolved else 0 for c in cases]
    open_cases = len(cases) - sum(events)

    print("%d cases, %d resolved, %d still open at the extract (%.1f%% censored)"
          % (len(cases), sum(events), open_cases, 100.0 * open_cases / max(1, len(cases))))

    naive = [c.duration_hours for c in cases if c.resolved]
    curve = kaplan_meier(durations, events)
    median = curve.median
    print("\nMedian time to resolution")
    print("  closed cases only : %.1f working hours" % (sorted(naive)[len(naive) // 2]))
    print("  Kaplan-Meier      : %s"
          % ("%.1f working hours" % median if median is not None
             else "not reached within the follow-up"))
    print("  The first figure conditions on the case having finished, so it "
          "systematically understates how long work takes.")

    horizon = args.horizon or max(durations)
    print("  restricted mean to %.0fh: %.1f working hours"
          % (horizon, curve.restricted_mean(horizon)))

    # Compare two groups if the attribute is present.
    if args.group:
        groups: Dict[str, List[Case]] = defaultdict(list)
        for case in cases:
            groups[case.attributes.get(args.group, "")].append(case)
        if len(groups) == 2:
            (label_a, cases_a), (label_b, cases_b) = sorted(groups.items())
            result = logrank_test(
                [c.duration_hours for c in cases_a], [1 if c.resolved else 0 for c in cases_a],
                [c.duration_hours for c in cases_b], [1 if c.resolved else 0 for c in cases_b],
                label_a or "(blank)", label_b or "(blank)",
            )
            print("\n" + result.to_text())
        else:
            print("\nSkipping the log-rank test: %r has %d levels, not 2."
                  % (args.group, len(groups)))

    numeric = [n for n in (args.numeric or []) if n]
    binary = {}
    for item in args.binary or []:
        if "=" not in item:
            raise SystemExit("--binary expects attribute=level, got %r" % item)
        attribute, level = item.split("=", 1)
        binary[attribute] = level
    if not numeric and not binary:
        print("\nNo covariates given (--numeric / --binary); skipping the Cox model.")
        return 0

    matrix = design_matrix(cases, numeric=numeric, binary=binary)
    model = fit_cox(durations, events, matrix.rows, names=matrix.names)
    print("\n" + model.result.to_text())
    print("  A hazard ratio above 1 means the case clears faster; below 1 means "
          "it lingers.")

    c_index, pairs = concordance_index(durations, events, model.risk_scores(matrix.rows))
    print("  concordance index %.4f over %d comparable pairs" % (c_index, pairs))

    sla_target = args.sla_hours or (cases.cases[0].sla_hours if len(cases) else 40.0)
    scored = sorted(
        (
            (model.breach_probability(row, sla_target), case_id)
            for row, case_id in zip(matrix.rows, matrix.case_ids)
        ),
        reverse=True,
    )
    print("\nHighest predicted breach risk against a %.0f-hour target:" % sla_target)
    for probability, case_id in scored[:10]:
        print("  %s  %.1f%%" % (case_id, 100.0 * probability))

    if args.out:
        out = _ensure_directory(args.out)
        _write_rows(os.path.join(out, "km_curve.csv"), curve.to_rows())
        _write_rows(os.path.join(out, "cox_coefficients.csv"), model.result.to_rows())
        _write_rows(
            os.path.join(out, "breach_risk.csv"),
            [{"case_id": cid, "breach_probability": round(p, 6)} for p, cid in scored],
        )
        print("\nWrote km_curve.csv, cox_coefficients.csv and breach_risk.csv to %s" % out)
    return 0


def command_daxlint(args: argparse.Namespace) -> int:
    """Lint a Power BI tabular model."""
    from opslab.daxlint import ALL_RULES, Severity, lint, load_model

    if args.list_rules:
        for rule in ALL_RULES:
            print("%-8s %-8s %s" % (rule.code, rule.severity.value, rule.summary))
        return 0

    from opslab import data

    model = load_model(data.model_path() if args.sample else args.model)
    findings = lint(model, select=args.select, ignore=args.ignore)

    if args.format == "json":
        print(json.dumps(
            {
                "model": model.name,
                "source": model.source_path,
                "findings": [f.to_dict() for f in findings],
            },
            indent=2,
        ))
    else:
        print("%s: %d table(s), %d measure(s), %d relationship(s)"
              % (model.name, len(model.tables), len(model.measures()),
                 len(model.relationships)))
        if not findings:
            print("No findings.")
        for finding in findings:
            print(finding.to_text())
            if args.explain and finding.remedy:
                print("         -> %s" % finding.remedy)
        counts = {severity: 0 for severity in Severity}
        for finding in findings:
            counts[finding.severity] += 1
        print("\n%d error(s), %d warning(s), %d info"
              % (counts[Severity.ERROR], counts[Severity.WARNING], counts[Severity.INFO]))

    threshold = {"error": (Severity.ERROR,),
                 "warning": (Severity.ERROR, Severity.WARNING),
                 "info": (Severity.ERROR, Severity.WARNING, Severity.INFO),
                 "never": ()}[args.fail_on]
    return 1 if any(f.severity in threshold for f in findings) else 0


def command_try(args: argparse.Namespace) -> int:
    """Run every module and write one HTML report you can open in a browser.

    This is the front door. It defaults to the sample shipped inside the package,
    so it needs no arguments, no simulation step and no network; pointing it at
    your own extract with ``--events``/``--cases`` produces the same report over
    your data, minus the generating coefficients, which only exist for the sample.
    """
    import shutil
    import webbrowser

    from opslab import data
    from opslab.report import write_report

    bundled = not (args.events or args.cases or args.model)
    events = args.events or data.events_path()
    cases = args.cases or data.cases_path()
    model = args.model or data.model_path()
    out = _ensure_directory(args.out)

    if bundled:
        source_note = (
            "The data is the sample shipped with the package: a synthetic "
            "life-and-pensions back office, six months of it, with a backlog surge "
            "deliberately injected part way through."
        )
        ground_truth = _ground_truth()
    else:
        source_note = "The data is your own extract."
        ground_truth = None

    print("Reading %s" % events)
    print("        %s" % cases)
    print("        %s" % model)

    report_path = write_report(
        os.path.join(out, "report.html"),
        events_path=events,
        cases_path=cases,
        model_path=model,
        ground_truth=ground_truth,
        source_note=source_note,
    )

    if bundled:
        for source in (events, cases):
            shutil.copyfile(source, os.path.join(out, os.path.basename(source)))
        print("\nCopied the sample CSVs alongside the report so you have something "
              "to point the other subcommands at.")

    print("\nWrote %s" % report_path)
    print("Open it with:  file://%s" % os.path.abspath(report_path))
    if args.open:
        try:
            webbrowser.open("file://" + os.path.abspath(report_path))
        except Exception as error:  # pragma: no cover - depends on the desktop
            print("Could not launch a browser (%s); open the path above." % error)
    next_events = os.path.join(out, os.path.basename(events)) if bundled else events
    print("\nThen run any single module on the same data, for example:")
    print("  opslab mine --events %s" % next_events)
    return 0


def command_export(args: argparse.Namespace) -> int:
    """Write every fitted result as JSON, for a front end to draw."""
    from opslab import data
    from opslab.exporter import write_payload

    bundled = not (args.events or args.cases or args.model)
    path = write_payload(
        args.out,
        as_javascript=args.out.endswith(".js"),
        events_path=args.events or data.events_path(),
        cases_path=args.cases or data.cases_path(),
        model_path=args.model or data.model_path(),
        ground_truth=_ground_truth() if bundled else None,
    )
    print("Wrote %s (%.0f KB)" % (path, os.path.getsize(path) / 1024.0))
    return 0


def _ground_truth() -> Dict[str, float]:
    """The coefficients the sample was generated from, for the report to check against."""
    from opslab.simulate.generator import GROUND_TRUTH_BETA

    return dict(GROUND_TRUTH_BETA)


def command_demo(args: argparse.Namespace) -> int:
    """Run every module end to end against freshly generated data."""
    from opslab import data
    from opslab.daxlint import lint, load_model
    from opslab.simulate import SimulationConfig, simulate

    out = _ensure_directory(args.out)
    config = SimulationConfig(n_cases=args.cases, seed=args.seed)
    result = simulate(config)
    events_path = os.path.join(out, "events.csv")
    cases_path = os.path.join(out, "cases.csv")
    result.log.to_csv(events_path)
    result.cases.to_csv(cases_path)

    print("=" * 72)
    print("1. Simulated operation")
    print("=" * 72)
    print("%d events across %d cases; %.1f%% still open at the extract date."
          % (len(result.log), len(result.cases), 100.0 * result.censoring_rate))
    print("A backlog surge was injected from %s." % config.shift_date)

    print("\n" + "=" * 72)
    print("2. Process mining")
    print("=" * 72)
    command_mine(argparse.Namespace(
        events=events_path, out=out, variants=5, top=6, min_edge_frequency=5,
    ))

    print("\n" + "=" * 72)
    print("3. Statistical process control")
    print("=" * 72)
    command_spc(argparse.Namespace(cases=cases_path, out=out, sla_hours=None))

    print("\n" + "=" * 72)
    print("4. SLA survival analysis")
    print("=" * 72)
    command_sla(argparse.Namespace(
        cases=cases_path, out=out, horizon=None, group="priority",
        numeric=["complexity", "backlog_index", "awaiting_third_party"],
        binary=["priority=Urgent", "channel=Post"],
        sla_hours=None,
    ))

    print("\n" + "=" * 72)
    print("5. Power BI model lint")
    print("=" * 72)
    sample = args.model or data.model_path()
    if os.path.exists(sample):
        model = load_model(sample)
        findings = lint(model)
        print("%s: %d finding(s) across %d measures"
              % (model.name, len(findings), len(model.measures())))
        for finding in findings[:12]:
            print(finding.to_text())
        _write_rows(os.path.join(out, "lint_findings.csv"),
                    [f.to_dict() for f in findings])
        print("\nWrote lint_findings.csv to %s" % out)
    else:
        print("Sample model not found at %s; skipping." % sample)

    print("\nAll outputs written to %s" % out)
    return 0


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="opslab",
        description="Operations analytics toolkit for BFSI back-office processes.",
    )
    parser.add_argument("--version", action="version", version="opslab %s" % __version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    try_parser = subparsers.add_parser(
        "try", help="run everything on the bundled sample and write an HTML report"
    )
    try_parser.add_argument("--out", default="out/try", help="output directory")
    try_parser.add_argument("--events", default="", help="your own event log CSV")
    try_parser.add_argument("--cases", default="", help="your own case table CSV")
    try_parser.add_argument("--model", default="", help="your own .bim, .tmdl or PBIP folder")
    try_parser.add_argument("--open", action="store_true",
                            help="open the report in a browser when it is written")
    try_parser.set_defaults(func=command_try)

    export_parser = subparsers.add_parser(
        "export", help="write every fitted result as JSON (or .js) for a front end"
    )
    export_parser.add_argument("--out", default="out/opslab.json",
                               help="output path; a .js suffix wraps it for a <script> tag")
    export_parser.add_argument("--events", default="", help="your own event log CSV")
    export_parser.add_argument("--cases", default="", help="your own case table CSV")
    export_parser.add_argument("--model", default="", help="your own .bim, .tmdl or PBIP folder")
    export_parser.set_defaults(func=command_export)

    simulate_parser = subparsers.add_parser(
        "simulate", help="generate a synthetic event log and case table"
    )
    simulate_parser.add_argument("--cases", type=int, default=1500)
    simulate_parser.add_argument("--seed", type=int, default=20260828)
    simulate_parser.add_argument("--out", default="out")
    simulate_parser.set_defaults(func=command_simulate)

    mine_parser = subparsers.add_parser(
        "mine", help="discover a process model from an event log"
    )
    mine_parser.add_argument("--events", required=True, help="event log CSV")
    mine_parser.add_argument("--out", default="", help="directory for the process map")
    mine_parser.add_argument("--variants", type=int, default=8)
    mine_parser.add_argument("--top", type=int, default=10)
    mine_parser.add_argument("--min-edge-frequency", type=int, default=1, dest="min_edge_frequency")
    mine_parser.set_defaults(func=command_mine)

    spc_parser = subparsers.add_parser(
        "spc", help="control-chart the weekly breach rate and handling time"
    )
    spc_parser.add_argument("--cases", required=True, help="case table CSV")
    spc_parser.add_argument("--out", default="")
    spc_parser.add_argument("--sla-hours", type=float, default=None, dest="sla_hours")
    spc_parser.set_defaults(func=command_spc)

    sla_parser = subparsers.add_parser(
        "sla", help="fit survival models to case turnaround time"
    )
    sla_parser.add_argument("--cases", required=True, help="case table CSV")
    sla_parser.add_argument("--out", default="")
    sla_parser.add_argument("--horizon", type=float, default=None)
    sla_parser.add_argument("--group", default="", help="attribute for a two-group log-rank test")
    sla_parser.add_argument("--numeric", action="append", help="numeric covariate (repeatable)")
    sla_parser.add_argument("--binary", action="append",
                            help="binary covariate as attribute=level (repeatable)")
    sla_parser.add_argument("--sla-hours", type=float, default=None, dest="sla_hours")
    sla_parser.set_defaults(func=command_sla)

    lint_parser = subparsers.add_parser(
        "daxlint", help="lint a Power BI tabular model (.bim, .tmdl or PBIP folder)"
    )
    lint_parser.add_argument("model", nargs="?", default="", help="model file or folder")
    lint_parser.add_argument("--sample", action="store_true",
                             help="lint the bundled example model instead of your own")
    lint_parser.add_argument("--format", choices=("text", "json"), default="text")
    lint_parser.add_argument("--select", action="append", help="only these rule codes")
    lint_parser.add_argument("--ignore", action="append", help="skip these rule codes")
    lint_parser.add_argument("--explain", action="store_true", help="print the remedy for each finding")
    lint_parser.add_argument("--fail-on", choices=("error", "warning", "info", "never"),
                             default="error", dest="fail_on")
    lint_parser.add_argument("--list-rules", action="store_true", dest="list_rules")
    lint_parser.set_defaults(func=command_daxlint)

    demo_parser = subparsers.add_parser(
        "demo", help="run every module end to end and write the outputs"
    )
    demo_parser.add_argument("--cases", type=int, default=1500)
    demo_parser.add_argument("--seed", type=int, default=20260828)
    demo_parser.add_argument("--out", default="out/demo")
    demo_parser.add_argument("--model", default="", help="model file for the lint step")
    demo_parser.set_defaults(func=command_demo)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``opslab`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "daxlint" and not (args.model or args.list_rules or args.sample):
        parser.error("daxlint requires a model path (or --sample, or --list-rules)")
    try:
        return int(args.func(args) or 0)
    except (ValueError, OSError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
