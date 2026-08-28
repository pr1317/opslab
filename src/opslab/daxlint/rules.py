"""Lint rules for tabular models and DAX.

Each rule states the concrete consequence of ignoring it, because a linter that
only says "this is not best practice" gets switched off.  Rules split into two
families: ``MOD`` rules read model metadata, ``DAX`` rules read expressions.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

from opslab.daxlint.lexer import (
    Token,
    TokenKind,
    column_references,
    function_calls,
    tokenize,
)
from opslab.daxlint.model import Measure, TabularModel

__all__ = ["Severity", "Finding", "Rule", "ALL_RULES", "lint"]


class Severity(Enum):
    """How much a finding matters."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Finding:
    """One rule violation against one object."""

    code: str
    severity: Severity
    object_name: str
    message: str
    remedy: str = ""
    line: int = 0

    def to_dict(self) -> dict:
        """A JSON-ready representation."""
        return {
            "code": self.code,
            "severity": self.severity.value,
            "object": self.object_name,
            "message": self.message,
            "remedy": self.remedy,
            "line": self.line,
        }

    def to_text(self) -> str:
        """A single ``code severity object: message`` line."""
        location = ":%d" % self.line if self.line else ""
        return "%-8s %-7s %s%s: %s" % (
            self.code, self.severity.value, self.object_name, location, self.message
        )


@dataclass(frozen=True)
class Rule:
    """A named check over a whole model."""

    code: str
    severity: Severity
    summary: str
    check: Callable[[TabularModel], List[Finding]]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
# Matches an identifier suffix whether it is separated ("case_id") or
# written CamelCase ("TeamID"). "no" is deliberately excluded: it would
# fire on innocent words ending in those two letters.
_KEY_NAME = re.compile(
    r"(?:^|[_\s]|(?<=[a-z]))(id|key|code|guid|number)$", re.IGNORECASE
)
_MONEY_NAME = re.compile(
    r"(amount|value|price|cost|premium|balance|fee|charge|payment|salary|revenue)",
    re.IGNORECASE,
)
_AGGREGATORS = {"SUM", "AVERAGE", "MIN", "MAX", "COUNT", "COUNTA", "DISTINCTCOUNT"}
_ITERATOR_FOR = {"SUMX": "SUM", "AVERAGEX": "AVERAGE", "MINX": "MIN", "MAXX": "MAX"}
_NUMERIC_TYPES = {"double", "int64", "decimal"}


def _looks_like_key(name: str) -> bool:
    return bool(_KEY_NAME.search(name))


def _measure_line(measure: Measure, token: Optional[Token]) -> int:
    return token.line if token is not None else 0


def _iter_measures(model: TabularModel) -> Iterable[Measure]:
    return (m for m in model.measures() if m.expression)


# --------------------------------------------------------------------------
# model rules
# --------------------------------------------------------------------------
def _mod001_measure_without_description(model: TabularModel) -> List[Finding]:
    return [
        Finding(
            "MOD001",
            Severity.INFO,
            measure.qualified_name,
            "measure has no description",
            "Add a description; it surfaces as the tooltip in the field list "
            "and is the only in-product documentation report authors ever see.",
        )
        for measure in model.measures()
        if not measure.description.strip()
    ]


def _mod002_measure_without_folder(model: TabularModel) -> List[Finding]:
    measures = model.measures()
    if len(measures) < 10:
        return []
    return [
        Finding(
            "MOD002",
            Severity.INFO,
            measure.qualified_name,
            "measure has no display folder in a model with %d measures" % len(measures),
            "Group measures into display folders so the field list stays navigable.",
        )
        for measure in measures
        if not measure.display_folder.strip()
    ]


def _mod003_bidirectional_relationship(model: TabularModel) -> List[Finding]:
    return [
        Finding(
            "MOD003",
            Severity.WARNING,
            relationship.describe(),
            "relationship filters in both directions",
            "Bi-directional filters can make a model ambiguous and slow down "
            "queries. Prefer a one-way relationship plus CROSSFILTER() in the "
            "specific measures that genuinely need it.",
        )
        for relationship in model.relationships
        if relationship.is_bidirectional
    ]


def _mod004_inactive_relationship_unused(model: TabularModel) -> List[Finding]:
    inactive = [r for r in model.relationships if not r.is_active]
    if not inactive:
        return []
    uses_userelationship = any(
        "USERELATIONSHIP" in m.expression.upper() for m in _iter_measures(model)
    )
    if uses_userelationship:
        return []
    return [
        Finding(
            "MOD004",
            Severity.WARNING,
            relationship.describe(),
            "relationship is inactive but no measure calls USERELATIONSHIP",
            "An inactive relationship no measure activates is dead weight: "
            "either activate it, delete it, or add the USERELATIONSHIP that "
            "was intended.",
        )
        for relationship in inactive
    ]


def _mod005_auto_date_tables(model: TabularModel) -> List[Finding]:
    auto = model.auto_date_tables
    if not auto:
        return []
    return [
        Finding(
            "MOD005",
            Severity.WARNING,
            model.name,
            "auto date/time is on: %d hidden date table(s) generated" % len(auto),
            "Each date column gets its own hidden table, inflating the model "
            "and preventing consistent time intelligence. Turn auto date/time "
            "off and mark a single shared date table instead.",
        )
    ]


def _mod006_currency_as_double(model: TabularModel) -> List[Finding]:
    findings: List[Finding] = []
    for column in model.columns():
        if column.data_type.lower() != "double":
            continue
        if not _MONEY_NAME.search(column.name):
            continue
        findings.append(
            Finding(
                "MOD006",
                Severity.WARNING,
                column.qualified_name,
                "monetary column is typed as a floating-point double",
                "Use the fixed decimal (currency) type. Binary floating point "
                "cannot represent most decimal fractions exactly, so totals "
                "drift by fractions of a penny and reconciliations fail.",
            )
        )
    return findings


def _mod007_key_column_summarised(model: TabularModel) -> List[Finding]:
    findings: List[Finding] = []
    for column in model.columns():
        if column.summarize_by.lower() in ("none", ""):
            continue
        if column.data_type.lower() not in _NUMERIC_TYPES:
            continue
        if not (column.is_key or _looks_like_key(column.name)):
            continue
        findings.append(
            Finding(
                "MOD007",
                Severity.ERROR,
                column.qualified_name,
                "identifier column defaults to being aggregated (summarizeBy=%s)"
                % column.summarize_by,
                "Set summarizeBy to none. Otherwise dragging the column into a "
                "visual silently produces the sum of a set of IDs, which looks "
                "like a real number and is not.",
            )
        )
    return findings


def _mod008_visible_foreign_key(model: TabularModel) -> List[Finding]:
    keys: Set[str] = set()
    for relationship in model.relationships:
        keys.add("%s|%s" % (relationship.from_table.lower(), relationship.from_column.lower()))
    findings: List[Finding] = []
    for column in model.columns():
        if "%s|%s" % (column.table.lower(), column.name.lower()) not in keys:
            continue
        if column.is_hidden:
            continue
        findings.append(
            Finding(
                "MOD008",
                Severity.INFO,
                column.qualified_name,
                "foreign key column is visible to report authors",
                "Hide relationship key columns. Slicing by the fact-side key "
                "instead of the dimension bypasses the relationship and gives "
                "wrong totals.",
            )
        )
    return findings


def _mod009_no_marked_date_table(model: TabularModel) -> List[Finding]:
    if any(t.is_date_table for t in model.tables):
        return []
    uses_time_intelligence = any(
        re.search(r"\b(DATEADD|SAMEPERIODLASTYEAR|TOTALYTD|DATESYTD|PARALLELPERIOD)\b",
                  m.expression, re.IGNORECASE)
        for m in _iter_measures(model)
    )
    if not uses_time_intelligence:
        return []
    return [
        Finding(
            "MOD009",
            Severity.ERROR,
            model.name,
            "time intelligence is used but no table is marked as a date table",
            "Mark the date dimension as a date table. Without it, functions "
            "like DATEADD and TOTALYTD silently return wrong results at the "
            "edges of the calendar.",
        )
    ]


def _mod010_measure_without_format(model: TabularModel) -> List[Finding]:
    return [
        Finding(
            "MOD010",
            Severity.INFO,
            measure.qualified_name,
            "measure has no format string",
            "Set a format string on the measure so every visual renders it "
            "consistently, rather than relying on each report author.",
        )
        for measure in model.measures()
        if not measure.format_string.strip() and not measure.is_hidden
    ]


def _mod011_duplicate_measure_logic(model: TabularModel) -> List[Finding]:
    def normalise(expression: str) -> str:
        tokens = tokenize(expression)
        return " ".join(t.text.upper() for t in tokens)

    grouped: Dict[str, List[Measure]] = defaultdict(list)
    for measure in _iter_measures(model):
        grouped[normalise(measure.expression)].append(measure)

    findings: List[Finding] = []
    for measures in grouped.values():
        if len(measures) < 2:
            continue
        names = ", ".join(m.qualified_name for m in measures[1:])
        findings.append(
            Finding(
                "MOD011",
                Severity.WARNING,
                measures[0].qualified_name,
                "identical DAX to: %s" % names,
                "Duplicate definitions drift apart the first time one of them "
                "is corrected. Keep one measure and reference it.",
            )
        )
    return findings


def _mod012_calculated_column_aggregation(model: TabularModel) -> List[Finding]:
    findings: List[Finding] = []
    for column in model.columns():
        if not column.is_calculated:
            continue
        calls = {t.upper for t in function_calls(tokenize(column.expression))}
        aggregated = calls & (_AGGREGATORS | set(_ITERATOR_FOR))
        if not aggregated:
            continue
        findings.append(
            Finding(
                "MOD012",
                Severity.WARNING,
                column.qualified_name,
                "calculated column aggregates (%s) and is stored per row"
                % ", ".join(sorted(aggregated)),
                "A calculated column is evaluated once at refresh and stored, "
                "so it cannot respond to slicers and it costs memory in every "
                "row. Express this as a measure instead.",
            )
        )
    return findings


# --------------------------------------------------------------------------
# DAX rules
# --------------------------------------------------------------------------
def _dax001_division_operator(model: TabularModel) -> List[Finding]:
    findings: List[Finding] = []
    for measure in _iter_measures(model):
        for token in tokenize(measure.expression):
            if token.kind is TokenKind.OPERATOR and token.text == "/":
                findings.append(
                    Finding(
                        "DAX001",
                        Severity.WARNING,
                        measure.qualified_name,
                        "uses the '/' operator instead of DIVIDE()",
                        "DIVIDE() handles division by zero explicitly and is "
                        "optimised in the engine; '/' raises an error or "
                        "returns infinity depending on the operands.",
                        token.line,
                    )
                )
                break
    return findings


def _dax002_unqualified_column(model: TabularModel) -> List[Finding]:
    measure_names = model.measure_names()
    columns_by_name = model.column_names()
    findings: List[Finding] = []
    for measure in _iter_measures(model):
        for reference in column_references(tokenize(measure.expression)):
            if reference.qualified:
                continue
            lowered = reference.name.lower()
            if lowered in measure_names:
                continue
            if lowered not in columns_by_name:
                continue
            findings.append(
                Finding(
                    "DAX002",
                    Severity.WARNING,
                    measure.qualified_name,
                    "column [%s] is referenced without its table" % reference.name,
                    "Always write column references as Table[Column] and "
                    "measure references as [Measure]. The convention is what "
                    "lets a reader tell the two apart at a glance.",
                    reference.token.line,
                )
            )
    return findings


def _dax003_qualified_measure(model: TabularModel) -> List[Finding]:
    measure_names = model.measure_names()
    columns_by_name = model.column_names()
    findings: List[Finding] = []
    for measure in _iter_measures(model):
        for reference in column_references(tokenize(measure.expression)):
            if not reference.qualified:
                continue
            lowered = reference.name.lower()
            if lowered not in measure_names or lowered in columns_by_name:
                continue
            findings.append(
                Finding(
                    "DAX003",
                    Severity.INFO,
                    measure.qualified_name,
                    "measure reference %s[%s] is table-qualified"
                    % (reference.table, reference.name),
                    "Measures belong to the model, not the table they are "
                    "filed under; qualifying them breaks if the measure is "
                    "moved, and reads like a column reference.",
                    reference.token.line,
                )
            )
    return findings


def _dax004_filter_whole_table(model: TabularModel) -> List[Finding]:
    findings: List[Finding] = []
    for measure in _iter_measures(model):
        tokens = tokenize(measure.expression)
        for index, token in enumerate(tokens):
            if token.upper != "FILTER" or index + 2 >= len(tokens):
                continue
            if tokens[index + 1].text != "(":
                continue
            argument = tokens[index + 2]
            # FILTER(Table, ...) rather than FILTER(ALL(Table[Col]), ...).
            if argument.kind not in (TokenKind.IDENTIFIER, TokenKind.QUOTED):
                continue
            follows = tokens[index + 3] if index + 3 < len(tokens) else None
            if follows is not None and follows.text == "(":
                continue
            if model.table(argument.value) is None:
                continue
            findings.append(
                Finding(
                    "DAX004",
                    Severity.WARNING,
                    measure.qualified_name,
                    "FILTER() iterates the whole '%s' table" % argument.value,
                    "Filtering an entire table materialises every row and every "
                    "column it references. Filter the single column instead - "
                    "FILTER(ALL(Table[Column]), ...) - or use a plain boolean "
                    "predicate inside CALCULATE.",
                    token.line,
                )
            )
    return findings


def _dax005_error_suppression(model: TabularModel) -> List[Finding]:
    findings: List[Finding] = []
    for measure in _iter_measures(model):
        for token in function_calls(tokenize(measure.expression)):
            if token.upper in ("IFERROR", "ISERROR"):
                findings.append(
                    Finding(
                        "DAX005",
                        Severity.WARNING,
                        measure.qualified_name,
                        "uses %s() to swallow errors" % token.upper,
                        "Error handling forces row-by-row evaluation and hides "
                        "the defect rather than fixing it. Guard the specific "
                        "condition instead - DIVIDE() for division, or an "
                        "explicit IF on the value that can be missing.",
                        token.line,
                    )
                )
                break
    return findings


def _dax006_iterator_over_column(model: TabularModel) -> List[Finding]:
    findings: List[Finding] = []
    for measure in _iter_measures(model):
        tokens = tokenize(measure.expression)
        for index, token in enumerate(tokens):
            replacement = _ITERATOR_FOR.get(token.upper)
            if replacement is None:
                continue
            # Match SUMX( Table, Table[Column] ) exactly: a table, a comma, a
            # single qualified column reference, then the closing bracket.
            window = tokens[index + 1 : index + 8]
            texts = [t.text for t in window]
            if len(window) < 6:
                continue
            if texts[0] != "(" or texts[2] != ",":
                continue
            if window[1].kind not in (TokenKind.IDENTIFIER, TokenKind.QUOTED):
                continue
            if window[3].kind not in (TokenKind.IDENTIFIER, TokenKind.QUOTED):
                continue
            if window[4].kind is not TokenKind.BRACKET or texts[5] != ")":
                continue
            if window[1].value.lower() != window[3].value.lower():
                continue
            findings.append(
                Finding(
                    "DAX006",
                    Severity.INFO,
                    measure.qualified_name,
                    "%s() over a plain column could be %s()"
                    % (token.upper, replacement),
                    "The iterator adds a row context for no benefit when the "
                    "expression is just a column. %s() is clearer and the "
                    "engine optimises it better." % replacement,
                    token.line,
                )
            )
    return findings


def _dax007_earlier(model: TabularModel) -> List[Finding]:
    findings: List[Finding] = []
    for measure in _iter_measures(model):
        for token in function_calls(tokenize(measure.expression)):
            if token.upper == "EARLIER":
                findings.append(
                    Finding(
                        "DAX007",
                        Severity.INFO,
                        measure.qualified_name,
                        "uses EARLIER() to reach an outer row context",
                        "Capture the outer value in a VAR before entering the "
                        "inner context. EARLIER() depends on nesting depth, "
                        "which makes the expression fragile to edit.",
                        token.line,
                    )
                )
                break
    return findings


def _dax008_repeated_subexpression(model: TabularModel) -> List[Finding]:
    findings: List[Finding] = []
    for measure in _iter_measures(model):
        tokens = tokenize(measure.expression)
        if any(t.upper == "VAR" for t in tokens):
            continue
        signatures: Counter = Counter()
        for index, token in enumerate(tokens):
            if token.kind is not TokenKind.IDENTIFIER or index + 1 >= len(tokens):
                continue
            if tokens[index + 1].text != "(":
                continue
            # Signature: the call and its argument list, stopping before the
            # operator that follows - otherwise the same call reads as three
            # different expressions depending on what comes after it.
            window = tokens[index : index + 4]
            signatures[" ".join(t.text.upper() for t in window)] += 1
        repeated = [sig for sig, count in signatures.items() if count >= 3]
        if not repeated:
            continue
        findings.append(
            Finding(
                "DAX008",
                Severity.INFO,
                measure.qualified_name,
                "repeats the same sub-expression %d times without a VAR"
                % max(signatures.values()),
                "Assign the repeated expression to a VAR. It is evaluated once "
                "instead of once per occurrence, and the measure becomes "
                "readable.",
            )
        )
    return findings


ALL_RULES: Sequence[Rule] = (
    Rule("MOD001", Severity.INFO, "measure without a description", _mod001_measure_without_description),
    Rule("MOD002", Severity.INFO, "measure without a display folder", _mod002_measure_without_folder),
    Rule("MOD003", Severity.WARNING, "bi-directional relationship", _mod003_bidirectional_relationship),
    Rule("MOD004", Severity.WARNING, "inactive relationship never activated", _mod004_inactive_relationship_unused),
    Rule("MOD005", Severity.WARNING, "auto date/time enabled", _mod005_auto_date_tables),
    Rule("MOD006", Severity.WARNING, "monetary column typed as double", _mod006_currency_as_double),
    Rule("MOD007", Severity.ERROR, "identifier column is aggregated by default", _mod007_key_column_summarised),
    Rule("MOD008", Severity.INFO, "foreign key column left visible", _mod008_visible_foreign_key),
    Rule("MOD009", Severity.ERROR, "time intelligence without a marked date table", _mod009_no_marked_date_table),
    Rule("MOD010", Severity.INFO, "measure without a format string", _mod010_measure_without_format),
    Rule("MOD011", Severity.WARNING, "duplicated measure logic", _mod011_duplicate_measure_logic),
    Rule("MOD012", Severity.WARNING, "calculated column doing aggregation", _mod012_calculated_column_aggregation),
    Rule("DAX001", Severity.WARNING, "'/' instead of DIVIDE()", _dax001_division_operator),
    Rule("DAX002", Severity.WARNING, "unqualified column reference", _dax002_unqualified_column),
    Rule("DAX003", Severity.INFO, "qualified measure reference", _dax003_qualified_measure),
    Rule("DAX004", Severity.WARNING, "FILTER() over a whole table", _dax004_filter_whole_table),
    Rule("DAX005", Severity.WARNING, "IFERROR/ISERROR error suppression", _dax005_error_suppression),
    Rule("DAX006", Severity.INFO, "iterator where an aggregator would do", _dax006_iterator_over_column),
    Rule("DAX007", Severity.INFO, "EARLIER() instead of a VAR", _dax007_earlier),
    Rule("DAX008", Severity.INFO, "repeated sub-expression without a VAR", _dax008_repeated_subexpression),
)


def lint(
    model: TabularModel,
    select: Optional[Iterable[str]] = None,
    ignore: Optional[Iterable[str]] = None,
) -> List[Finding]:
    """Run the rule set over a model.

    ``select`` restricts the run to the given codes; ``ignore`` removes codes
    from it.  Findings come back sorted by severity, then code, then object.
    """
    selected = {c.upper() for c in select} if select else None
    ignored = {c.upper() for c in ignore} if ignore else set()

    findings: List[Finding] = []
    for rule in ALL_RULES:
        if selected is not None and rule.code not in selected:
            continue
        if rule.code in ignored:
            continue
        findings.extend(rule.check(model))

    order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    return sorted(findings, key=lambda f: (order[f.severity], f.code, f.object_name, f.line))
