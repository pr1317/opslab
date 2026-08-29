"""Lexer, model readers and lint rules."""

import json
import os
import textwrap

import pytest

from opslab import data
from opslab.daxlint import ALL_RULES, Severity, lint, load_model, tokenize
from opslab.daxlint.lexer import TokenKind, column_references, function_calls
from opslab.daxlint.model import Measure, Table, TabularModel, load_tmdl, load_tmsl

EXAMPLE = data.model_path()


# -- lexer -----------------------------------------------------------------
def test_comments_and_strings_do_not_produce_tokens():
    tokens = tokenize('SUM( Sales[Amount] ) -- FILTER lives in this comment\n')
    assert all(t.upper != "FILTER" for t in tokens)


def test_block_comments_are_skipped():
    tokens = tokenize("/* FILTER( x ) */ SUM( a[b] )")
    assert [t.upper for t in function_calls(tokens)] == ["SUM"]


def test_a_function_name_inside_a_string_is_not_a_call():
    tokens = tokenize('IF( x = "FILTER(", 1, 0 )')
    assert [t.upper for t in function_calls(tokens)] == ["IF"]


def test_bracket_and_quoted_tokens_strip_their_delimiters():
    tokens = tokenize("'Fact Sales'[Net Amount]")
    assert tokens[0].kind is TokenKind.QUOTED and tokens[0].value == "Fact Sales"
    assert tokens[1].kind is TokenKind.BRACKET and tokens[1].value == "Net Amount"


def test_qualified_and_unqualified_references_are_distinguished():
    references = column_references(tokenize("Sales[Amount] + [Total]"))
    assert references[0].qualified and references[0].table == "Sales"
    assert not references[1].qualified


def test_a_function_argument_is_not_read_as_a_qualifier():
    # SUM([x]) must not treat SUM as the table qualifying [x].
    references = column_references(tokenize("SUM( [Amount] )"))
    assert not references[0].qualified


def test_line_numbers_are_tracked_across_newlines():
    tokens = tokenize("SUM( a[b] )\nDIVIDE( 1, 2 )")
    assert [t.line for t in function_calls(tokens)] == [1, 2]


def test_an_unknown_character_does_not_abort_the_lex():
    tokens = tokenize("SUM( a[b] ) § 1")
    assert [t.upper for t in function_calls(tokens)] == ["SUM"]


# -- TMSL ------------------------------------------------------------------
def test_tmsl_reader_loads_the_sample_model():
    model = load_tmsl(EXAMPLE)
    assert model.name == "PensionsOperations"
    assert {t.name for t in model.tables} >= {"Cases", "Date", "Team"}
    assert len(model.measures()) == 12
    assert model.table("cases") is not None  # lookups are case-insensitive


def test_tmsl_reader_detects_auto_date_tables():
    model = load_tmsl(EXAMPLE)
    assert [t.name for t in model.auto_date_tables] == ["LocalDateTable_9f3c1e77"]


def test_tmsl_reader_reads_relationship_direction():
    model = load_tmsl(EXAMPLE)
    bidirectional = [r for r in model.relationships if r.is_bidirectional]
    assert len(bidirectional) == 1
    assert bidirectional[0].to_table == "Team"


def test_multi_line_expressions_are_joined():
    import tempfile

    document = {
        "name": "M",
        "model": {"tables": [{
            "name": "T",
            "measures": [{"name": "X", "expression": ["VAR a = 1", "RETURN a"]}],
        }]},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".bim", delete=False) as handle:
        json.dump(document, handle)
        path = handle.name
    try:
        model = load_tmsl(path)
        assert model.measures()[0].expression == "VAR a = 1\nRETURN a"
    finally:
        os.unlink(path)


# -- TMDL ------------------------------------------------------------------
def test_tmdl_reader_parses_tables_measures_and_columns(tmp_path):
    content = textwrap.dedent("""
        table Cases

            column CaseID
                dataType: int64
                summarizeBy: sum
                isKey

            measure 'Breach Rate' = DIVIDE( SUM( Cases[Breached] ), COUNTROWS( Cases ) )
                formatString: 0.0%
                displayFolder: Service
    """)
    path = tmp_path / "Cases.tmdl"
    path.write_text(content, encoding="utf-8")

    model = load_tmdl(str(path))
    assert [t.name for t in model.tables] == ["Cases"]
    column = model.tables[0].columns[0]
    assert column.name == "CaseID" and column.summarize_by == "sum" and column.is_key
    measure = model.tables[0].measures[0]
    assert measure.name == "Breach Rate"
    assert measure.display_folder == "Service"
    assert "DIVIDE" in measure.expression


def test_tmdl_reader_walks_a_definition_folder(tmp_path):
    folder = tmp_path / "definition" / "tables"
    folder.mkdir(parents=True)
    (folder / "A.tmdl").write_text("table A\n\n\tcolumn X\n\t\tdataType: string\n", encoding="utf-8")
    (folder / "B.tmdl").write_text("table B\n\n\tcolumn Y\n\t\tdataType: string\n", encoding="utf-8")
    model = load_tmdl(str(tmp_path))
    assert sorted(t.name for t in model.tables) == ["A", "B"]


def test_load_model_rejects_an_unknown_extension(tmp_path):
    path = tmp_path / "model.xyz"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported model file"):
        load_model(str(path))


# -- rules -----------------------------------------------------------------
@pytest.fixture(scope="module")
def findings():
    return lint(load_model(EXAMPLE))


def test_every_rule_fires_on_the_deliberately_flawed_sample(findings):
    """The example model exists to exercise the whole rule set."""
    fired = {f.code for f in findings}
    missing = sorted({rule.code for rule in ALL_RULES} - fired)
    assert missing == [], "rules never exercised: %s" % missing


@pytest.mark.parametrize(
    "code, expected_object",
    [
        ("MOD007", "Cases[CaseID]"),
        ("MOD006", "Cases[BenefitAmount]"),
        ("MOD012", "Cases[TeamTotalHours]"),
        ("DAX002", "Cases[Unqualified Total]"),
        ("DAX005", "Cases[Safe Rate]"),
        ("DAX007", "Cases[Rank Within Team]"),
    ],
)
def test_rules_point_at_the_right_object(findings, code, expected_object):
    objects = [f.object_name for f in findings if f.code == code]
    assert expected_object in objects


def test_key_columns_are_detected_in_camel_case(findings):
    flagged = {f.object_name for f in findings if f.code == "MOD007"}
    assert "Cases[TeamID]" in flagged


def test_duplicate_measure_logic_is_reported_once(findings):
    duplicates = [f for f in findings if f.code == "MOD011"]
    assert len(duplicates) == 1
    assert "Breach Rate (Legacy)" in duplicates[0].message


def test_findings_are_sorted_with_errors_first(findings):
    severities = [f.severity for f in findings]
    assert severities[0] is Severity.ERROR
    assert severities == sorted(
        severities, key=lambda s: {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}[s]
    )


def test_select_restricts_the_run():
    findings = lint(load_model(EXAMPLE), select=["DAX001"])
    assert {f.code for f in findings} == {"DAX001"}


def test_ignore_removes_a_rule():
    findings = lint(load_model(EXAMPLE), ignore=["MOD001", "MOD002", "MOD010"])
    assert not {"MOD001", "MOD002", "MOD010"} & {f.code for f in findings}


def test_a_clean_model_produces_no_findings():
    model = TabularModel(name="Clean", tables=[Table(
        name="Fact",
        measures=[Measure(
            name="Total",
            table="Fact",
            expression="DIVIDE( SUM( Fact[Amount] ), COUNTROWS( Fact ) )",
            description="Mean amount per row.",
            display_folder="Core",
            format_string="#,0.00",
        )],
    )])
    assert lint(model) == []


def test_dax002_does_not_fire_on_a_measure_reference():
    model = TabularModel(name="M", tables=[Table(
        name="F",
        measures=[
            Measure(name="Base", table="F", expression="COUNTROWS( F )",
                    description="d", display_folder="f", format_string="0"),
            Measure(name="Derived", table="F", expression="[Base] * 2",
                    description="d", display_folder="f", format_string="0"),
        ],
    )])
    assert not any(f.code == "DAX002" for f in lint(model))


def test_mod004_stays_quiet_when_userelationship_is_present():
    from opslab.daxlint.model import Relationship

    model = TabularModel(
        name="M",
        tables=[Table(name="F", measures=[Measure(
            name="M1", table="F",
            expression="CALCULATE( COUNTROWS( F ), USERELATIONSHIP( F[A], D[B] ) )",
            description="d", display_folder="f", format_string="0",
        )])],
        relationships=[Relationship(from_table="F", from_column="A", to_table="D",
                                    to_column="B", is_active=False)],
    )
    assert not any(f.code == "MOD004" for f in lint(model))


def test_findings_serialise_to_json(findings):
    payload = json.dumps([f.to_dict() for f in findings])
    restored = json.loads(payload)
    assert restored[0]["severity"] == "error"
    assert "remedy" in restored[0]
