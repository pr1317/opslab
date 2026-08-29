"""End-to-end exercise of the command line interface."""

import os

import pytest

from opslab.cli import main

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "pensions_ops.bim"
)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """A small simulated extract shared by the reading subcommands."""
    out = tmp_path_factory.mktemp("data")
    assert main(["simulate", "--cases", "300", "--seed", "42", "--out", str(out)]) == 0
    return out


def test_simulate_writes_both_tables(generated):
    assert (generated / "events.csv").exists()
    assert (generated / "cases.csv").exists()


def test_mine_writes_a_process_map(generated, tmp_path):
    code = main(["mine", "--events", str(generated / "events.csv"), "--out", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "process_map.dot").read_text().startswith("digraph process {")
    assert (tmp_path / "process_map.mmd").read_text().startswith("flowchart TD")
    assert (tmp_path / "variants.csv").exists()
    assert (tmp_path / "conformance.txt").exists()


def test_spc_writes_chart_data(generated, tmp_path):
    code = main(["spc", "--cases", str(generated / "cases.csv"), "--out", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "spc_breach_rate.csv").exists()
    assert (tmp_path / "spc_handling_hours.csv").exists()


def test_sla_fits_a_model_and_scores_risk(generated, tmp_path, capsys):
    code = main([
        "sla", "--cases", str(generated / "cases.csv"), "--out", str(tmp_path),
        "--numeric", "complexity", "--binary", "priority=Urgent", "--group", "priority",
    ])
    assert code == 0
    output = capsys.readouterr().out
    assert "Kaplan-Meier" in output
    assert "Log-rank test" in output
    assert "Cox proportional hazards" in output
    assert (tmp_path / "cox_coefficients.csv").exists()
    assert (tmp_path / "breach_risk.csv").exists()
    assert (tmp_path / "km_curve.csv").exists()


def test_sla_skips_the_model_without_covariates(generated, capsys):
    assert main(["sla", "--cases", str(generated / "cases.csv")]) == 0
    assert "skipping the Cox model" in capsys.readouterr().out


def test_sla_rejects_a_malformed_binary_covariate(generated):
    with pytest.raises(SystemExit):
        main(["sla", "--cases", str(generated / "cases.csv"), "--binary", "priority"])


def test_daxlint_exits_non_zero_on_an_error(capsys):
    assert main(["daxlint", EXAMPLE]) == 1
    assert "MOD007" in capsys.readouterr().out


def test_daxlint_can_be_told_not_to_fail(capsys):
    assert main(["daxlint", EXAMPLE, "--fail-on", "never"]) == 0


def test_daxlint_emits_json(capsys):
    import json

    assert main(["daxlint", EXAMPLE, "--format", "json", "--fail-on", "never"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "PensionsOperations"
    assert payload["findings"]


def test_daxlint_select_narrows_the_run(capsys):
    main(["daxlint", EXAMPLE, "--select", "DAX001", "--fail-on", "never"])
    output = capsys.readouterr().out
    assert "DAX001" in output and "MOD007" not in output


def test_daxlint_lists_its_rules(capsys):
    assert main(["daxlint", "--list-rules"]) == 0
    assert "MOD001" in capsys.readouterr().out


def test_daxlint_requires_a_model_path():
    with pytest.raises(SystemExit):
        main(["daxlint"])


def test_a_missing_file_is_reported_not_raised(tmp_path, capsys):
    assert main(["mine", "--events", str(tmp_path / "nope.csv")]) == 2
    assert "error:" in capsys.readouterr().err


def test_demo_runs_every_module(tmp_path, capsys):
    assert main(["demo", "--cases", "250", "--out", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    for section in ("Simulated operation", "Process mining",
                    "Statistical process control", "SLA survival analysis",
                    "Power BI model lint"):
        assert section in output
    assert (tmp_path / "lint_findings.csv").exists()
