"""The browser demo must compute the same numbers as the Python it exports.

`web/opslab.js` re-implements the survival arithmetic so a reader can move a
slider without a server. A re-implementation is a second source of truth, and a
second source of truth drifts, so this pins the two together: the same covariate
vectors are scored by the fitted Python model and by the JavaScript running over
the exported payload, and the answers must agree to 1e-9.
"""

import json
import os
import shutil
import subprocess

import pytest

from opslab import data
from opslab.eventlog import CaseTable
from opslab.exporter import build_payload
from opslab.sla import design_matrix, fit_cox

REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPOSITORY, "web", "opslab.js")

#: Corners of the covariate space plus a typical case, paired with real SLA
#: targets from the sample. Order matches the fitted coefficient names.
PROBES = [
    ([0.42, 0.50, 0.0, 0.0, 0.0], 64.0),
    ([0.90, 0.95, 1.0, 1.0, 1.0], 128.0),
    ([0.05, 0.20, 0.0, 1.0, 0.0], 48.0),
    ([0.50, 0.50, 1.0, 0.0, 1.0], 96.0),
    ([0.01, 0.19, 0.0, 0.0, 0.0], 39.0),
    ([0.95, 1.00, 1.0, 0.0, 1.0], 52.0),
]

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


@pytest.fixture(scope="module")
def fitted():
    cases = CaseTable.from_csv(data.cases_path())
    durations = [case.duration_hours for case in cases]
    events = [1 if case.resolved else 0 for case in cases]
    matrix = design_matrix(
        cases,
        numeric=["complexity", "backlog_index", "awaiting_third_party"],
        binary={"priority": "Urgent", "channel": "Post"},
    )
    return fit_cox(durations, events, matrix.rows, names=matrix.names)


@pytest.fixture(scope="module")
def payload():
    return build_payload(data.events_path(), data.cases_path(), data.model_path())


@pytest.fixture(scope="module")
def javascript_results(payload, tmp_path_factory):
    """Score every probe in node, over the exported payload."""
    directory = tmp_path_factory.mktemp("web")
    data_file = directory / "payload.json"
    data_file.write_text(json.dumps(payload), encoding="utf-8")

    runner = directory / "run.js"
    runner.write_text(
        "const opslab = require(%s);\n"
        "const cox = require(%s).sla.cox;\n"
        "const probes = %s;\n"
        "console.log(JSON.stringify(probes.map(function (probe) {\n"
        "  return {\n"
        "    breach: opslab.breachProbability(cox, probe[0], probe[1]),\n"
        "    lp: opslab.linearPredictor(cox.coefficients, probe[0]),\n"
        "    hazard: opslab.cumulativeHazardAt(cox.baseline, probe[1])\n"
        "  };\n"
        "})));\n"
        % (json.dumps(SCRIPT), json.dumps(str(data_file)), json.dumps(PROBES)),
        encoding="utf-8",
    )
    output = subprocess.run(
        ["node", str(runner)], capture_output=True, text=True, check=True
    )
    return json.loads(output.stdout)


def test_the_javascript_reproduces_the_python_breach_probability(
    fitted, javascript_results
):
    """The browser scores against the exported payload; Python against its own fit."""
    for (covariates, sla_hours), actual in zip(PROBES, javascript_results):
        expected = fitted.breach_probability(covariates, sla_hours)
        assert abs(expected - actual["breach"]) < 1e-9, (covariates, sla_hours)


def test_the_javascript_reproduces_the_linear_predictor(fitted, javascript_results):
    for (covariates, _), actual in zip(PROBES, javascript_results):
        expected = sum(b * x for b, x in zip(fitted.result.coefficients, covariates))
        assert abs(expected - actual["lp"]) < 1e-9


def test_the_javascript_reads_the_same_step_of_the_baseline_hazard(
    fitted, javascript_results
):
    """An off-by-one in the step lookup would be invisible in the probability alone."""
    for (_, sla_hours), actual in zip(PROBES, javascript_results):
        expected = fitted.cumulative_hazard_at(sla_hours)
        assert abs(expected - actual["hazard"]) < 1e-9


def test_the_baseline_hazard_is_exported_unthinned(payload):
    """Dropping steps from a cumulative hazard biases every probability down."""
    times = payload["sla"]["cox"]["baseline"]["times"]
    hazard = payload["sla"]["cox"]["baseline"]["cumulativeHazard"]
    assert len(times) == len(hazard) > 1000
    assert times == sorted(times)
    assert hazard == sorted(hazard)


def test_the_payload_carries_a_laid_out_map_at_every_offered_threshold(payload):
    for threshold in payload["mining"]["thresholds"]:
        svg = payload["mining"]["maps"][str(threshold)]
        assert svg.startswith("<svg")


def test_the_exported_payload_is_json_serialisable(payload):
    """The demo loads it through a <script> tag, so it must round-trip exactly."""
    assert json.loads(json.dumps(payload)) == payload
