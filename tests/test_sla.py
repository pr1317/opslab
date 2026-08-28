"""Survival estimation and Cox regression."""

import math

import pytest

from opslab.eventlog import Case, CaseTable
from opslab.sla import (
    concordance_index,
    design_matrix,
    fit_cox,
    kaplan_meier,
    logrank_test,
    nelson_aalen,
)
from opslab.sla.coxph import _partial_likelihood


# -- Kaplan-Meier ----------------------------------------------------------
def test_kaplan_meier_matches_a_hand_computed_curve():
    # Times 1..5; the observations at 2 and 5 are censored.
    #   t=1: 1 - 1/5            = 0.8
    #   t=3: 0.8 * (1 - 1/3)    = 0.5333...
    #   t=4: 0.5333 * (1 - 1/2) = 0.2666...
    curve = kaplan_meier([1, 2, 3, 4, 5], [1, 0, 1, 1, 0])
    assert curve.survival == pytest.approx([0.8, 0.8, 0.8 * 2 / 3, 0.8 * 2 / 3 / 2, 0.8 * 2 / 3 / 2])
    assert curve.at_risk == [5, 4, 3, 2, 1]


def test_kaplan_meier_without_censoring_is_the_empirical_curve():
    curve = kaplan_meier([1, 2, 3, 4], [1, 1, 1, 1])
    assert curve.survival == pytest.approx([0.75, 0.5, 0.25, 0.0])


def test_survival_at_is_a_right_continuous_step():
    curve = kaplan_meier([1, 2, 3, 4], [1, 1, 1, 1])
    assert curve.survival_at(0.5) == pytest.approx(1.0)
    assert curve.survival_at(1.0) == pytest.approx(0.75)
    assert curve.survival_at(1.9) == pytest.approx(0.75)


def test_median_is_none_when_follow_up_never_reaches_it():
    # Only one event in ten observations: the curve never falls to 0.5.
    curve = kaplan_meier(list(range(1, 11)), [1] + [0] * 9)
    assert curve.median is None


def test_confidence_band_stays_inside_zero_and_one():
    curve = kaplan_meier([1, 2, 3, 4, 5, 6], [1, 1, 0, 1, 0, 1])
    assert all(0.0 <= lo <= hi <= 1.0 for lo, hi in zip(curve.lower, curve.upper))


def test_restricted_mean_of_a_flat_curve_is_the_horizon():
    # No events, so survival stays at 1 and the area to t is exactly t.
    curve = kaplan_meier([5, 6, 7], [0, 0, 0])
    assert curve.restricted_mean(4.0) == pytest.approx(4.0)


def test_ignoring_censoring_understates_time_to_resolution():
    durations = [10, 20, 30, 40, 50, 60]
    events = [1, 1, 1, 0, 0, 0]
    closed_only = sorted(d for d, e in zip(durations, events) if e)
    naive_median = closed_only[len(closed_only) // 2]
    curve = kaplan_meier(durations, events)
    # The Kaplan-Meier curve has not dropped to 0.5 by the naive median.
    assert curve.survival_at(naive_median) > 0.5


def test_nelson_aalen_accumulates_the_hazard():
    estimate = nelson_aalen([1, 2, 3], [1, 1, 1])
    assert estimate.cumulative_hazard == pytest.approx([1 / 3, 1 / 3 + 1 / 2, 1 / 3 + 1 / 2 + 1.0])


@pytest.mark.parametrize(
    "durations, events",
    [([1, 2], [1]), ([], []), ([-1.0], [1]), ([1.0], [2])],
)
def test_survival_estimators_reject_invalid_input(durations, events):
    with pytest.raises(ValueError):
        kaplan_meier(durations, events)


# -- log-rank --------------------------------------------------------------
def test_logrank_finds_no_difference_between_identical_groups():
    durations = [1, 2, 3, 4, 5, 6]
    events = [1] * 6
    result = logrank_test(durations, events, durations, events)
    assert result.statistic == pytest.approx(0.0, abs=1e-9)
    assert result.p_value == pytest.approx(1.0, abs=1e-9)


def test_logrank_detects_a_clear_separation():
    fast = list(range(1, 16))
    slow = list(range(40, 55))
    result = logrank_test(fast, [1] * 15, slow, [1] * 15)
    assert result.p_value < 1e-6
    # The fast group resolves more often than expected under the null.
    assert result.observed_a > result.expected_a


def test_logrank_expected_counts_sum_to_the_observed_total():
    result = logrank_test([1, 3, 5], [1, 1, 0], [2, 4, 6], [1, 0, 1])
    assert result.expected_a + result.expected_b == pytest.approx(
        result.observed_a + result.observed_b
    )


# -- Cox -------------------------------------------------------------------
def _toy_data():
    durations = [6, 7, 10, 15, 19, 25, 3, 4, 8, 12]
    events = [1, 1, 1, 0, 1, 0, 1, 1, 1, 1]
    covariates = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]]
    return durations, events, covariates


def test_cox_gradient_vanishes_at_the_fitted_coefficients():
    durations, events, covariates = _toy_data()
    model = fit_cox(durations, events, covariates, names=["treatment"])
    order = sorted(range(len(durations)), key=lambda i: -durations[i])
    _, gradient, _ = _partial_likelihood(
        model.result.coefficients, order, durations, events, covariates, 1
    )
    assert gradient[0] == pytest.approx(0.0, abs=1e-6)


def test_cox_analytic_gradient_matches_finite_differences():
    """The derivatives are hand-derived, so check them numerically."""
    durations, events, covariates = _toy_data()
    order = sorted(range(len(durations)), key=lambda i: -durations[i])
    beta = [0.37]
    step = 1e-6
    high, _, _ = _partial_likelihood([beta[0] + step], order, durations, events, covariates, 1)
    low, _, _ = _partial_likelihood([beta[0] - step], order, durations, events, covariates, 1)
    _, gradient, _ = _partial_likelihood(beta, order, durations, events, covariates, 1)
    assert gradient[0] == pytest.approx((high - low) / (2 * step), abs=1e-5)


def test_cox_analytic_hessian_matches_finite_differences():
    durations, events, covariates = _toy_data()
    order = sorted(range(len(durations)), key=lambda i: -durations[i])
    beta = [0.37]
    step = 1e-5
    _, grad_high, _ = _partial_likelihood([beta[0] + step], order, durations, events, covariates, 1)
    _, grad_low, _ = _partial_likelihood([beta[0] - step], order, durations, events, covariates, 1)
    _, _, hessian = _partial_likelihood(beta, order, durations, events, covariates, 1)
    assert hessian[0][0] == pytest.approx((grad_high[0] - grad_low[0]) / (2 * step), rel=1e-4)


def test_cox_log_likelihood_improves_on_the_null():
    durations, events, covariates = _toy_data()
    model = fit_cox(durations, events, covariates, names=["treatment"])
    assert model.result.log_likelihood >= model.result.null_log_likelihood
    assert model.result.likelihood_ratio_statistic >= 0.0
    assert model.result.converged


def test_cox_sign_follows_the_faster_group():
    # The second group resolves systematically sooner, so its hazard is higher.
    durations = [20, 22, 24, 26, 2, 3, 4, 5]
    events = [1] * 8
    covariates = [[0.0]] * 4 + [[1.0]] * 4
    model = fit_cox(durations, events, covariates, names=["fast"])
    assert model.result.coefficient("fast") > 0
    assert model.result.hazard_ratios[0] > 1.0


def test_cox_recovers_known_simulation_coefficients():
    from opslab.simulate import GROUND_TRUTH_BETA, SimulationConfig, simulate

    result = simulate(SimulationConfig(n_cases=2500, seed=7))
    matrix = design_matrix(
        result.cases,
        numeric=["complexity", "backlog_index", "awaiting_third_party"],
        binary={"priority": "Urgent", "channel": "Post"},
    )
    model = fit_cox(
        [c.duration_hours for c in result.cases],
        [1 if c.resolved else 0 for c in result.cases],
        matrix.rows,
        names=matrix.names,
    )
    assert model.result.converged
    for name, estimate, se in zip(
        model.result.names, model.result.coefficients, model.result.standard_errors
    ):
        truth = GROUND_TRUTH_BETA[name]
        assert abs(estimate - truth) < 3.0 * se, "%s: %.3f vs %.3f" % (name, estimate, truth)


def test_baseline_survival_decreases_with_time():
    durations, events, covariates = _toy_data()
    model = fit_cox(durations, events, covariates, names=["treatment"])
    values = [model.survival([0.0], t) for t in (1, 5, 10, 20)]
    assert all(later <= earlier for earlier, later in zip(values, values[1:]))
    assert all(0.0 <= v <= 1.0 for v in values)


def test_breach_probability_rises_with_a_slower_case():
    durations, events, covariates = _toy_data()
    model = fit_cox(durations, events, covariates, names=["treatment"])
    fast = model.breach_probability([1.0], 10.0)
    slow = model.breach_probability([0.0], 10.0)
    assert slow > fast


def test_cox_rejects_data_without_events():
    with pytest.raises(ValueError, match="no events"):
        fit_cox([1, 2, 3], [0, 0, 0], [[1.0], [0.0], [1.0]])


def test_cox_reports_collinear_covariates_clearly():
    durations = [1, 2, 3, 4]
    events = [1, 1, 1, 1]
    # The second column is an exact copy of the first.
    covariates = [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
    with pytest.raises(ValueError, match="collinear"):
        fit_cox(durations, events, covariates)


# -- concordance -----------------------------------------------------------
def test_concordance_is_one_for_a_perfect_ranking():
    durations = [1, 2, 3, 4]
    events = [1, 1, 1, 1]
    # Higher risk means faster resolution.
    index, pairs = concordance_index(durations, events, [4.0, 3.0, 2.0, 1.0])
    assert index == pytest.approx(1.0)
    assert pairs == 6


def test_concordance_is_zero_for_a_reversed_ranking():
    index, _ = concordance_index([1, 2, 3, 4], [1] * 4, [1.0, 2.0, 3.0, 4.0])
    assert index == pytest.approx(0.0)


def test_tied_risk_scores_count_as_half():
    index, _ = concordance_index([1, 2], [1, 1], [5.0, 5.0])
    assert index == pytest.approx(0.5)


def test_concordance_skips_pairs_made_incomparable_by_censoring():
    # The first observation is censored, so no pair is comparable.
    index, pairs = concordance_index([1, 2], [0, 0], [2.0, 1.0])
    assert pairs == 0
    assert index == pytest.approx(0.5)


# -- design matrix ---------------------------------------------------------
def _case(case_id, priority, channel, complexity):
    return Case(
        case_id=case_id,
        arrived=__import__("datetime").datetime(2026, 1, 1, 9, 0),
        duration_hours=10.0,
        resolved=True,
        sla_hours=40.0,
        attributes={"priority": priority, "channel": channel, "complexity": complexity},
    )


def test_design_matrix_encodes_binary_and_numeric_columns():
    cases = CaseTable([
        _case("A", "Urgent", "Post", "0.5"),
        _case("B", "Standard", "Portal", "0.25"),
    ])
    matrix = design_matrix(cases, numeric=["complexity"], binary={"priority": "Urgent"})
    assert matrix.names == ["complexity", "priority_urgent"]
    assert matrix.rows == [[0.5, 1.0], [0.25, 0.0]]


def test_design_matrix_holds_out_a_reference_level():
    cases = CaseTable([
        _case("A", "Urgent", "Post", "0.5"),
        _case("B", "Standard", "Portal", "0.25"),
        _case("C", "Standard", "Phone", "0.75"),
    ])
    matrix = design_matrix(cases, categorical=["channel"])
    # "Phone" sorts first and becomes the reference, so it gets no column.
    assert matrix.references["channel"] == "Phone"
    assert matrix.names == ["channel_portal", "channel_post"]


def test_design_matrix_rejects_a_non_numeric_column():
    cases = CaseTable([_case("A", "Urgent", "Post", "not-a-number")])
    with pytest.raises(ValueError, match="not numeric"):
        design_matrix(cases, numeric=["complexity"])


def test_design_matrix_rejects_a_constant_categorical():
    cases = CaseTable([_case("A", "Urgent", "Post", "0.5")])
    with pytest.raises(ValueError, match="fewer than two levels"):
        design_matrix(cases, categorical=["channel"])
