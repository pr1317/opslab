"""Control charts, run rules, capability and measurement system analysis."""

import math

import pytest

from opslab.spc import (
    apply_nelson_rules,
    c_chart,
    capability,
    gage_rr,
    individuals_chart,
    p_chart,
    u_chart,
    xbar_r_chart,
)
from opslab.spc.capability import sigma_level_from_dpmo
from opslab.spc.charts import D2, moving_range_chart
from opslab.spc.rules import NELSON_RULES, expand_windows


# -- run rules -------------------------------------------------------------
def _codes(standardised):
    return apply_nelson_rules(standardised)


def test_rule1_flags_a_point_beyond_three_sigma():
    flags = _codes([0.0, 0.5, 3.5, -0.2])
    assert flags[2] == ["NELSON_1"]
    assert flags[0] == []


def test_rule2_needs_nine_on_the_same_side():
    assert "NELSON_2" not in _codes([0.5] * 8)[-1]
    assert "NELSON_2" in _codes([0.5] * 9)[-1]


def test_rule3_detects_a_monotonic_run_of_six():
    assert "NELSON_3" in _codes([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])[-1]
    assert "NELSON_3" not in _codes([0.1, 0.2, 0.3, 0.4, 0.5, 0.5])[-1]


def test_rule4_detects_fourteen_alternating_points():
    series = [0.5 if i % 2 else -0.5 for i in range(14)]
    assert "NELSON_4" in _codes(series)[-1]


def test_rule5_needs_two_of_three_beyond_two_sigma_on_one_side():
    assert "NELSON_5" in _codes([2.5, 0.1, 2.2])[-1]
    # Opposite sides do not count as a run.
    assert "NELSON_5" not in _codes([-2.5, 0.1, 2.2])[-1]


def test_rule6_needs_four_of_five_beyond_one_sigma_on_one_side():
    assert "NELSON_6" in _codes([1.5, 1.2, 0.1, 1.3, 1.4])[-1]
    assert "NELSON_6" not in _codes([1.5, 0.2, 0.1, 1.3, 1.4])[-1]


def test_rule7_detects_fifteen_points_hugging_the_centre():
    assert "NELSON_7" in _codes([0.2] * 15)[-1]


def test_rule8_needs_both_sides_beyond_one_sigma():
    assert "NELSON_8" in _codes([1.5, -1.5] * 4)[-1]
    # All on one side is rule 2 territory, not rule 8.
    assert "NELSON_8" not in _codes([1.5] * 8)[-1]


def test_expand_windows_widens_a_run_back_over_its_window():
    flags = _codes([0.5] * 9)
    widened = expand_windows(flags)
    assert all("NELSON_2" in codes for codes in widened)


def test_rules_are_skipped_when_the_series_is_shorter_than_the_window():
    assert _codes([0.1, 0.2]) == [[], []]


# -- charts ----------------------------------------------------------------
def test_individuals_chart_limits_use_the_average_moving_range():
    values = [10.0, 12.0, 10.0, 12.0, 10.0, 12.0]
    chart = individuals_chart(values)
    # Every moving range is 2.0, so sigma is 2.0 / d2(2).
    expected_sigma = 2.0 / D2[2]
    assert chart.sigma == pytest.approx(expected_sigma)
    assert chart.center == pytest.approx(11.0)
    assert chart.points[0].upper == pytest.approx(11.0 + 3 * expected_sigma)


def test_individuals_chart_flags_a_clear_outlier():
    chart = individuals_chart([10, 11, 9, 10, 12, 10, 9, 11, 10, 10, 60])
    assert chart.points[-1].violations
    assert not chart.in_control()


def test_moving_range_chart_uses_the_d4_factor():
    chart = moving_range_chart([10.0, 12.0, 10.0, 12.0])
    assert chart.center == pytest.approx(2.0)
    assert chart.points[0].upper == pytest.approx(3.267 * 2.0)
    assert chart.points[0].lower == 0.0


def test_p_chart_limits_widen_on_small_subgroups():
    chart = p_chart([5, 5], [500, 50])
    wide, narrow = chart.points[1], chart.points[0]
    assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)


def test_p_chart_centre_is_the_pooled_proportion():
    chart = p_chart([2, 8], [50, 50])
    assert chart.center == pytest.approx(10 / 100)


def test_p_chart_limits_stay_inside_zero_and_one():
    chart = p_chart([0, 1, 0], [5, 5, 5])
    assert all(0.0 <= p.lower and p.upper <= 1.0 for p in chart.points)


def test_p_chart_warns_when_the_normal_approximation_is_weak():
    chart = p_chart([0, 1, 0], [5, 5, 5])
    assert any("normal approximation" in note for note in chart.notes)


def test_xbar_r_chart_uses_the_a2_factor():
    subgroups = [[10, 12], [11, 13], [9, 11]]
    chart = xbar_r_chart(subgroups)
    # Every range is 2.0 and A2 for n=2 is 1.880.
    assert chart.points[0].upper - chart.center == pytest.approx(1.880 * 2.0)


def test_c_chart_uses_the_poisson_standard_deviation():
    chart = c_chart([4, 4, 4, 4])
    assert chart.sigma == pytest.approx(2.0)


def test_u_chart_scales_limits_by_exposure():
    chart = u_chart([10, 10], [100.0, 10.0])
    assert chart.points[1].sigma > chart.points[0].sigma


@pytest.mark.parametrize(
    "call, kwargs",
    [
        (individuals_chart, {"values": [1.0]}),
        (p_chart, {"defectives": [1], "sizes": [0]}),
        (p_chart, {"defectives": [5], "sizes": [2]}),
        (u_chart, {"counts": [1], "exposures": [0.0]}),
        (c_chart, {"counts": [-1]}),
    ],
)
def test_charts_reject_invalid_input(call, kwargs):
    with pytest.raises(ValueError):
        call(**kwargs)


def test_xbar_r_chart_requires_equal_subgroups():
    with pytest.raises(ValueError, match="equally sized"):
        xbar_r_chart([[1, 2], [1, 2, 3]])


# -- capability ------------------------------------------------------------
def test_pp_is_the_spread_over_six_overall_sigma():
    import statistics

    values = [8.0, 12.0] * 8
    report = capability(values, lsl=0.0, usl=20.0)
    assert report.sigma_overall == pytest.approx(statistics.stdev(values))
    assert report.pp == pytest.approx(20.0 / (6.0 * report.sigma_overall))


def test_cpk_penalises_an_off_centre_process():
    centred = capability([9.0, 11.0] * 8, lsl=0.0, usl=20.0)
    shifted = capability([17.0, 19.0] * 8, lsl=0.0, usl=20.0)
    assert centred.cpk > shifted.cpk
    assert centred.cp == pytest.approx(shifted.cp)


def test_one_sided_specification_leaves_cp_undefined():
    report = capability([10.0, 12.0] * 8, usl=20.0)
    assert report.cp is None and report.pp is None
    assert report.cpk is not None
    assert any("One-sided" in note for note in report.notes)


def test_capability_flags_a_drifting_process():
    drifting = [10.0, 10.1, 10.2, 20.0, 30.0, 40.0, 50.0, 60.0]
    report = capability(drifting, lsl=0.0, usl=100.0)
    assert any("shifting over time" in note for note in report.notes)


def test_sigma_level_uses_the_conventional_shift():
    # 3.4 DPMO is the textbook "six sigma" defect rate.
    assert sigma_level_from_dpmo(3.4) == pytest.approx(6.0, abs=0.01)


def test_capability_requires_a_specification_limit():
    with pytest.raises(ValueError, match="specification limit"):
        capability([1.0, 2.0, 3.0])


def test_capability_rejects_a_constant_sample():
    with pytest.raises(ValueError, match="constant"):
        capability([5.0] * 6, usl=10.0)


# -- Gage R&R --------------------------------------------------------------
def _balanced_study(noise):
    """Three operators measure five parts twice, with controllable noise."""
    measurements = {}
    for part_index, part_value in enumerate([10.0, 12.0, 14.0, 16.0, 18.0]):
        for operator_index, operator in enumerate(["A", "B", "C"]):
            bias = operator_index * 0.1
            measurements[("P%d" % part_index, operator)] = [
                part_value + bias - noise,
                part_value + bias + noise,
            ]
    return measurements


def test_gage_rr_variance_components_sum_to_the_total():
    report = gage_rr(_balanced_study(0.2))
    total = (
        report.var_repeatability
        + report.var_reproducibility
        + report.var_interaction
        + report.var_part
    )
    assert total == pytest.approx(report.var_total)


def test_gage_rr_finds_a_capable_system_when_parts_dominate():
    report = gage_rr(_balanced_study(0.05))
    assert report.study_variation()["part_to_part"] > 95.0
    assert report.verdict == "acceptable"
    assert report.ndc >= 5


def test_gage_rr_condemns_a_noisy_system():
    report = gage_rr(_balanced_study(8.0))
    assert report.verdict == "unacceptable"


def test_gage_rr_repeatability_is_the_within_cell_variance():
    # Each cell holds v-noise and v+noise, whose sample variance (ddof=1) is
    # 2 * noise^2. That is exactly the mean square error the ANOVA reports.
    report = gage_rr(_balanced_study(0.5))
    assert report.var_repeatability == pytest.approx(0.5)


def test_gage_rr_rejects_an_unbalanced_design():
    study = _balanced_study(0.2)
    study[("P0", "A")] = [10.0]
    with pytest.raises(ValueError, match="unbalanced"):
        gage_rr(study)


def test_gage_rr_requires_replicates():
    study = {("P%d" % p, o): [float(p)] for p in range(3) for o in "AB"}
    with pytest.raises(ValueError, match="replicates"):
        gage_rr(study)
