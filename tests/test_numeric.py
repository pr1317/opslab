"""Checks against externally known values, not just internal consistency."""

import math

import pytest

from opslab.numeric import (
    chi2_sf,
    inverse,
    mean,
    norm_cdf,
    norm_ppf,
    norm_sf,
    quantile,
    solve,
    stdev,
)


@pytest.mark.parametrize(
    "probability, expected",
    [(0.95, 1.6448536270), (0.975, 1.9599639845), (0.995, 2.5758293035), (0.5, 0.0)],
)
def test_norm_ppf_matches_published_quantiles(probability, expected):
    assert norm_ppf(probability) == pytest.approx(expected, abs=1e-9)


def test_norm_ppf_inverts_norm_cdf():
    for x in (-3.5, -1.0, 0.0, 0.25, 2.75):
        assert norm_ppf(norm_cdf(x)) == pytest.approx(x, abs=1e-9)


def test_norm_sf_is_accurate_in_the_far_tail():
    # 1 - cdf loses all precision here; erfc does not.
    assert norm_sf(8.0) == pytest.approx(6.220960574e-16, rel=1e-6)


@pytest.mark.parametrize(
    "statistic, df", [(3.8414588, 1), (5.9914645, 2), (7.8147279, 3), (9.4877290, 4)]
)
def test_chi2_sf_matches_the_five_percent_critical_values(statistic, df):
    assert chi2_sf(statistic, df) == pytest.approx(0.05, abs=1e-6)


def test_chi2_sf_is_bounded_and_monotonic():
    previous = 1.0
    for x in (0.0, 0.5, 1.0, 4.0, 10.0, 40.0):
        value = chi2_sf(x, 3)
        assert 0.0 <= value <= 1.0
        assert value <= previous + 1e-12
        previous = value


def test_solve_recovers_a_known_solution():
    assert solve([[2.0, 1.0], [1.0, 3.0]], [3.0, 5.0]) == pytest.approx([0.8, 1.4])


def test_inverse_matches_the_closed_form_two_by_two():
    result = inverse([[4.0, 7.0], [2.0, 6.0]])
    assert result[0] == pytest.approx([0.6, -0.7])
    assert result[1] == pytest.approx([-0.2, 0.4])


def test_solve_rejects_a_singular_matrix():
    with pytest.raises(ValueError, match="singular"):
        solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


def test_quantile_uses_linear_interpolation():
    assert quantile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    assert quantile([1, 2, 3, 4], 0.25) == pytest.approx(1.75)
    assert quantile([5], 0.9) == 5


def test_mean_and_stdev_agree_with_the_standard_library():
    import statistics

    values = [3.0, 1.5, 4.25, 9.0, 2.5]
    assert mean(values) == pytest.approx(statistics.mean(values))
    assert stdev(values) == pytest.approx(statistics.stdev(values))


def test_descriptive_helpers_reject_degenerate_input():
    with pytest.raises(ValueError):
        mean([])
    with pytest.raises(ValueError):
        stdev([1.0])
    with pytest.raises(ValueError):
        quantile([1.0], 1.5)
