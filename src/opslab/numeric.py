"""Small numeric kernel shared by the statistical modules.

Everything here is implemented against the standard library so the whole
toolkit installs with ``pip install .`` and no compiler, wheel or network
access.  The routines are the handful of primitives the rest of the package
needs: normal and chi-square distribution functions, and dense linear algebra
small enough for the covariate counts a Cox model on operational data uses.
"""

from __future__ import annotations

import math
from typing import List, Sequence

__all__ = [
    "norm_cdf",
    "norm_sf",
    "norm_ppf",
    "chi2_sf",
    "mean",
    "stdev",
    "quantile",
    "solve",
    "inverse",
]

Matrix = List[List[float]]


# --------------------------------------------------------------------------
# descriptive helpers
# --------------------------------------------------------------------------
def mean(values: Sequence[float]) -> float:
    """Arithmetic mean.  Raises on an empty sequence."""
    if not values:
        raise ValueError("mean() requires at least one value")
    return math.fsum(values) / len(values)


def stdev(values: Sequence[float], ddof: int = 1) -> float:
    """Standard deviation with ``ddof`` degrees of freedom removed."""
    n = len(values)
    if n - ddof <= 0:
        raise ValueError("stdev() requires more than ddof values")
    mu = mean(values)
    return math.sqrt(math.fsum((v - mu) ** 2 for v in values) / (n - ddof))


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (the "type 7" definition, as in R)."""
    if not values:
        raise ValueError("quantile() requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(math.floor(pos))
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


# --------------------------------------------------------------------------
# normal distribution
# --------------------------------------------------------------------------
def norm_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Standard normal CDF, built on :func:`math.erf`."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def norm_sf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal survival function, ``1 - cdf`` computed via ``erfc`` for tails."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return 0.5 * math.erfc((x - mu) / (sigma * math.sqrt(2.0)))


# Acklam's rational approximation to the inverse normal CDF.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425


def norm_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Inverse normal CDF (quantile function).

    Acklam's approximation refined by one Halley step, which takes the result
    to full double precision across the whole open interval ``(0, 1)``.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")

    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = ((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) /
             ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0))
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = ((((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q /
             (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0))
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) /
              ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0))

    # One Halley refinement against the exact CDF.
    err = norm_cdf(x) - p
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    if pdf > 0.0:
        u = err / pdf
        x = x - u / (1.0 + 0.5 * x * u)
    return mu + sigma * x


# --------------------------------------------------------------------------
# chi-square distribution (regularised incomplete gamma, Numerical Recipes)
# --------------------------------------------------------------------------
_ITMAX = 300
_EPS = 3.0e-14
_FPMIN = 1.0e-300


def _gamma_p_series(a: float, x: float) -> float:
    """Lower regularised incomplete gamma P(a, x) by its series expansion."""
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(_ITMAX):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q_continued_fraction(a: float, x: float) -> float:
    """Upper regularised incomplete gamma Q(a, x) by a continued fraction."""
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _ITMAX + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(x: float, df: int) -> float:
    """Upper tail probability of a chi-square distribution with ``df`` d.f."""
    if df < 1:
        raise ValueError("df must be >= 1")
    if x <= 0.0:
        return 1.0
    a = df / 2.0
    z = x / 2.0
    if z < a + 1.0:
        return max(0.0, min(1.0, 1.0 - _gamma_p_series(a, z)))
    return max(0.0, min(1.0, _gamma_q_continued_fraction(a, z)))


# --------------------------------------------------------------------------
# dense linear algebra
# --------------------------------------------------------------------------
def solve(matrix: Matrix, rhs: Sequence[float]) -> List[float]:
    """Solve ``matrix @ x = rhs`` by Gaussian elimination with partial pivoting."""
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("solve() requires a square matrix")
    if len(rhs) != n:
        raise ValueError("right-hand side length must match the matrix order")

    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-13:
            raise ValueError("matrix is singular to working precision")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot_value
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]

    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        acc = aug[row][n] - math.fsum(aug[row][k] * solution[k] for k in range(row + 1, n))
        solution[row] = acc / aug[row][row]
    return solution


def inverse(matrix: Matrix) -> Matrix:
    """Explicit matrix inverse, used for Cox standard errors."""
    n = len(matrix)
    columns = []
    for j in range(n):
        basis = [1.0 if i == j else 0.0 for i in range(n)]
        columns.append(solve(matrix, basis))
    return [[columns[j][i] for j in range(n)] for i in range(n)]
