"""Cox proportional-hazards regression, fitted by Newton-Raphson.

Why a Cox model rather than logistic regression on "did it breach": the
logistic model has to throw away every case that is still open, and those are
disproportionately the slow ones.  The Cox model uses them - an open case at
40 hours is genuine evidence that resolution took *more* than 40 hours - and
it estimates the effect on the whole time distribution rather than at one
arbitrary threshold.

Ties are handled with Breslow's approximation.  The partial likelihood, its
gradient and its Hessian are accumulated in a single pass over subjects sorted
by descending time, which keeps the fit O(n*p^2) rather than O(n^2*p^2) and
makes a pure-Python implementation practical on operational volumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from opslab.numeric import inverse, norm_sf, norm_ppf, solve

__all__ = ["CoxPHResult", "CoxPHModel", "fit_cox"]


@dataclass
class CoxPHResult:
    """Coefficients and inference for one fitted Cox model."""

    names: List[str]
    coefficients: List[float]
    standard_errors: List[float]
    log_likelihood: float
    null_log_likelihood: float
    n: int
    n_events: int
    iterations: int
    converged: bool
    confidence: float = 0.95

    @property
    def hazard_ratios(self) -> List[float]:
        """exp(beta): the multiplicative effect on the resolution hazard."""
        return [math.exp(b) for b in self.coefficients]

    @property
    def z_values(self) -> List[float]:
        """Wald z statistics."""
        return [
            b / se if se > 0 else float("nan")
            for b, se in zip(self.coefficients, self.standard_errors)
        ]

    @property
    def p_values(self) -> List[float]:
        """Two-sided Wald p-values."""
        return [
            2.0 * norm_sf(abs(z)) if z == z else float("nan") for z in self.z_values
        ]

    def confidence_intervals(self) -> List[Tuple[float, float]]:
        """Wald intervals for the coefficients on the log-hazard scale."""
        z = norm_ppf(0.5 + self.confidence / 2.0)
        return [
            (b - z * se, b + z * se)
            for b, se in zip(self.coefficients, self.standard_errors)
        ]

    @property
    def likelihood_ratio_statistic(self) -> float:
        """Overall model chi-square against the null model."""
        return 2.0 * (self.log_likelihood - self.null_log_likelihood)

    def coefficient(self, name: str) -> float:
        """One coefficient by name."""
        return self.coefficients[self.names.index(name)]

    def to_rows(self) -> List[dict]:
        """One row per covariate, ready for CSV or a report table."""
        intervals = self.confidence_intervals()
        return [
            {
                "term": name,
                "coef": round(b, 6),
                "hazard_ratio": round(math.exp(b), 6),
                "std_err": round(se, 6),
                "z": round(z, 4),
                "p_value": p,
                "ci_lower": round(lo, 6),
                "ci_upper": round(hi, 6),
            }
            for name, b, se, z, p, (lo, hi) in zip(
                self.names, self.coefficients, self.standard_errors,
                self.z_values, self.p_values, intervals,
            )
        ]

    def to_text(self) -> str:
        """A regression table in the conventional layout."""
        lines = [
            "Cox proportional hazards (Breslow ties)",
            "  n = %d, events = %d, log-likelihood = %.4f%s"
            % (self.n, self.n_events, self.log_likelihood,
               "" if self.converged else "  [DID NOT CONVERGE]"),
            "  %-24s %9s %9s %9s %9s %s"
            % ("term", "coef", "exp(coef)", "se", "z", "p"),
        ]
        for row in self.to_rows():
            lines.append(
                "  %-24s %9.4f %9.4f %9.4f %9.3f %.3g"
                % (row["term"], row["coef"], row["hazard_ratio"],
                   row["std_err"], row["z"], row["p_value"])
            )
        return "\n".join(lines)


@dataclass
class CoxPHModel:
    """A fitted model together with its Breslow baseline hazard."""

    result: CoxPHResult
    #: Distinct event times, ascending.
    baseline_times: List[float] = field(default_factory=list)
    #: Breslow cumulative baseline hazard at each event time.
    baseline_cumulative_hazard: List[float] = field(default_factory=list)

    def linear_predictor(self, covariates: Sequence[float]) -> float:
        """x'beta for one covariate vector."""
        if len(covariates) != len(self.result.coefficients):
            raise ValueError("covariate vector has the wrong length")
        return math.fsum(
            b * x for b, x in zip(self.result.coefficients, covariates)
        )

    def cumulative_hazard_at(self, time: float) -> float:
        """Baseline cumulative hazard at ``time`` (a right-continuous step)."""
        value = 0.0
        for t, h in zip(self.baseline_times, self.baseline_cumulative_hazard):
            if t <= time:
                value = h
            else:
                break
        return value

    def survival(self, covariates: Sequence[float], time: float) -> float:
        """P(case still open after ``time``) for a case with these covariates."""
        baseline = self.cumulative_hazard_at(time)
        return math.exp(-baseline * math.exp(self.linear_predictor(covariates)))

    def breach_probability(self, covariates: Sequence[float], sla_hours: float) -> float:
        """P(resolution takes longer than the SLA) - the breach risk score.

        Because the event modelled is resolution, the probability of *still
        being open* at the SLA target is exactly the probability of breaching
        it.
        """
        if sla_hours <= 0:
            raise ValueError("sla_hours must be positive")
        return self.survival(covariates, sla_hours)

    def risk_scores(self, rows: Sequence[Sequence[float]]) -> List[float]:
        """Relative risk exp(x'beta) for many cases."""
        return [math.exp(self.linear_predictor(row)) for row in rows]


def _prepare(
    durations: Sequence[float],
    events: Sequence[int],
    covariates: Sequence[Sequence[float]],
) -> Tuple[List[float], List[int], List[List[float]], int]:
    if not (len(durations) == len(events) == len(covariates)):
        raise ValueError("durations, events and covariates must be the same length")
    if not durations:
        raise ValueError("at least one observation is required")
    p = len(covariates[0])
    if p == 0:
        raise ValueError("at least one covariate is required")
    if any(len(row) != p for row in covariates):
        raise ValueError("all covariate rows must have the same length")
    if any(d < 0 for d in durations):
        raise ValueError("durations must be non-negative")
    flags = [int(bool(e)) for e in events]
    if sum(flags) == 0:
        raise ValueError("the data contain no events; a Cox model is not identifiable")
    return [float(d) for d in durations], flags, [list(map(float, r)) for r in covariates], p


def _partial_likelihood(
    beta: Sequence[float],
    order: Sequence[int],
    durations: Sequence[float],
    events: Sequence[int],
    covariates: Sequence[Sequence[float]],
    p: int,
) -> Tuple[float, List[float], List[List[float]]]:
    """Breslow log partial likelihood with its gradient and Hessian.

    ``order`` indexes subjects by descending duration, which lets the risk-set
    sums S0, S1 and S2 be accumulated once rather than rebuilt at every event
    time.
    """
    loglik = 0.0
    gradient = [0.0] * p
    hessian = [[0.0] * p for _ in range(p)]

    s0 = 0.0
    s1 = [0.0] * p
    s2 = [[0.0] * p for _ in range(p)]

    position = 0
    n = len(order)
    while position < n:
        time = durations[order[position]]
        # Everything at this time joins the risk set before the deaths here are scored.
        death_sum = [0.0] * p
        n_deaths = 0
        while position < n and durations[order[position]] == time:
            subject = order[position]
            x = covariates[subject]
            weight = math.exp(math.fsum(b * xi for b, xi in zip(beta, x)))
            s0 += weight
            for i in range(p):
                s1[i] += weight * x[i]
                wi = weight * x[i]
                for j in range(i, p):
                    s2[i][j] += wi * x[j]
            if events[subject]:
                n_deaths += 1
                loglik += math.fsum(b * xi for b, xi in zip(beta, x))
                for i in range(p):
                    death_sum[i] += x[i]
            position += 1

        if n_deaths == 0:
            continue

        loglik -= n_deaths * math.log(s0)
        ratio = [s1[i] / s0 for i in range(p)]
        for i in range(p):
            gradient[i] += death_sum[i] - n_deaths * ratio[i]
        for i in range(p):
            for j in range(i, p):
                value = n_deaths * (s2[i][j] / s0 - ratio[i] * ratio[j])
                hessian[i][j] -= value
                if i != j:
                    hessian[j][i] -= value

    return loglik, gradient, hessian


def fit_cox(
    durations: Sequence[float],
    events: Sequence[int],
    covariates: Sequence[Sequence[float]],
    names: Optional[Sequence[str]] = None,
    max_iterations: int = 50,
    tolerance: float = 1e-9,
    confidence: float = 0.95,
) -> CoxPHModel:
    """Fit a Cox proportional-hazards model.

    Parameters
    ----------
    durations:
        Observed time for each case - resolution time, or time observed so far
        for a case still open.
    events:
        1 when the case resolved, 0 when it was still open at the extract
        (right-censored).
    covariates:
        One row of covariate values per case.

    Newton-Raphson with step halving; the step is halved whenever a full step
    would decrease the log-likelihood, which keeps the fit stable on the
    near-collinear dummy sets operational data tends to produce.
    """
    durations, flags, matrix, p = _prepare(durations, events, covariates)
    labels = list(names) if names is not None else ["x%d" % i for i in range(p)]
    if len(labels) != p:
        raise ValueError("names length must match the number of covariates")

    order = sorted(range(len(durations)), key=lambda i: -durations[i])
    null_loglik, _, _ = _partial_likelihood([0.0] * p, order, durations, flags, matrix, p)

    beta = [0.0] * p
    loglik = null_loglik
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        _, gradient, hessian = _partial_likelihood(beta, order, durations, flags, matrix, p)
        negative_hessian = [[-hessian[i][j] for j in range(p)] for i in range(p)]
        try:
            step = solve(negative_hessian, gradient)
        except ValueError as error:
            raise ValueError(
                "Cox fit failed: the information matrix is singular, which "
                "usually means two covariates are collinear or a dummy level "
                "never varies (%s)" % error
            )

        scale = 1.0
        improved = False
        for _ in range(30):
            candidate = [b + scale * s for b, s in zip(beta, step)]
            candidate_loglik, _, _ = _partial_likelihood(
                candidate, order, durations, flags, matrix, p
            )
            if candidate_loglik >= loglik - 1e-12:
                beta, new_loglik, improved = candidate, candidate_loglik, True
                break
            scale *= 0.5
        if not improved:
            break
        if abs(new_loglik - loglik) < tolerance * (abs(loglik) + tolerance):
            loglik = new_loglik
            converged = True
            break
        loglik = new_loglik

    _, _, hessian = _partial_likelihood(beta, order, durations, flags, matrix, p)
    negative_hessian = [[-hessian[i][j] for j in range(p)] for i in range(p)]
    try:
        covariance = inverse(negative_hessian)
        standard_errors = [math.sqrt(max(covariance[i][i], 0.0)) for i in range(p)]
    except ValueError:
        standard_errors = [float("nan")] * p

    result = CoxPHResult(
        names=labels,
        coefficients=beta,
        standard_errors=standard_errors,
        log_likelihood=loglik,
        null_log_likelihood=null_loglik,
        n=len(durations),
        n_events=sum(flags),
        iterations=iterations,
        converged=converged,
        confidence=confidence,
    )

    times, cumulative = _breslow_baseline(beta, order, durations, flags, matrix)
    return CoxPHModel(result=result, baseline_times=times, baseline_cumulative_hazard=cumulative)


def _breslow_baseline(
    beta: Sequence[float],
    order: Sequence[int],
    durations: Sequence[float],
    events: Sequence[int],
    covariates: Sequence[Sequence[float]],
) -> Tuple[List[float], List[float]]:
    """Breslow estimate of the cumulative baseline hazard.

    Walks descending time to accumulate the risk-set denominator, then reverses
    to produce an ascending cumulative curve.
    """
    increments: List[Tuple[float, float]] = []
    s0 = 0.0
    position = 0
    n = len(order)
    while position < n:
        time = durations[order[position]]
        n_deaths = 0
        while position < n and durations[order[position]] == time:
            subject = order[position]
            x = covariates[subject]
            s0 += math.exp(math.fsum(b * xi for b, xi in zip(beta, x)))
            if events[subject]:
                n_deaths += 1
            position += 1
        if n_deaths and s0 > 0:
            increments.append((time, n_deaths / s0))

    increments.reverse()
    times: List[float] = []
    cumulative: List[float] = []
    running = 0.0
    for time, increment in increments:
        running += increment
        times.append(time)
        cumulative.append(running)
    return times, cumulative
