"""Non-parametric survival estimation for case turnaround times.

The problem this module exists to solve: at any extract date a share of cases
are still open, and their eventual duration is unknown - only bounded below.
Dropping them biases the average downwards (the slow cases are precisely the
ones still running); counting their current age as if final biases it the other
way.  Both are routine in operational reporting.  Treating the open cases as
*right-censored* is the correct handling, and it is what every estimator here
assumes.

The "event" throughout is **case resolution**, so a survival curve reads as
"share of cases still open after t working hours".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from opslab.numeric import chi2_sf, norm_ppf

__all__ = [
    "KaplanMeier",
    "kaplan_meier",
    "NelsonAalen",
    "nelson_aalen",
    "LogRankResult",
    "logrank_test",
]


def _validate(durations: Sequence[float], events: Sequence[int]) -> None:
    if len(durations) != len(events):
        raise ValueError("durations and events must be the same length")
    if not durations:
        raise ValueError("at least one observation is required")
    if any(d < 0 for d in durations):
        raise ValueError("durations must be non-negative")
    if any(e not in (0, 1, True, False) for e in events):
        raise ValueError("events must be 0/1 indicators")


def _risk_table(
    durations: Sequence[float], events: Sequence[int]
) -> List[Tuple[float, int, int, int]]:
    """Build ``(time, n_at_risk, n_events, n_censored)`` rows at each distinct time."""
    order = sorted(range(len(durations)), key=lambda i: durations[i])
    rows: List[Tuple[float, int, int, int]] = []
    n_remaining = len(durations)
    index = 0
    while index < len(order):
        time = durations[order[index]]
        n_events = 0
        n_censored = 0
        while index < len(order) and durations[order[index]] == time:
            if events[order[index]]:
                n_events += 1
            else:
                n_censored += 1
            index += 1
        rows.append((time, n_remaining, n_events, n_censored))
        n_remaining -= n_events + n_censored
    return rows


@dataclass
class KaplanMeier:
    """A fitted Kaplan-Meier survival curve."""

    times: List[float]
    survival: List[float]
    at_risk: List[int]
    n_events: List[int]
    n_censored: List[int]
    lower: List[float]
    upper: List[float]
    confidence: float
    label: str = ""

    @property
    def n(self) -> int:
        """Total observations that went into the fit."""
        return self.at_risk[0] if self.at_risk else 0

    def survival_at(self, time: float) -> float:
        """Survival probability at ``time`` (the step function, right-continuous)."""
        value = 1.0
        for t, s in zip(self.times, self.survival):
            if t <= time:
                value = s
            else:
                break
        return value

    def quantile(self, probability: float = 0.5) -> Optional[float]:
        """Smallest time at which survival drops to or below ``1 - probability``.

        Returns ``None`` when the curve never falls that far, which is the
        correct answer for a median that the follow-up did not reach - not an
        extrapolated guess.
        """
        if not 0.0 < probability < 1.0:
            raise ValueError("probability must lie strictly between 0 and 1")
        threshold = 1.0 - probability
        for t, s in zip(self.times, self.survival):
            if s <= threshold:
                return t
        return None

    @property
    def median(self) -> Optional[float]:
        """Median time to resolution, or ``None`` if not reached."""
        return self.quantile(0.5)

    def restricted_mean(self, horizon: float) -> float:
        """Area under the curve up to ``horizon`` (restricted mean survival time).

        This is the estimator to quote when the median is not reached: it is
        well defined for any follow-up length, provided the horizon is stated.
        """
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        area = 0.0
        previous_time = 0.0
        previous_survival = 1.0
        for t, s in zip(self.times, self.survival):
            if t >= horizon:
                break
            area += previous_survival * (t - previous_time)
            previous_time, previous_survival = t, s
        area += previous_survival * (horizon - previous_time)
        return area

    def to_rows(self) -> List[dict]:
        """CSV-ready rows."""
        return [
            {
                "time": round(t, 6),
                "at_risk": n,
                "events": d,
                "censored": c,
                "survival": round(s, 6),
                "lower": round(lo, 6),
                "upper": round(hi, 6),
            }
            for t, n, d, c, s, lo, hi in zip(
                self.times, self.at_risk, self.n_events, self.n_censored,
                self.survival, self.lower, self.upper,
            )
        ]


def kaplan_meier(
    durations: Sequence[float],
    events: Sequence[int],
    confidence: float = 0.95,
    label: str = "",
) -> KaplanMeier:
    """Fit a Kaplan-Meier curve with log-log transformed confidence bands.

    The log-log transform is used rather than the plain Greenwood interval
    because it keeps the band inside ``[0, 1]``, which the linear interval does
    not once the curve approaches either end.
    """
    _validate(durations, events)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")

    z = norm_ppf(0.5 + confidence / 2.0)
    rows = _risk_table(durations, [int(bool(e)) for e in events])

    times: List[float] = []
    survival: List[float] = []
    at_risk: List[int] = []
    n_events: List[int] = []
    n_censored: List[int] = []
    lower: List[float] = []
    upper: List[float] = []

    running_survival = 1.0
    greenwood = 0.0
    for time, n, d, c in rows:
        if d > 0:
            running_survival *= (1.0 - d / n)
            if n > d:
                greenwood += d / (n * (n - d))
            else:
                greenwood = float("inf")
        times.append(time)
        survival.append(running_survival)
        at_risk.append(n)
        n_events.append(d)
        n_censored.append(c)

        if running_survival >= 1.0 or running_survival <= 0.0 or not math.isfinite(greenwood):
            lower.append(max(0.0, min(1.0, running_survival)))
            upper.append(max(0.0, min(1.0, running_survival)))
            continue
        log_survival = math.log(running_survival)
        se_loglog = math.sqrt(greenwood) / abs(log_survival)
        half = z * se_loglog
        lower.append(running_survival ** math.exp(half))
        upper.append(running_survival ** math.exp(-half))

    return KaplanMeier(
        times=times,
        survival=survival,
        at_risk=at_risk,
        n_events=n_events,
        n_censored=n_censored,
        lower=lower,
        upper=upper,
        confidence=confidence,
        label=label,
    )


@dataclass
class NelsonAalen:
    """A fitted Nelson-Aalen cumulative hazard estimate."""

    times: List[float]
    cumulative_hazard: List[float]
    at_risk: List[int]
    n_events: List[int]

    def hazard_at(self, time: float) -> float:
        """Cumulative hazard at ``time``."""
        value = 0.0
        for t, h in zip(self.times, self.cumulative_hazard):
            if t <= time:
                value = h
            else:
                break
        return value


def nelson_aalen(durations: Sequence[float], events: Sequence[int]) -> NelsonAalen:
    """Fit the Nelson-Aalen cumulative hazard estimator.

    Preferred over Kaplan-Meier when the question is about the *rate* at which
    cases clear rather than the share still open - for instance when comparing
    the effect of a staffing change on throughput.
    """
    _validate(durations, events)
    rows = _risk_table(durations, [int(bool(e)) for e in events])
    times: List[float] = []
    hazards: List[float] = []
    at_risk: List[int] = []
    n_events: List[int] = []
    running = 0.0
    for time, n, d, _ in rows:
        if d > 0:
            running += d / n
        times.append(time)
        hazards.append(running)
        at_risk.append(n)
        n_events.append(d)
    return NelsonAalen(times, hazards, at_risk, n_events)


@dataclass
class LogRankResult:
    """Outcome of a two-group log-rank test."""

    observed_a: float
    expected_a: float
    observed_b: float
    expected_b: float
    variance: float
    statistic: float
    p_value: float
    label_a: str = "A"
    label_b: str = "B"

    def to_text(self) -> str:
        """A one-block summary."""
        return (
            "Log-rank test: %s vs %s\n"
            "  %-10s observed %.0f, expected %.2f\n"
            "  %-10s observed %.0f, expected %.2f\n"
            "  chi-square %.3f on 1 df, p = %.4g"
            % (
                self.label_a, self.label_b,
                self.label_a, self.observed_a, self.expected_a,
                self.label_b, self.observed_b, self.expected_b,
                self.statistic, self.p_value,
            )
        )


def logrank_test(
    durations_a: Sequence[float],
    events_a: Sequence[int],
    durations_b: Sequence[float],
    events_b: Sequence[int],
    label_a: str = "A",
    label_b: str = "B",
) -> LogRankResult:
    """Compare two survival curves with the log-rank test.

    This is the right test for "did turnaround improve after the change?" when
    some cases are still open: a t-test on the closed cases silently conditions
    on the outcome it is trying to measure.
    """
    _validate(durations_a, events_a)
    _validate(durations_b, events_b)

    events_a = [int(bool(e)) for e in events_a]
    events_b = [int(bool(e)) for e in events_b]
    all_times = sorted({d for d, e in zip(durations_a, events_a) if e}
                       | {d for d, e in zip(durations_b, events_b) if e})

    observed_a = 0.0
    expected_a = 0.0
    variance = 0.0
    for time in all_times:
        n_a = sum(1 for d in durations_a if d >= time)
        n_b = sum(1 for d in durations_b if d >= time)
        d_a = sum(1 for d, e in zip(durations_a, events_a) if d == time and e)
        d_b = sum(1 for d, e in zip(durations_b, events_b) if d == time and e)
        n = n_a + n_b
        d = d_a + d_b
        if n <= 1 or d == 0:
            continue
        observed_a += d_a
        expected_a += d * n_a / n
        variance += d * (n - d) * n_a * n_b / (n * n * (n - 1))

    total_events_a = sum(events_a)
    total_events_b = sum(events_b)
    expected_b = (total_events_a + total_events_b) - expected_a

    if variance <= 0.0:
        statistic, p_value = 0.0, 1.0
    else:
        statistic = (observed_a - expected_a) ** 2 / variance
        p_value = chi2_sf(statistic, 1)

    return LogRankResult(
        observed_a=float(total_events_a),
        expected_a=expected_a,
        observed_b=float(total_events_b),
        expected_b=expected_b,
        variance=variance,
        statistic=statistic,
        p_value=p_value,
        label_a=label_a,
        label_b=label_b,
    )
