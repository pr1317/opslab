"""Control charts for variables and attributes data.

Chart selection is not cosmetic.  Daily SLA breach counts are binomial and
belong on a p-chart with limits that widen on low-volume days; putting them on
an individuals chart with flat limits manufactures special causes every bank
holiday.  Each constructor here states the data model it assumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from opslab.numeric import mean
from opslab.spc.rules import NelsonRule, apply_nelson_rules

__all__ = [
    "ChartPoint",
    "ControlChart",
    "individuals_chart",
    "xbar_r_chart",
    "p_chart",
    "c_chart",
    "u_chart",
    "D2",
    "D3_CONST",
    "A2",
    "D3",
    "D4",
]

#: Unbiasing constant d2 relating the mean range to sigma, by subgroup size.
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
      7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}

#: Standard deviation of the relative range, by subgroup size.
D3_CONST = {2: 0.853, 3: 0.888, 4: 0.880, 5: 0.864, 6: 0.848,
            7: 0.833, 8: 0.820, 9: 0.808, 10: 0.797}

#: X-bar chart limit factor, by subgroup size.
A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483,
      7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}

#: Lower range-chart limit factor, by subgroup size.
D3 = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0,
      7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}

#: Upper range-chart limit factor, by subgroup size.
D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004,
      7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}


@dataclass
class ChartPoint:
    """One plotted point, carrying its own limits so charts may vary them."""

    index: int
    label: str
    value: float
    center: float
    lower: float
    upper: float
    sigma: float
    violations: List[str] = field(default_factory=list)

    @property
    def standardised(self) -> float:
        """Deviation from the centre line in sigma units."""
        if self.sigma <= 0.0:
            return 0.0
        return (self.value - self.center) / self.sigma

    @property
    def out_of_limits(self) -> bool:
        """True when the point sits outside its own control limits."""
        return self.value > self.upper or self.value < self.lower


@dataclass
class ControlChart:
    """A completed control chart."""

    kind: str
    points: List[ChartPoint]
    center: float
    #: Within-subgroup sigma estimate; ``None`` for varying-limit charts.
    sigma: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def signals(self) -> List[ChartPoint]:
        """Points that broke at least one rule."""
        return [p for p in self.points if p.violations]

    def in_control(self) -> bool:
        """True when no point triggered any rule."""
        return not self.signals()

    def to_rows(self) -> List[dict]:
        """A flat, CSV-ready representation."""
        return [
            {
                "index": p.index,
                "label": p.label,
                "value": round(p.value, 6),
                "center": round(p.center, 6),
                "lcl": round(p.lower, 6),
                "ucl": round(p.upper, 6),
                "sigma": round(p.sigma, 6),
                "violations": "|".join(p.violations),
            }
            for p in self.points
        ]

    def to_text(self, width: int = 52) -> str:
        """An ASCII rendering, for terminals and CI logs."""
        if not self.points:
            return "%s: no data" % self.kind
        low = min(min(p.lower, p.value) for p in self.points)
        high = max(max(p.upper, p.value) for p in self.points)
        span = (high - low) or 1.0
        lines = ["%s (centre %.4g)" % (self.kind, self.center)]
        for point in self.points:
            slot = int(round((point.value - low) / span * (width - 1)))
            row = [" "] * width
            for marker, position in (
                ("|", int(round((point.lower - low) / span * (width - 1)))),
                ("|", int(round((point.upper - low) / span * (width - 1)))),
                (":", int(round((point.center - low) / span * (width - 1)))),
            ):
                if 0 <= position < width:
                    row[position] = marker
            row[max(0, min(width - 1, slot))] = "*" if point.violations else "o"
            flag = (" <- " + ",".join(point.violations)) if point.violations else ""
            lines.append("%-12s %s %10.4g%s" % (point.label[:12], "".join(row), point.value, flag))
        return "\n".join(lines)


def _labels(labels: Optional[Sequence[str]], count: int) -> List[str]:
    if labels is None:
        return [str(i + 1) for i in range(count)]
    if len(labels) != count:
        raise ValueError("labels length must match the number of points")
    return list(labels)


def _finish(
    kind: str,
    points: List[ChartPoint],
    center: float,
    sigma: Optional[float],
    rules: Optional[Sequence[NelsonRule]],
    notes: Optional[List[str]] = None,
) -> ControlChart:
    """Apply run rules and assemble the chart."""
    flags = apply_nelson_rules([p.standardised for p in points], rules)
    for point, codes in zip(points, flags):
        point.violations = list(codes)
    return ControlChart(kind=kind, points=points, center=center, sigma=sigma,
                        notes=list(notes or []))


def individuals_chart(
    values: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    rules: Optional[Sequence[NelsonRule]] = None,
) -> ControlChart:
    """Individuals (X) chart with sigma estimated from the average moving range.

    The moving-range estimator is deliberate: it uses only the variation
    *between consecutive observations*, so a sustained shift inflates the chart
    signal rather than quietly widening the limits, which is what the overall
    standard deviation would do.
    """
    if len(values) < 2:
        raise ValueError("individuals_chart() needs at least two observations")
    series = [float(v) for v in values]
    ranges = [abs(series[i + 1] - series[i]) for i in range(len(series) - 1)]
    center = mean(series)
    sigma = mean(ranges) / D2[2]
    names = _labels(labels, len(series))
    points = [
        ChartPoint(
            index=i,
            label=names[i],
            value=series[i],
            center=center,
            lower=center - 3.0 * sigma,
            upper=center + 3.0 * sigma,
            sigma=sigma,
        )
        for i in range(len(series))
    ]
    return _finish("Individuals (X)", points, center, sigma, rules)


def moving_range_chart(
    values: Sequence[float],
    labels: Optional[Sequence[str]] = None,
) -> ControlChart:
    """The moving-range companion to :func:`individuals_chart`."""
    if len(values) < 2:
        raise ValueError("moving_range_chart() needs at least two observations")
    series = [float(v) for v in values]
    ranges = [abs(series[i + 1] - series[i]) for i in range(len(series) - 1)]
    center = mean(ranges)
    sigma = center * D3_CONST[2] / D2[2]
    names = _labels(labels, len(series))[1:]
    points = [
        ChartPoint(
            index=i,
            label=names[i],
            value=ranges[i],
            center=center,
            lower=max(0.0, D3[2] * center),
            upper=D4[2] * center,
            sigma=sigma,
        )
        for i in range(len(ranges))
    ]
    # Only the beyond-limits test is meaningful on a range chart.
    from opslab.spc.rules import NELSON_RULES

    return _finish("Moving range (mR)", points, center, sigma, (NELSON_RULES[0],))


def xbar_r_chart(
    subgroups: Sequence[Sequence[float]],
    labels: Optional[Sequence[str]] = None,
    rules: Optional[Sequence[NelsonRule]] = None,
) -> ControlChart:
    """X-bar chart with limits from the average subgroup range.

    Requires equally sized subgroups of 2 to 10 observations, the range over
    which the classical control-chart constants are tabulated.
    """
    if not subgroups:
        raise ValueError("xbar_r_chart() needs at least one subgroup")
    size = len(subgroups[0])
    if any(len(g) != size for g in subgroups):
        raise ValueError("xbar_r_chart() requires equally sized subgroups")
    if size not in A2:
        raise ValueError("subgroup size must be between 2 and 10, got %d" % size)

    means = [mean([float(v) for v in g]) for g in subgroups]
    ranges = [max(g) - min(g) for g in subgroups]
    center = mean(means)
    rbar = mean(ranges)
    sigma = rbar / D2[size]
    half_width = A2[size] * rbar
    names = _labels(labels, len(subgroups))
    points = [
        ChartPoint(
            index=i,
            label=names[i],
            value=means[i],
            center=center,
            lower=center - half_width,
            upper=center + half_width,
            sigma=sigma / math.sqrt(size),
        )
        for i in range(len(means))
    ]
    return _finish("X-bar (R)", points, center, sigma, rules)


def p_chart(
    defectives: Sequence[int],
    sizes: Sequence[int],
    labels: Optional[Sequence[str]] = None,
    rules: Optional[Sequence[NelsonRule]] = None,
) -> ControlChart:
    """Proportion-defective chart with limits that follow subgroup size.

    This is the right chart for a daily SLA breach rate: the limits widen on
    quiet days, which stops a single breach out of nine cases being reported as
    a process failure.
    """
    if len(defectives) != len(sizes):
        raise ValueError("defectives and sizes must be the same length")
    if not defectives:
        raise ValueError("p_chart() needs at least one subgroup")
    if any(n <= 0 for n in sizes):
        raise ValueError("subgroup sizes must be positive")
    if any(d < 0 or d > n for d, n in zip(defectives, sizes)):
        raise ValueError("each defective count must lie between 0 and its subgroup size")

    total_defective = sum(defectives)
    total_size = sum(sizes)
    center = total_defective / total_size
    names = _labels(labels, len(defectives))
    points: List[ChartPoint] = []
    for i, (d, n) in enumerate(zip(defectives, sizes)):
        sigma = math.sqrt(max(center * (1.0 - center), 0.0) / n)
        points.append(
            ChartPoint(
                index=i,
                label=names[i],
                value=d / n,
                center=center,
                lower=max(0.0, center - 3.0 * sigma),
                upper=min(1.0, center + 3.0 * sigma),
                sigma=sigma,
            )
        )
    notes = []
    if min(sizes) * center < 5:
        notes.append(
            "Smallest subgroup expects fewer than 5 defectives; the normal "
            "approximation behind the limits is weak - treat single-point "
            "signals with caution."
        )
    return _finish("p (proportion defective)", points, center, None, rules, notes)


def c_chart(
    counts: Sequence[int],
    labels: Optional[Sequence[str]] = None,
    rules: Optional[Sequence[NelsonRule]] = None,
) -> ControlChart:
    """Count-of-defects chart for a constant area of opportunity (Poisson)."""
    if not counts:
        raise ValueError("c_chart() needs at least one observation")
    if any(c < 0 for c in counts):
        raise ValueError("counts must be non-negative")
    center = mean([float(c) for c in counts])
    sigma = math.sqrt(center)
    names = _labels(labels, len(counts))
    points = [
        ChartPoint(
            index=i,
            label=names[i],
            value=float(counts[i]),
            center=center,
            lower=max(0.0, center - 3.0 * sigma),
            upper=center + 3.0 * sigma,
            sigma=sigma,
        )
        for i in range(len(counts))
    ]
    return _finish("c (defect count)", points, center, sigma, rules)


def u_chart(
    counts: Sequence[int],
    exposures: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    rules: Optional[Sequence[NelsonRule]] = None,
) -> ControlChart:
    """Defects-per-unit chart for a varying area of opportunity."""
    if len(counts) != len(exposures):
        raise ValueError("counts and exposures must be the same length")
    if not counts:
        raise ValueError("u_chart() needs at least one observation")
    if any(e <= 0 for e in exposures):
        raise ValueError("exposures must be positive")
    center = sum(counts) / math.fsum(exposures)
    names = _labels(labels, len(counts))
    points: List[ChartPoint] = []
    for i, (c, e) in enumerate(zip(counts, exposures)):
        sigma = math.sqrt(center / e)
        points.append(
            ChartPoint(
                index=i,
                label=names[i],
                value=c / e,
                center=center,
                lower=max(0.0, center - 3.0 * sigma),
                upper=center + 3.0 * sigma,
                sigma=sigma,
            )
        )
    return _finish("u (defects per unit)", points, center, None, rules)
