"""Process capability and sigma level.

Two distinctions are kept explicit here because both are routinely fudged in
operational reporting:

* **Capability (Cp/Cpk) versus performance (Pp/Ppk).**  Capability uses the
  within-subgroup sigma, i.e. what the process could do if it held still;
  performance uses the overall sigma, i.e. what it actually delivered.  When
  the two diverge the process is drifting, and quoting only the flattering one
  hides that.
* **One-sided specifications.**  Cycle time has an upper limit and no lower
  one.  Cp is undefined in that case, and reporting a two-sided figure by
  inventing a mirror limit overstates capability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

from opslab.numeric import mean, norm_cdf, norm_ppf, norm_sf, stdev
from opslab.spc.charts import D2

__all__ = ["CapabilityReport", "capability", "sigma_level_from_dpmo"]

#: The conventional long-term drift allowance built into "six sigma" quoting.
LONG_TERM_SHIFT = 1.5


@dataclass
class CapabilityReport:
    """Capability, performance and defect-rate summary for one characteristic."""

    n: int
    mean: float
    sigma_within: float
    sigma_overall: float
    lsl: Optional[float]
    usl: Optional[float]
    target: Optional[float]
    cp: Optional[float]
    cpk: Optional[float]
    pp: Optional[float]
    ppk: Optional[float]
    cpm: Optional[float]
    #: Expected defective fraction under a fitted normal, using overall sigma.
    expected_defective: float
    #: Observed defective fraction in the sample.
    observed_defective: float
    dpmo: float
    #: Short-term sigma level, i.e. the benchmark Z plus the 1.5 shift.
    sigma_level: float
    notes: List[str]

    def to_dict(self) -> dict:
        """A flat dictionary, ready for JSON or a reporting table."""
        def rounded(value: Optional[float]) -> Optional[float]:
            return None if value is None else round(value, 6)

        return {
            "n": self.n,
            "mean": rounded(self.mean),
            "sigma_within": rounded(self.sigma_within),
            "sigma_overall": rounded(self.sigma_overall),
            "lsl": self.lsl,
            "usl": self.usl,
            "target": self.target,
            "cp": rounded(self.cp),
            "cpk": rounded(self.cpk),
            "pp": rounded(self.pp),
            "ppk": rounded(self.ppk),
            "cpm": rounded(self.cpm),
            "expected_defective": rounded(self.expected_defective),
            "observed_defective": rounded(self.observed_defective),
            "dpmo": round(self.dpmo, 2),
            "sigma_level": rounded(self.sigma_level),
            "notes": list(self.notes),
        }

    def to_text(self) -> str:
        """A short report in the layout a process owner expects."""
        def show(label: str, value: Optional[float]) -> str:
            return "  %-18s %s" % (label, "n/a" if value is None else "%.3f" % value)

        lines = [
            "Capability report (n=%d)" % self.n,
            show("mean", self.mean),
            show("sigma (within)", self.sigma_within),
            show("sigma (overall)", self.sigma_overall),
            show("Cp", self.cp),
            show("Cpk", self.cpk),
            show("Pp", self.pp),
            show("Ppk", self.ppk),
            show("Cpm", self.cpm),
            "  %-18s %.0f" % ("DPMO", self.dpmo),
            show("sigma level", self.sigma_level),
            "  %-18s %.4f (observed %.4f)"
            % ("expected defect p", self.expected_defective, self.observed_defective),
        ]
        lines.extend("  ! " + note for note in self.notes)
        return "\n".join(lines)


def _within_sigma(values: Sequence[float], subgroup_size: int) -> float:
    """Estimate within-subgroup sigma from the average moving range."""
    if subgroup_size != 1:
        raise ValueError("only individuals data (subgroup_size=1) is supported here")
    ranges = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    return mean(ranges) / D2[2]


def sigma_level_from_dpmo(dpmo: float, shift: float = LONG_TERM_SHIFT) -> float:
    """Convert a defect rate to a short-term sigma level.

    The benchmark Z is taken from the lower tail and negated rather than
    computed as ``norm_ppf(1 - p)``: for the very small defect rates a capable
    process produces, ``1 - p`` rounds to exactly 1.0 in double precision and
    the quantile function is undefined there.

    The result is capped at 6 sigma before the shift is added.  Beyond that the
    figure is an extrapolation of a fitted normal tail far outside the range of
    any real sample, and quoting it would imply a precision the data cannot
    support.
    """
    if not 0.0 <= dpmo <= 1_000_000.0:
        raise ValueError("dpmo must lie between 0 and 1,000,000")
    proportion = dpmo / 1_000_000.0
    if proportion <= 0.0:
        return 6.0 + shift
    if proportion >= 1.0:
        return 0.0
    return min(-norm_ppf(proportion), 6.0) + shift


def capability(
    values: Sequence[float],
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
    target: Optional[float] = None,
) -> CapabilityReport:
    """Compute capability indices for a sample of individual measurements.

    At least one specification limit is required.  Passing only ``usl`` - the
    normal situation for a turnaround-time SLA - yields a one-sided report in
    which ``Cp`` and ``Pp`` are ``None`` by definition.
    """
    series = [float(v) for v in values]
    if len(series) < 2:
        raise ValueError("capability() needs at least two observations")
    if lsl is None and usl is None:
        raise ValueError("at least one specification limit is required")
    if lsl is not None and usl is not None and lsl >= usl:
        raise ValueError("lsl must be below usl")

    mu = mean(series)
    sigma_overall = stdev(series)
    sigma_within = _within_sigma(series, 1)
    notes: List[str] = []
    if sigma_within <= 0.0 or sigma_overall <= 0.0:
        raise ValueError("cannot assess capability on a constant sample")

    two_sided = lsl is not None and usl is not None
    cp = (usl - lsl) / (6.0 * sigma_within) if two_sided else None
    pp = (usl - lsl) / (6.0 * sigma_overall) if two_sided else None

    def one_sided_k(sigma: float) -> Optional[float]:
        candidates = []
        if usl is not None:
            candidates.append((usl - mu) / (3.0 * sigma))
        if lsl is not None:
            candidates.append((mu - lsl) / (3.0 * sigma))
        return min(candidates) if candidates else None

    cpk = one_sided_k(sigma_within)
    ppk = one_sided_k(sigma_overall)

    cpm = None
    if two_sided and target is not None:
        denominator = math.sqrt(sigma_overall ** 2 + (mu - target) ** 2)
        cpm = (usl - lsl) / (6.0 * denominator) if denominator > 0 else None

    expected = 0.0
    if usl is not None:
        expected += norm_sf(usl, mu, sigma_overall)
    if lsl is not None:
        expected += norm_cdf(lsl, mu, sigma_overall)
    expected = min(max(expected, 0.0), 1.0)

    observed = sum(
        1
        for v in series
        if (usl is not None and v > usl) or (lsl is not None and v < lsl)
    ) / len(series)

    dpmo = expected * 1_000_000.0
    level = sigma_level_from_dpmo(dpmo)

    if not two_sided:
        notes.append(
            "One-sided specification: Cp and Pp are undefined and reported as n/a."
        )
    if sigma_within > 0 and sigma_overall / sigma_within > 1.3:
        notes.append(
            "Overall sigma exceeds within-subgroup sigma by more than 30%%: the "
            "process is shifting over time, so Ppk (%.2f) is the honest figure "
            "to quote, not Cpk (%.2f)." % (ppk if ppk is not None else float("nan"),
                                           cpk if cpk is not None else float("nan"))
        )
    if abs(observed - expected) > max(0.05, 0.5 * expected):
        notes.append(
            "Observed defect rate (%.4f) differs materially from the normal-model "
            "expectation (%.4f); the data are probably skewed, which cycle times "
            "usually are - prefer the observed rate." % (observed, expected)
        )

    return CapabilityReport(
        n=len(series),
        mean=mu,
        sigma_within=sigma_within,
        sigma_overall=sigma_overall,
        lsl=lsl,
        usl=usl,
        target=target,
        cp=cp,
        cpk=cpk,
        pp=pp,
        ppk=ppk,
        cpm=cpm,
        expected_defective=expected,
        observed_defective=observed,
        dpmo=dpmo,
        sigma_level=level,
        notes=notes,
    )
