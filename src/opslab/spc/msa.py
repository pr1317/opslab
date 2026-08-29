"""Measurement system analysis: crossed ANOVA Gage R&R.

Before a team argues about whether handling time improved, it is worth knowing
how much of the observed variation is the measurement system itself - in a
back office, that is the consistency with which assessors score the same case.
This is the standard crossed design: every operator measures every part,
several times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from opslab.numeric import mean

__all__ = ["GageRRReport", "gage_rr"]

#: Multiplier converting a standard deviation to the conventional study range.
STUDY_VARIATION_MULTIPLIER = 6.0


@dataclass
class GageRRReport:
    """Variance components and the usual acceptance figures."""

    n_parts: int
    n_operators: int
    n_replicates: int
    var_repeatability: float
    var_reproducibility: float
    var_interaction: float
    var_part: float
    #: Total gage variation: repeatability plus reproducibility.
    var_gage: float
    var_total: float
    #: Number of distinct categories the system can resolve.
    ndc: int
    notes: List[str]

    def contribution(self) -> Dict[str, float]:
        """Percentage of total variance attributable to each component."""
        if self.var_total <= 0:
            return {}
        return {
            "repeatability": 100.0 * self.var_repeatability / self.var_total,
            "reproducibility": 100.0 * self.var_reproducibility / self.var_total,
            "interaction": 100.0 * self.var_interaction / self.var_total,
            "gage_rr": 100.0 * self.var_gage / self.var_total,
            "part_to_part": 100.0 * self.var_part / self.var_total,
        }

    def study_variation(self) -> Dict[str, float]:
        """Percentage of study variation (standard deviations, not variances)."""
        if self.var_total <= 0:
            return {}
        total_sd = math.sqrt(self.var_total)
        return {
            name: 100.0 * math.sqrt(var) / total_sd
            for name, var in (
                ("repeatability", self.var_repeatability),
                ("reproducibility", self.var_reproducibility),
                ("interaction", self.var_interaction),
                ("gage_rr", self.var_gage),
                ("part_to_part", self.var_part),
            )
        }

    @property
    def verdict(self) -> str:
        """AIAG-style acceptance call based on %study variation."""
        pct = self.study_variation().get("gage_rr", 100.0)
        if pct < 10.0:
            return "acceptable"
        if pct < 30.0:
            return "marginal"
        return "unacceptable"

    def to_text(self) -> str:
        """A compact report."""
        study = self.study_variation()
        contribution = self.contribution()
        lines = [
            "Gage R&R (%d parts x %d operators x %d replicates)"
            % (self.n_parts, self.n_operators, self.n_replicates),
            "  %-16s %10s %10s" % ("component", "%contrib", "%study"),
        ]
        for name in ("repeatability", "reproducibility", "interaction", "gage_rr", "part_to_part"):
            lines.append(
                "  %-16s %9.2f%% %9.2f%%"
                % (name, contribution.get(name, 0.0), study.get(name, 0.0))
            )
        lines.append("  %-16s %d" % ("distinct categories", self.ndc))
        lines.append("  %-16s %s" % ("verdict", self.verdict))
        lines.extend("  ! " + note for note in self.notes)
        return "\n".join(lines)


def _grand_mean(measurements: Dict[Tuple[str, str], Sequence[float]]) -> float:
    values = [v for series in measurements.values() for v in series]
    return mean(values)


def gage_rr(
    measurements: Dict[Tuple[str, str], Sequence[float]],
) -> GageRRReport:
    """Run a crossed ANOVA Gage R&R.

    Parameters
    ----------
    measurements:
        Maps ``(part, operator)`` to that operator's repeated readings of that
        part.  The design must be balanced: every operator measures every part
        the same number of times.

    Negative variance components, which the ANOVA can produce when a true
    component is near zero, are truncated at zero in the conventional way and
    noted in the report.
    """
    if not measurements:
        raise ValueError("gage_rr() needs measurements")

    parts = sorted({part for part, _ in measurements})
    operators = sorted({operator for _, operator in measurements})
    n_parts, n_ops = len(parts), len(operators)
    if n_parts < 2 or n_ops < 2:
        raise ValueError("a crossed study needs at least 2 parts and 2 operators")

    replicate_counts = {len(v) for v in measurements.values()}
    if len(replicate_counts) != 1:
        raise ValueError("unbalanced design: every cell needs the same replicate count")
    n_reps = replicate_counts.pop()
    if n_reps < 2:
        raise ValueError("at least 2 replicates per cell are required to separate repeatability")
    if len(measurements) != n_parts * n_ops:
        raise ValueError("unbalanced design: every operator must measure every part")

    grand = _grand_mean(measurements)
    cell_means = {key: mean(vals) for key, vals in measurements.items()}
    part_means = {
        p: mean([cell_means[(p, o)] for o in operators]) for p in parts
    }
    op_means = {
        o: mean([cell_means[(p, o)] for p in parts]) for o in operators
    }

    ss_part = n_ops * n_reps * math.fsum((part_means[p] - grand) ** 2 for p in parts)
    ss_op = n_parts * n_reps * math.fsum((op_means[o] - grand) ** 2 for o in operators)
    ss_interaction = n_reps * math.fsum(
        (cell_means[(p, o)] - part_means[p] - op_means[o] + grand) ** 2
        for p in parts
        for o in operators
    )
    ss_error = math.fsum(
        (v - cell_means[key]) ** 2 for key, vals in measurements.items() for v in vals
    )

    df_part = n_parts - 1
    df_op = n_ops - 1
    df_interaction = df_part * df_op
    df_error = n_parts * n_ops * (n_reps - 1)

    ms_part = ss_part / df_part
    ms_op = ss_op / df_op
    ms_interaction = ss_interaction / df_interaction
    ms_error = ss_error / df_error

    notes: List[str] = []

    def truncate(value: float, label: str) -> float:
        if value < 0.0:
            notes.append(
                "%s variance estimated below zero (%.5g) and truncated to 0; the "
                "component is not distinguishable from noise here." % (label, value)
            )
            return 0.0
        return value

    var_repeat = ms_error
    var_interaction = truncate((ms_interaction - ms_error) / n_reps, "interaction")
    var_op = truncate((ms_op - ms_interaction) / (n_parts * n_reps), "operator")
    var_part = truncate((ms_part - ms_interaction) / (n_ops * n_reps), "part")

    var_gage = var_repeat + var_op + var_interaction
    var_total = var_gage + var_part

    if var_gage > 0 and var_part > 0:
        ndc = int(math.floor(1.41 * math.sqrt(var_part / var_gage)))
    else:
        ndc = 0
    if ndc < 5:
        notes.append(
            "Fewer than 5 distinct categories: the measurement system cannot "
            "reliably rank the parts it is being used to judge."
        )

    return GageRRReport(
        n_parts=n_parts,
        n_operators=n_ops,
        n_replicates=n_reps,
        var_repeatability=var_repeat,
        var_reproducibility=var_op,
        var_interaction=var_interaction,
        var_part=var_part,
        var_gage=var_gage,
        var_total=var_total,
        ndc=ndc,
        notes=notes,
    )
