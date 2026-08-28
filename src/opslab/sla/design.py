"""Turn a case table into a numeric design matrix.

Kept separate from the model so the encoding is inspectable: which level became
the reference, which columns are dummies, and what the raw values were.  Silent
one-hot encoding inside a fit method is a reliable source of coefficients
nobody can interpret.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from opslab.eventlog import CaseTable

__all__ = ["DesignMatrix", "design_matrix"]


@dataclass
class DesignMatrix:
    """A numeric matrix plus the metadata needed to read it back."""

    names: List[str]
    rows: List[List[float]]
    case_ids: List[str]
    #: Reference level chosen for each categorical input.
    references: Dict[str, str] = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Number of rows."""
        return len(self.rows)

    @property
    def p(self) -> int:
        """Number of columns."""
        return len(self.names)

    def column(self, name: str) -> List[float]:
        """One column by name."""
        index = self.names.index(name)
        return [row[index] for row in self.rows]


def _as_float(value: str, column: str, case_id: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            "case %s: column %r is not numeric (%r)" % (case_id, column, value)
        )


def design_matrix(
    cases: CaseTable,
    numeric: Sequence[str] = (),
    categorical: Sequence[str] = (),
    references: Optional[Dict[str, str]] = None,
    binary: Optional[Dict[str, str]] = None,
) -> DesignMatrix:
    """Build a design matrix from case attributes.

    Parameters
    ----------
    numeric:
        Attributes used as-is, parsed as floats.
    categorical:
        Attributes dummy-coded with one level held out as the reference.
    references:
        Explicit reference level per categorical attribute; defaults to the
        alphabetically first observed level.
    binary:
        Attributes reduced to a single 0/1 indicator, mapping attribute name to
        the level that counts as 1.  Clearer than a dummy set when only one
        level is of interest, e.g. ``{"priority": "Urgent"}``.
    """
    case_list = list(cases)
    if not case_list:
        raise ValueError("design_matrix() needs at least one case")

    references = dict(references or {})
    binary = dict(binary or {})

    levels: Dict[str, List[str]] = {}
    for attribute in categorical:
        observed = sorted({c.attributes.get(attribute, "") for c in case_list})
        if len(observed) < 2:
            raise ValueError(
                "categorical attribute %r has fewer than two levels" % attribute
            )
        reference = references.get(attribute, observed[0])
        if reference not in observed:
            raise ValueError(
                "reference level %r not observed for %r" % (reference, attribute)
            )
        references[attribute] = reference
        levels[attribute] = [level for level in observed if level != reference]

    names: List[str] = list(numeric)
    for attribute, level in binary.items():
        names.append("%s_%s" % (attribute, level.lower().replace(" ", "_")))
    for attribute in categorical:
        for level in levels[attribute]:
            names.append("%s_%s" % (attribute, level.lower().replace(" ", "_")))

    rows: List[List[float]] = []
    for case in case_list:
        row: List[float] = []
        for attribute in numeric:
            if attribute not in case.attributes:
                raise ValueError("case %s has no attribute %r" % (case.case_id, attribute))
            row.append(_as_float(case.attributes[attribute], attribute, case.case_id))
        for attribute, level in binary.items():
            row.append(1.0 if case.attributes.get(attribute) == level else 0.0)
        for attribute in categorical:
            value = case.attributes.get(attribute, "")
            for level in levels[attribute]:
                row.append(1.0 if value == level else 0.0)
        rows.append(row)

    return DesignMatrix(
        names=names,
        rows=rows,
        case_ids=[c.case_id for c in case_list],
        references=references,
    )
