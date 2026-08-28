"""Nelson run rules.

The rules are evaluated on standardised deviations from the centre line, so
they work unchanged on charts whose limits vary point to point (a p-chart with
unequal daily volumes, for instance).

Following the usual convention, a run rule flags the point that *completes*
the pattern rather than every point inside the window; use
:func:`expand_windows` when the whole run is wanted for root-cause work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["NelsonRule", "NELSON_RULES", "apply_nelson_rules", "expand_windows"]


@dataclass(frozen=True)
class NelsonRule:
    """One run rule: a window length and a predicate over standardised values."""

    code: str
    window: int
    description: str
    predicate: Callable[[Sequence[float]], bool]

    def __str__(self) -> str:  # pragma: no cover - display helper
        return "%s: %s" % (self.code, self.description)


def _same_side(values: Sequence[float]) -> bool:
    return all(v > 0 for v in values) or all(v < 0 for v in values)


def _rule1(window: Sequence[float]) -> bool:
    return abs(window[-1]) > 3.0


def _rule2(window: Sequence[float]) -> bool:
    return _same_side(window)


def _rule3(window: Sequence[float]) -> bool:
    increasing = all(window[i] < window[i + 1] for i in range(len(window) - 1))
    decreasing = all(window[i] > window[i + 1] for i in range(len(window) - 1))
    return increasing or decreasing


def _rule4(window: Sequence[float]) -> bool:
    directions = [window[i + 1] - window[i] for i in range(len(window) - 1)]
    if any(d == 0 for d in directions):
        return False
    return all(directions[i] * directions[i + 1] < 0 for i in range(len(directions) - 1))


def _rule5(window: Sequence[float]) -> bool:
    # Two of three beyond 2 sigma on the same side, the last point included.
    if abs(window[-1]) <= 2.0:
        return False
    beyond = [v for v in window if abs(v) > 2.0]
    return len(beyond) >= 2 and _same_side(beyond)


def _rule6(window: Sequence[float]) -> bool:
    # Four of five beyond 1 sigma on the same side, the last point included.
    if abs(window[-1]) <= 1.0:
        return False
    beyond = [v for v in window if abs(v) > 1.0]
    return len(beyond) >= 4 and _same_side(beyond)


def _rule7(window: Sequence[float]) -> bool:
    return all(abs(v) < 1.0 for v in window)


def _rule8(window: Sequence[float]) -> bool:
    if not all(abs(v) > 1.0 for v in window):
        return False
    return any(v > 0 for v in window) and any(v < 0 for v in window)


#: Nelson's eight tests for special causes, in their conventional order.
NELSON_RULES: Tuple[NelsonRule, ...] = (
    NelsonRule("NELSON_1", 1, "one point beyond 3 sigma", _rule1),
    NelsonRule("NELSON_2", 9, "nine points in a row on the same side of the centre", _rule2),
    NelsonRule("NELSON_3", 6, "six points in a row steadily increasing or decreasing", _rule3),
    NelsonRule("NELSON_4", 14, "fourteen points in a row alternating up and down", _rule4),
    NelsonRule("NELSON_5", 3, "two of three consecutive points beyond 2 sigma, same side", _rule5),
    NelsonRule("NELSON_6", 5, "four of five consecutive points beyond 1 sigma, same side", _rule6),
    NelsonRule("NELSON_7", 15, "fifteen points in a row within 1 sigma of the centre", _rule7),
    NelsonRule("NELSON_8", 8, "eight points in a row all beyond 1 sigma, straddling the centre", _rule8),
)

_RULES_BY_CODE: Dict[str, NelsonRule] = {rule.code: rule for rule in NELSON_RULES}


def apply_nelson_rules(
    standardised: Sequence[float],
    rules: Optional[Sequence[NelsonRule]] = None,
) -> List[List[str]]:
    """Evaluate run rules over standardised values.

    Returns one list of triggered rule codes per input point.
    """
    active = tuple(rules) if rules is not None else NELSON_RULES
    flags: List[List[str]] = [[] for _ in standardised]
    for rule in active:
        if rule.window > len(standardised):
            continue
        for end in range(rule.window - 1, len(standardised)):
            window = standardised[end - rule.window + 1 : end + 1]
            if rule.predicate(window):
                flags[end].append(rule.code)
    return flags


def expand_windows(flags: Sequence[Sequence[str]]) -> List[List[str]]:
    """Widen each flag back across the window that produced it.

    Useful when presenting a chart to a process owner, who generally wants to
    see the whole run highlighted rather than only its final point.
    """
    widened: List[List[str]] = [[] for _ in flags]
    for end, codes in enumerate(flags):
        for code in codes:
            rule = _RULES_BY_CODE.get(code)
            if rule is None:
                continue
            for index in range(max(0, end - rule.window + 1), end + 1):
                if code not in widened[index]:
                    widened[index].append(code)
    return widened
