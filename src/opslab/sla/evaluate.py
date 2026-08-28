"""Evaluation metrics for censored risk models."""

from __future__ import annotations

from typing import Sequence, Tuple

__all__ = ["concordance_index"]


def concordance_index(
    durations: Sequence[float],
    events: Sequence[int],
    risk_scores: Sequence[float],
) -> Tuple[float, int]:
    """Harrell's concordance index for a censored outcome.

    A pair of cases is *comparable* when the one that resolved first is known
    to have done so - which rules out pairs where the earlier observation was
    censored, since that case might really have run longer.  Restricting to
    comparable pairs is what makes the index usable on data with open cases;
    an ordinary AUC on "breached / did not breach" cannot express the doubt.

    A higher risk score is expected to mean faster resolution, matching the
    ``exp(x'beta)`` convention of :class:`~opslab.sla.coxph.CoxPHModel`.

    Returns ``(c_index, comparable_pairs)``.  With no comparable pairs the
    index is undefined and 0.5 is returned alongside a zero count, so the
    caller can tell an uninformative model from an unmeasurable one.
    """
    if not (len(durations) == len(events) == len(risk_scores)):
        raise ValueError("durations, events and risk_scores must be the same length")

    concordant = 0.0
    comparable = 0
    n = len(durations)
    for i in range(n):
        if not events[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # i resolved first, so i is genuinely the faster of the pair.
            if durations[j] > durations[i] or (
                durations[j] == durations[i] and not events[j]
            ):
                comparable += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1.0
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5

    if comparable == 0:
        return 0.5, 0
    return concordant / comparable, comparable
