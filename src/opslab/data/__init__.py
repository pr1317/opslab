"""Sample data shipped inside the package.

``opslab try`` runs against these three files so that the toolkit has something
real to work on before anyone has pointed it at their own extract. The event log
and case table are the committed output of::

    opslab simulate --cases 1500 --seed 20260828

They are stored rather than regenerated on demand for two reasons. A committed
CSV can be read on GitHub without installing anything, which is most of the
point of a sample; and it pins the numbers, so the figures quoted in the README,
the published report and a fresh checkout are the same figures.

The paths are resolved from ``__file__`` rather than through
``importlib.resources``, because every caller here wants a filesystem path to
hand to ``EventLog.from_csv`` or to print. That holds for any install pip
produces; it would not hold for a zipimported package.
"""

from __future__ import annotations

import os

__all__ = ["cases_path", "events_path", "model_path", "paths"]

_HERE = os.path.dirname(os.path.abspath(__file__))


def events_path() -> str:
    """Path to the sample event log: one row per activity instance."""
    return os.path.join(_HERE, "sample_events.csv")


def cases_path() -> str:
    """Path to the sample case table: one row per case, with censoring flags."""
    return os.path.join(_HERE, "sample_cases.csv")


def model_path() -> str:
    """Path to the sample Power BI model, a ``.bim`` written to fail the rules."""
    return os.path.join(_HERE, "pensions_ops.bim")


def paths() -> dict:
    """All three sample paths, keyed by the argument name each one feeds."""
    return {"events": events_path(), "cases": cases_path(), "model": model_path()}
