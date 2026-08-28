"""opslab - operations analytics toolkit for BFSI back-office processes.

Four independent, dependency-free modules that share one synthetic event log:

* :mod:`opslab.simulate`      - generator for realistic case-management event logs
* :mod:`opslab.processmining` - discovery, variant analysis and conformance checking
* :mod:`opslab.spc`           - control charts, Nelson rules and process capability
* :mod:`opslab.sla`           - right-censored SLA breach modelling (Kaplan-Meier, Cox)
* :mod:`opslab.daxlint`       - static analysis for Power BI tabular models and DAX
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
