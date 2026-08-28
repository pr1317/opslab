# Using your own extract

Nothing in the toolkit depends on the simulator. Two CSVs drive everything.

## `events.csv` — what happened

One row per completed activity.

| Column | Required | Notes |
|---|---|---|
| `case_id` | yes | Groups events into cases |
| `activity` | yes | The step that completed |
| `timestamp` | yes | `YYYY-MM-DD HH:MM:SS` (also accepts `T` separator, minute precision, or date only) |
| `resource` | no | Who did it — needed for handover analysis and segregation-of-duties checks |
| *anything else* | no | Carried through as event attributes |

```csv
case_id,activity,timestamp,resource,case_type
C00001,Receive Request,2026-01-05 09:14:00,BEN-03,Retirement Quote
C00001,Validate Details,2026-01-05 11:42:00,BEN-07,Retirement Quote
```

## `cases.csv` — how each case ended

One row per case. This is the table the survival models read.

| Column | Required | Notes |
|---|---|---|
| `case_id` | yes | Joins to the event log |
| `arrived` | yes | When the clock started |
| `duration_hours` | yes | Working hours consumed — **time observed so far** for an open case |
| `resolved` | yes | `1` if the case finished, `0` if still open at the extract |
| `sla_hours` | yes | The target this case was measured against |
| *anything else* | no | Available as model covariates |

The `resolved` flag is the important one. Setting it to `1` for every row — or
excluding open cases from the extract — reintroduces exactly the bias the module
exists to remove.

## Extracting from a case management system

A typical SQL shape, assuming an audit or workflow history table:

```sql
-- events.csv
SELECT case_ref            AS case_id,
       activity_name       AS activity,
       completed_at        AS timestamp,
       performed_by        AS resource,
       case_type
FROM   workflow_history
WHERE  completed_at >= :window_start
  AND  completed_at <  :extract_moment;

-- cases.csv
SELECT c.case_ref                                AS case_id,
       c.opened_at                               AS arrived,
       c.working_hours_elapsed                   AS duration_hours,
       CASE WHEN c.closed_at IS NOT NULL THEN 1 ELSE 0 END AS resolved,
       c.sla_target_hours                        AS sla_hours,
       c.priority, c.channel
FROM   cases c
WHERE  c.opened_at >= :window_start;
```

Note the `cases` query has **no** `closed_at IS NOT NULL` filter. That omission is
the point: open cases carry the information about how slow the process is becoming.

If the source system stores only wall-clock timestamps, compute working hours with
the calendar:

```python
from opslab.calendar import BusinessCalendar

calendar = BusinessCalendar(start_hour=8.0, end_hour=18.0, holidays=[...])
hours = calendar.working_hours_between(opened_at, closed_at or extract_moment)
```

## Defining your reference process

Conformance rules are declared in code, so they live in version control next to the
process documentation:

```python
from opslab.processmining import (
    MaxOccurrences, Ordering, RequiredActivities,
    SegregationOfDuties, StartActivities, check_conformance,
)

rules = [
    StartActivities(("Receive Request",)),
    RequiredActivities(("Validate Details", "Peer Check")),
    Ordering("Peer Check", "Release Payment"),          # never pay before checking
    MaxOccurrences("Rework Calculation", 1),
    SegregationOfDuties(("Calculate Benefit",), "Peer Check"),
]

report = check_conformance(log, rules)
print(report.to_text())
for case_id in report.cases_for("FOUR_EYES"):
    ...  # feed the exceptions into a QA sample
```

Scope rules to a case type where the processes genuinely differ — running a rule set
built for benefit calculations against complaints reports violations that are really
just a mis-scoped rule.

## Linting a Power BI model

For a PBIP project, point the linter at the model definition folder:

```bash
opslab daxlint "MyReport.SemanticModel/definition" --fail-on error
```

For a `.bim` export from Analysis Services or Tabular Editor:

```bash
opslab daxlint model.bim --explain
```

In CI, `--fail-on error` blocks a merge on the rules that produce wrong numbers
(aggregated identifier columns, time intelligence without a marked date table) while
letting documentation findings through. Tighten to `--fail-on warning` once the
model is clean.
