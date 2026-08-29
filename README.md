# opslab — operations analytics for BFSI back-office processes

A dependency-free Python toolkit for the questions that come up in life-and-pensions
and financial-services operations, and that general-purpose data science libraries
answer badly:

- **Where does the time actually go** in a case-management process, once you stop
  counting overnight gaps as work?
- **Did the process really change**, or is this week's number ordinary variation?
- **How long do cases take**, when a tenth of them are still open and the slow ones
  are exactly the ones you cannot see yet?
- **Is the Power BI model that reports all this** built on anything sound?

Four modules answer those, over one sample dataset that ships inside the package.

Everything is pure standard library — no numpy, scipy, pandas or lifelines. It
installs anywhere Python 3.9+ runs, including a locked-down corporate laptop.

---

## Try it

**[Read the sample report →](https://pr1317.github.io/opslab2/)**  Every chart on
that page — the discovered process map, the control charts, the survival curve,
the coefficient forest plot — was drawn by this package from the standard library
alone, and it is rebuilt from `main` on every push.

**[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pr1317/opslab2/blob/main/notebooks/try_opslab.ipynb)**
Runs the whole thing in a browser, nothing to install. About a minute.

**Locally**, in three lines:

```bash
pip install git+https://github.com/pr1317/opslab2
opslab try            # writes out/try/report.html, then open it
```

`opslab try` takes no arguments. It runs all four modules against the sample event
log, case table and Power BI model that ship in the wheel — nothing is downloaded,
nothing is simulated on the fly — and writes one self-contained HTML file with no
external references, so it opens from a `file://` URL on a machine with no network.

It also copies the sample CSVs next to the report, so the other subcommands have
something to point at:

```bash
opslab mine  --events out/try/sample_events.csv --out out/try
opslab spc   --cases  out/try/sample_cases.csv
opslab sla   --cases  out/try/sample_cases.csv --numeric complexity --binary priority=Urgent
opslab daxlint --sample --explain
```

Point it at your own extract with the same command:

```bash
opslab try --events my_events.csv --cases my_cases.csv --out report
```

---

## Why these four

Most analytics portfolios stop at a churn model on a public dataset. These are the
problems that are specific to *operations*, and each one is here because the
obvious tool gets it wrong:

| Module | The usual approach | Why it fails | What this does instead |
|---|---|---|---|
| `processmining` | A hand-drawn Visio process map | Documents what was designed, not what happens | Discovers the map from the event log, with rework loops and timing on the edges |
| `spc` | Traffic-light KPI dashboard vs. last month | Reacts to noise, misses real shifts | Control charts with Nelson run rules, so a signal means something |
| `sla` | Logistic regression on "did it breach" | Throws away every case still open — the slow ones | Right-censored survival models that use those cases as evidence |
| `daxlint` | Manual review, or none | Doesn't scale, misses systematic defects | 20 automated rules over the tabular model and its DAX |

---

## 1. Process mining (`opslab.processmining`)

Discovers a directly-follows graph from an event log, ranks variants, finds rework
and bottlenecks, and checks the process against declared controls.

```bash
opslab mine --events out/demo/events.csv --out out/demo
```

```
Discovered 20 activities and 36 transitions across 1483 cases

Top 5 of 85 variants (58.3% of cases):
   316  Receive Request -> Validate Details -> Index Documents -> Calculate Benefit -> ...

Bottlenecks by total working hours consumed:
  Receive Request     -> Validate Details      14551h total    11.0h mean  n=1325
  Index Documents     -> Calculate Benefit     10455h total    22.1h mean  n=473
  Request Information -> Await Customer         8908h total    19.8h mean  n=449

Conformance: 90.96% of 1483 cases fully conformant
  FOUR_EYES     76 violation(s) across 74 case(s)
      C00003: BEN-11 both performed 'Calculate Benefit' and checked it
```

Two decisions worth calling out:

**Bottlenecks rank by total time, not mean.** A two-hour wait on every case costs
the operation more than a two-week wait that happens twice, and only the first is
worth a project. Ranking by mean duration surfaces the rare, dramatic and
irrelevant.

**Conformance is declarative.** Rather than replaying tokens against a Petri net
few teams have and fewer keep current, the reference process is a set of rules —
required activities, forbidden transitions, ordering, occurrence caps, and
segregation of duties. Each names the cases that broke it and maps onto a control
an auditor would recognise. `SegregationOfDuties` is the one that earns its keep in
financial services: it catches cases where the person who calculated a benefit also
signed it off.

Process maps export to Graphviz DOT and to Mermaid, which renders directly in
GitHub and most wikis.

## 2. Statistical process control (`opslab.spc`)

Control charts (individuals, moving range, X-bar R, p, c, u), all eight Nelson run
rules, process capability, and a crossed ANOVA Gage R&R.

```bash
opslab spc --cases out/demo/cases.csv
```

```
Weekly SLA breach rate
p (proportion defective) (centre 0.379)
2026-W24                 |          :        * |      0.5070 <- NELSON_5,NELSON_6
2026-W25               |            :          *  |    0.5417 <- NELSON_2,NELSON_5,NELSON_6
2026-W26                |           :          * |     0.5357 <- NELSON_2,NELSON_5,NELSON_6
  ! Smallest subgroup expects fewer than 5 defectives; the normal approximation
    behind the limits is weak - treat single-point signals with caution.
```

The control limits move with each week's volume, and the sustained run from W24
is the injected backlog surge being caught by the run rules rather than by
someone noticing a red cell.

**Chart choice is not cosmetic.** A weekly breach rate is binomial, so it belongs on
a p-chart whose limits widen on low-volume weeks. Putting it on a chart with flat
limits manufactures a special cause every bank holiday. The p-chart also warns when
subgroups are too small for the normal approximation behind its limits.

**Capability separates Cp/Cpk from Pp/Ppk** — what the process could do if it held
still, versus what it actually delivered — and reports one-sided specifications
honestly. Turnaround time has an upper limit and no lower one, so `Cp` is undefined
and comes back as `n/a` rather than a flattering number invented from a mirror
limit. The report flags a drifting process and warns when the observed defect rate
diverges from the fitted normal, which for cycle times it usually does.

## 3. SLA survival analysis (`opslab.sla`)

Kaplan-Meier and Nelson-Aalen estimators, the log-rank test, and Cox
proportional-hazards regression with Breslow ties — all handling right-censored
cases.

```bash
opslab sla --cases out/demo/cases.csv \
  --numeric complexity --numeric backlog_index --numeric awaiting_third_party \
  --binary priority=Urgent --binary channel=Post --group priority
```

```
1483 cases, 1324 resolved, 159 still open at the extract (10.7% censored)

Median time to resolution
  closed cases only : 46.7 working hours
  Kaplan-Meier      : 52.8 working hours

Cox proportional hazards (Breslow ties)
  term                          coef exp(coef)        se         z p
  complexity                 -1.2720    0.2803    0.1385    -9.181 4.26e-20
  awaiting_third_party       -1.9687    0.1396    0.0818   -24.081 3.94e-128
  priority_urgent             0.5856    1.7960    0.0648     9.039 1.58e-19
  concordance index 0.7224 over 1025104 comparable pairs
```

**This is the module the rest of the repo is arranged around.** At any extract date
some cases are still open, and their duration is unknown — only bounded below.
Dropping them biases the average down, because the slow cases are precisely the ones
still running. Counting their current age as final biases it up. The 46.7 vs 52.8
gap above is that bias, on data where the true answer is known.

It gets worse than a biased average. Bucketing finished cases by arrival date can
show turnaround *improving* in the exact week the process got worse, because the
new slow cases have not finished yet and so are invisible. There is a test pinning
that behaviour down: `test_bucketing_resolved_cases_by_arrival_hides_the_surge`.

The Cox model uses the open cases as the evidence they are, and estimates effects on
the whole time distribution rather than at one arbitrary threshold. `breach_probability()`
turns a fitted model into a per-case risk score against any target.

## 4. Power BI model linter (`opslab.daxlint`)

Twenty rules over a tabular model and its DAX. Reads TMSL (`.bim`, `model.json`) and
TMDL (`.tmdl` files, or a PBIP `definition/` folder — what a model under source
control looks like now).

```bash
opslab daxlint --sample --explain
```

```
MOD007   error   Cases[CaseID]: identifier column defaults to being aggregated (summarizeBy=sum)
         -> Set summarizeBy to none. Otherwise dragging the column into a visual
            silently produces the sum of a set of IDs, which looks like a real
            number and is not.
MOD009   error   PensionsOperations: time intelligence is used but no table is marked as a date table
DAX004   warning Cases[Slow Cases]:1: FILTER() iterates the whole 'Cases' table
DAX002   warning Cases[Unqualified Total]:1: column [HandlingHours] is referenced without its table
```

Every rule states the concrete consequence of ignoring it, because a linter that
only says "not best practice" gets switched off. It exits non-zero on errors, so it
drops into CI over a PBIP repository; `--select`, `--ignore` and `--fail-on` control
what that means.

The DAX rules run on a real tokenizer, not regular expressions over source text — so
`FILTER` inside a comment or a string literal does not trigger anything. Because the
linter has the model as well as the expression, it can tell a measure reference from
a column reference and enforce the convention that matters: columns always qualified,
measures never.

Run `opslab daxlint --list-rules` for the full set.

---

## The dataset is verifiable

`opslab.simulate` generates a life-and-pensions back office: five case types,
rework loops, four-eyes violations, working-hours gaps, a dated backlog surge, and
cases still open at the extract.

Case durations are drawn from a Weibull proportional-hazards model whose
coefficients are published in `GROUND_TRUTH_BETA`. That makes the analysis
checkable rather than merely plausible — `test_cox_recovers_known_simulation_coefficients`
asserts every fitted coefficient lands within three standard errors of the value
that generated the data:

| term | truth | fitted | distance |
|---|---|---|---|
| `complexity` | −1.10 | −0.98 | 1.15 se |
| `backlog_index` | −0.85 | −0.92 | 0.55 se |
| `awaiting_third_party` | −1.90 | −1.87 | 0.50 se |
| `priority_urgent` | +0.55 | +0.56 | 0.18 se |
| `channel_post` | −0.45 | −0.42 | 0.60 se |

<sub>`simulate(SimulationConfig(n_cases=2500, seed=7))`, as used by the test.</sub>

The long-running cases are driven by an **observed** covariate
(`awaiting_third_party`, derived from whether the trace waits on a customer or an
external scheme) rather than a hidden long-runner class. That distinction is
deliberate: latent heterogeneity would bias the Cox coefficients toward zero and
quietly destroy the guarantee above.

## Everything runs on a working calendar

A case that arrives at 16:30 on Friday and closes at 09:30 on Monday consumed one
working hour, not sixty-five. `BusinessCalendar` (configurable hours, weekdays and
holidays) backs every duration in the toolkit, so control charts, bottleneck
rankings and survival models all measure the same clock the service credits are
measured against.

## Working with your own data

Nothing depends on the simulator. The modules read two CSVs:

**`events.csv`** — `case_id, activity, timestamp, resource`, plus any extra columns,
which are carried through as event attributes.

**`cases.csv`** — `case_id, arrived, duration_hours, resolved, sla_hours`, plus
attributes used as model covariates. `resolved` is `0` for a case still open, and
`duration_hours` is then the time observed so far.

```python
from opslab.calendar import BusinessCalendar
from opslab.eventlog import EventLog
from opslab.processmining import discover_dfg, to_svg

log = EventLog.from_csv("events.csv")
graph = discover_dfg(log, BusinessCalendar(start_hour=8, end_hour=18))
open("map.svg", "w").write(to_svg(graph, min_edge_frequency=10))
```

`to_dot` and `to_mermaid` render the same graph for Graphviz and for anything that
renders Markdown; `to_svg` draws it directly, so a process map needs no other tool
installed.

The sample that `opslab try` uses is reachable from Python too, which is the
quickest way to check your own loader against a known-good file:

```python
from opslab import data

data.events_path()   # the sample event log
data.cases_path()    # the sample case table, including cases still open
data.model_path()    # a .bim written to fail every lint rule at least once
```

## Development

```bash
pip install -e ".[dev]"
pytest -q          # 215 tests, ~5 seconds
opslab demo --out out/demo
```

Tests check against externally known values rather than only internal consistency:
normal and chi-square quantiles against published tables, control-chart constants
against the standard factors, Kaplan-Meier against a hand-computed curve, and the
Cox model's analytic gradient and Hessian against finite differences.

CI runs the suite, the end-to-end demo and the report build on Python 3.9,
3.11 and 3.12. A second workflow publishes the report to GitHub Pages.

## Licence

MIT.
