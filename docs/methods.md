# Methods and references

Implementation notes for the statistical machinery, and the choices that are
judgement calls rather than standard practice.

## Numeric kernel (`opslab.numeric`)

| Routine | Method |
|---|---|
| `norm_cdf`, `norm_sf` | `math.erf` / `math.erfc`. The survival function uses `erfc` directly so the far tail keeps its precision. |
| `norm_ppf` | Acklam's rational approximation, refined by one Halley step against the exact CDF, giving full double precision on `(0, 1)`. |
| `chi2_sf` | Regularised incomplete gamma: series expansion below the mean, continued fraction above (Numerical Recipes §6.2). |
| `solve`, `inverse` | Gaussian elimination with partial pivoting. |

Sizes here are small — a Cox model on operational data has a handful of covariates —
so dense elimination is the right trade against pulling in a linear algebra
dependency.

## Control charts (`opslab.spc.charts`)

Limits use the classical control-chart constants (`d2`, `d3`, `A2`, `D3`, `D4`)
tabulated for subgroups of 2 to 10.

The individuals chart estimates sigma from the **average moving range**, not the
sample standard deviation. This is deliberate and it matters: the moving range uses
only variation between consecutive observations, so a sustained shift shows up as a
signal. The overall standard deviation would absorb that shift into wider limits and
report the process as stable.

## Nelson run rules (`opslab.spc.rules`)

All eight tests, evaluated on standardised deviations from the centre line so they
work unchanged on charts whose limits vary point to point.

Following the usual convention, a run rule flags the point that **completes** the
pattern rather than every point in the window. `expand_windows()` widens the flags
back across the run, which is what a process owner wants when investigating.

Rules 5 and 6 additionally require the completing point itself to be beyond the
relevant sigma boundary. Without that, the rule fires on a window whose qualifying
points have already passed, and the flag lands on an innocent observation.

## Capability (`opslab.spc.capability`)

- `Cp`, `Cpk` use within-subgroup sigma (moving range) — capability.
- `Pp`, `Ppk` use overall sigma — performance.
- `Cpm` (Taguchi) includes the deviation from target.
- Sigma level is the benchmark Z plus the conventional 1.5 long-term shift.

Two guards worth knowing about:

**One-sided specifications.** `Cp` and `Pp` are undefined with only one limit and
return `None`. Turnaround time is the common case.

**The tail.** The benchmark Z is computed as `-norm_ppf(p)` rather than
`norm_ppf(1 - p)`; for the small defect rates a capable process produces, `1 - p`
rounds to exactly 1.0 in double precision and the quantile function is undefined
there. The result is capped at 6 sigma before the shift, because beyond that the
figure extrapolates a fitted normal tail far outside the range of any real sample.

## Gage R&R (`opslab.spc.msa`)

Crossed ANOVA with interaction, on a balanced design. Variance components:

```
repeatability    = MS_error
interaction      = (MS_interaction - MS_error) / replicates
reproducibility  = (MS_operator - MS_interaction) / (parts * replicates)
part-to-part     = (MS_part - MS_interaction) / (operators * replicates)
```

Negative components — which the ANOVA can produce when a true component is near zero
— are truncated at zero in the conventional way and noted explicitly in the report
rather than silently. Acceptance follows the AIAG bands on %study variation (<10%
acceptable, <30% marginal), with the number of distinct categories as
`1.41 * sd_part / sd_gage`.

## Survival estimation (`opslab.sla.survival`)

The event throughout is **case resolution**, so a survival curve reads as "share of
cases still open after *t* working hours".

Kaplan-Meier confidence bands use the log-log transform rather than the plain
Greenwood interval, because the linear interval leaves `[0, 1]` once the curve
approaches either end.

`quantile()` returns `None` when the curve never falls far enough, which is the
correct answer for a median the follow-up did not reach — not an extrapolated guess.
`restricted_mean(horizon)` is the estimator to quote instead, and it is well defined
for any follow-up length provided the horizon is stated.

The log-rank test uses the hypergeometric variance, compared against chi-square on
1 degree of freedom.

## Cox proportional hazards (`opslab.sla.coxph`)

Newton-Raphson on the Breslow partial likelihood, with step halving whenever a full
step would decrease the log-likelihood.

The partial likelihood, gradient and Hessian are accumulated in a single pass over
subjects sorted by **descending** time. Because the risk set at time *t* is everyone
with time ≥ *t*, walking backwards lets the risk-set sums S₀, S₁ and S₂ be
accumulated incrementally. That makes the fit O(n·p²) rather than the O(n²·p²) a
naive implementation costs, which is what makes pure Python practical here — 2,500
cases fit in under 0.2 seconds.

Standard errors come from the inverse observed information; p-values are two-sided
Wald. The baseline cumulative hazard is Breslow's estimator, from which
`survival()` and `breach_probability()` follow.

The analytic derivatives are checked against finite differences in
`test_cox_analytic_gradient_matches_finite_differences` and its Hessian counterpart.

`concordance_index` is Harrell's C. A pair is comparable only when the case that
resolved first is *known* to have done so, which excludes pairs where the earlier
observation was censored — the doubt an ordinary AUC on "breached / did not breach"
cannot express.

## Ties, and why Breslow

Breslow's approximation is used for tied event times. Efron's is more accurate when
ties are heavy, but working-hour durations are near-continuous and ties are rare;
Breslow keeps the accumulation in one pass. Should the toolkit ever need to handle
coarse (say, whole-day) durations, Efron would be the change to make.

## Known limitations

- The proportional-hazards assumption is not tested. A Schoenfeld residual check is
  the obvious next addition.
- No time-varying covariates: each case contributes one row.
- `concordance_index` is O(n²) and gets slow beyond roughly 20,000 cases.
- Gage R&R requires a balanced crossed design; nested and unbalanced studies are
  rejected rather than approximated.
