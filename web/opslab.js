/* opslab — the fitted models, evaluated in the browser.
 *
 * Everything here is arithmetic over the numbers `opslab export` wrote. No
 * fitting happens client side: the coefficients and the Breslow baseline hazard
 * come from the Python fit, and this file only evaluates them. A parity test in
 * the repository checks these functions against the Python implementation.
 */
(function (global) {
  "use strict";

  /* x'beta for one covariate vector. */
  function linearPredictor(coefficients, covariates) {
    var total = 0;
    for (var i = 0; i < coefficients.length; i++) {
      total += coefficients[i] * covariates[i];
    }
    return total;
  }

  /* Baseline cumulative hazard at `time`: a right-continuous step function, so
   * the value is the one attached to the latest event time at or before it. */
  function cumulativeHazardAt(baseline, time) {
    var times = baseline.times;
    if (!times.length || time < times[0]) return 0;
    var low = 0, high = times.length - 1, best = -1;
    while (low <= high) {
      var mid = (low + high) >> 1;
      if (times[mid] <= time) { best = mid; low = mid + 1; } else { high = mid - 1; }
    }
    return best < 0 ? 0 : baseline.cumulativeHazard[best];
  }

  /* P(the case is still open after `time`). */
  function survival(cox, covariates, time) {
    var baseline = cumulativeHazardAt(cox.baseline, time);
    return Math.exp(-baseline * Math.exp(linearPredictor(cox.coefficients, covariates)));
  }

  /* The event modelled is resolution, so still being open at the target is
   * exactly the probability of breaching it. */
  function breachProbability(cox, covariates, slaHours) {
    if (!(slaHours > 0)) throw new Error("slaHours must be positive");
    return survival(cox, covariates, slaHours);
  }

  /* The whole survival curve for one covariate vector, sampled for drawing. */
  function survivalCurve(cox, covariates, horizon, steps) {
    steps = steps || 160;
    var risk = Math.exp(linearPredictor(cox.coefficients, covariates));
    var points = [];
    for (var i = 0; i <= steps; i++) {
      var t = horizon * i / steps;
      points.push([t, Math.exp(-cumulativeHazardAt(cox.baseline, t) * risk)]);
    }
    return points;
  }

  /* Each covariate's contribution to the log hazard, largest effect first.
   * A positive contribution raises the resolution hazard, i.e. clears the case
   * sooner, which is why the sign is flipped when this is read as breach risk. */
  function contributions(cox, covariates) {
    var rows = [];
    for (var i = 0; i < cox.names.length; i++) {
      rows.push({
        name: cox.names[i],
        value: covariates[i],
        coefficient: cox.coefficients[i],
        contribution: cox.coefficients[i] * covariates[i]
      });
    }
    rows.sort(function (a, b) {
      return Math.abs(b.contribution) - Math.abs(a.contribution);
    });
    return rows;
  }

  /* Time at which this case's survival first drops to or below 1 - p, or null
   * when the follow-up never reaches it — the honest answer for a median the
   * data does not cover. */
  function quantile(cox, covariates, probability) {
    var risk = Math.exp(linearPredictor(cox.coefficients, covariates));
    var threshold = 1 - probability;
    var times = cox.baseline.times;
    for (var i = 0; i < times.length; i++) {
      if (Math.exp(-cox.baseline.cumulativeHazard[i] * risk) <= threshold) {
        return times[i];
      }
    }
    return null;
  }

  global.opslab = {
    linearPredictor: linearPredictor,
    cumulativeHazardAt: cumulativeHazardAt,
    survival: survival,
    breachProbability: breachProbability,
    survivalCurve: survivalCurve,
    contributions: contributions,
    quantile: quantile
  };
})(typeof window !== "undefined" ? window : globalThis);

if (typeof module !== "undefined" && module.exports) {
  module.exports = (typeof window !== "undefined" ? window : globalThis).opslab;
}
