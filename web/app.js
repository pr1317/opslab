/* opslab demo — wiring the exported models to the page.
 *
 * Drawing is done by hand rather than with a chart library, for the same reason
 * the Python has no dependencies: the whole point is that this runs anywhere,
 * and a control chart is a polyline.
 */
(function () {
  "use strict";

  var DATA = window.OPSLAB_DATA;
  var MODEL = window.opslab;
  if (!DATA || !MODEL) return;

  var COX = DATA.sla.cox;
  var NAMES = COX.names;

  /* Covariate presentation. The fitted names are terse because they are column
   * names; a reader needs the units and the direction. */
  var LABELS = {
    complexity: {
      label: "Case complexity",
      help: "0 = a straightforward quote, 1 = a contested death claim"
    },
    backlog_index: {
      label: "Backlog pressure",
      help: "How loaded the team was when the case arrived"
    },
    awaiting_third_party: {
      label: "Waiting on a third party",
      help: "The case sat with the customer or a technical team"
    },
    priority_urgent: { label: "Urgent priority", help: "" },
    channel_post: { label: "Arrived by post", help: "Rather than online or by phone" }
  };

  var PRESETS = [
    { name: "A typical case", values: { complexity: 0.42, backlog_index: 0.50,
      awaiting_third_party: 0, priority_urgent: 0, channel_post: 0 }, sla: 64 },
    { name: "Complex, chased, on paper", values: { complexity: 0.88, backlog_index: 0.86,
      awaiting_third_party: 1, priority_urgent: 0, channel_post: 1 }, sla: 96 },
    { name: "Urgent and clean", values: { complexity: 0.15, backlog_index: 0.30,
      awaiting_third_party: 0, priority_urgent: 1, channel_post: 0 }, sla: 48 }
  ];

  var state = { covariates: {}, sla: 64 };

  // ---------------------------------------------------------------- helpers
  function el(id) { return document.getElementById(id); }
  function percent(value, places) { return (100 * value).toFixed(places === undefined ? 1 : places) + "%"; }
  function thousands(value) { return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }

  function svg(width, height, body, label) {
    return '<svg viewBox="0 0 ' + width + " " + height + '" width="100%" ' +
      'preserveAspectRatio="xMidYMid meet" role="img" aria-label="' + label +
      '" style="min-width:480px;overflow:visible">' + body + "</svg>";
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"]/g, function (character) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[character];
    });
  }

  /* Maps data coordinates onto a fixed viewBox. */
  function frame(width, height, xRange, yRange, margins) {
    var m = margins || { top: 16, right: 18, bottom: 40, left: 56 };
    var plotWidth = width - m.left - m.right;
    var plotHeight = height - m.top - m.bottom;
    return {
      width: width, height: height, m: m,
      x: function (value) {
        return m.left + (value - xRange[0]) / (xRange[1] - xRange[0] || 1) * plotWidth;
      },
      y: function (value) {
        return m.top + (1 - (value - yRange[0]) / (yRange[1] - yRange[0] || 1)) * plotHeight;
      }
    };
  }

  function points(pairs) {
    return pairs.map(function (pair) {
      return pair[0].toFixed(2) + "," + pair[1].toFixed(2);
    }).join(" ");
  }

  function ticks(low, high, count) {
    var step = (high - low) / (count - 1), out = [];
    for (var i = 0; i < count; i++) out.push(low + step * i);
    return out;
  }

  function axes(f, yTicks, xTicks, formatY) {
    var parts = [];
    yTicks.forEach(function (value) {
      var y = f.y(value);
      parts.push('<line class="grid" x1="' + f.m.left + '" y1="' + y.toFixed(2) +
        '" x2="' + (f.width - f.m.right) + '" y2="' + y.toFixed(2) + '"/>');
      parts.push('<text class="tick" x="' + (f.m.left - 8) + '" y="' + y.toFixed(2) +
        '" text-anchor="end" dy="3.5">' + formatY(value) + "</text>");
    });
    xTicks.forEach(function (tick) {
      parts.push('<text class="tick" x="' + f.x(tick.at).toFixed(2) + '" y="' +
        (f.height - f.m.bottom + 18) + '" text-anchor="middle">' + escapeHtml(tick.label) + "</text>");
    });
    parts.push('<line class="axis" x1="' + f.m.left + '" y1="' + (f.height - f.m.bottom) +
      '" x2="' + (f.width - f.m.right) + '" y2="' + (f.height - f.m.bottom) + '"/>');
    return parts.join("");
  }

  // ------------------------------------------------------------ 0. summary
  function renderSummary() {
    var s = DATA.summary, sla = DATA.sla;
    var tiles = [
      [thousands(s.cases), "cases", thousands(s.events) + " events"],
      [percent(sla.censoringRate), "still open", sla.open + " at the extract date"],
      [sla.medianClosedOnly + "h", "median, closed only", "the biased figure"],
      [sla.medianKaplanMeier + "h", "median, Kaplan-Meier", "the open cases counted too"]
    ];
    el("summary").innerHTML = tiles.map(function (tile) {
      return '<div class="stat"><b>' + tile[0] + "</b><span>" + tile[1] +
        "</span><em>" + tile[2] + "</em></div>";
    }).join("");
  }

  // --------------------------------------------------------- 1. process map
  function renderMap() {
    var thresholds = DATA.mining.thresholds;
    var slider = el("threshold");
    slider.max = String(thresholds.length - 1);
    function draw() {
      var threshold = thresholds[Number(slider.value)];
      el("thresholdValue").textContent = threshold;
      var markup = DATA.mining.maps[String(threshold)];
      el("map").innerHTML = markup;
      var shown = (markup.match(/class="mp-edge/g) || []).length;
      el("mapNote").innerHTML = "Showing <b>" + shown + "</b> of " +
        DATA.mining.transitions + " transitions between " + DATA.mining.activities +
        " activities. The log holds <b>" + DATA.mining.variants +
        "</b> distinct end-to-end paths; the six most common cover " +
        percent(DATA.mining.topVariants.slice(0, 6).reduce(function (sum, v) {
          return sum + v.share;
        }, 0)) + " of cases.";
    }
    slider.addEventListener("input", draw);
    draw();
  }

  // -------------------------------------------------------- 2. control chart
  function renderChart(seriesName) {
    var chart = DATA.spc[seriesName];
    var isRate = seriesName === "breachRate";
    var pts = chart.points;

    var low = Math.min.apply(null, pts.map(function (p) { return Math.min(p.lower, p.value); }));
    var high = Math.max.apply(null, pts.map(function (p) { return Math.max(p.upper, p.value); }));
    var pad = (high - low) * 0.12 || 1;
    if (isRate) { low = Math.max(0, low - pad); high = Math.min(1, high + pad); }
    else { low -= pad; high += pad; }

    var f = frame(760, 280, [0, pts.length - 1], [low, high]);
    var formatY = isRate
      ? function (v) { return (100 * v).toFixed(0) + "%"; }
      : function (v) { return v.toFixed(0); };

    /* Every label carries the same year prefix, which is most of its width. */
    var labels = pts.map(function (p) { return p.label.replace(/^\d{4}-/, ""); });
    var stride = Math.max(1, Math.floor(pts.length / 9));
    var xTicks = [];
    for (var i = 0; i < pts.length; i += stride) xTicks.push({ at: i, label: labels[i] });

    var body = [axes(f, ticks(low, high, 5), xTicks, formatY)];
    body.push('<polyline class="limit" points="' + points(pts.map(function (p, i) {
      return [f.x(i), f.y(p.upper)]; })) + '"/>');
    body.push('<polyline class="limit" points="' + points(pts.map(function (p, i) {
      return [f.x(i), f.y(p.lower)]; })) + '"/>');
    body.push('<polyline class="center" points="' + points(pts.map(function (p, i) {
      return [f.x(i), f.y(p.center)]; })) + '"/>');
    body.push('<polyline class="series" points="' + points(pts.map(function (p, i) {
      return [f.x(i), f.y(p.value)]; })) + '"/>');

    pts.forEach(function (p, i) {
      var signal = p.violations.length > 0;
      var shown = isRate ? percent(p.value) : p.value.toFixed(1) + "h";
      var tip = p.label + ": " + shown + (signal ? " — " + p.violations.join(", ") : "");
      body.push('<circle class="pt' + (signal ? " sig" : "") + '" cx="' + f.x(i).toFixed(2) +
        '" cy="' + f.y(p.value).toFixed(2) + '" r="' + (signal ? "4.2" : "2.8") +
        '"><title>' + escapeHtml(tip) + "</title></circle>");
    });

    el("chart").innerHTML = svg(f.width, f.height, body.join(""),
      isRate ? "Weekly SLA breach rate control chart" : "Weekly mean handling hours control chart");

    var signals = pts.filter(function (p) { return p.violations.length; });
    var earliest = signals.length ? signals[0].label : null;
    el("chartNote").innerHTML = signals.length
      ? "<b>" + signals.length + "</b> of " + pts.length +
        " weeks broke a rule, the first in <b>" + earliest +
        "</b>. A threshold dashboard would have flagged the worst week and then cried " +
        "wolf every week after; rules 2, 5 and 6 catch a sustained shift that never " +
        "breaks a limit, which is the failure a threshold misses entirely."
      : "Both charts are in statistical control: the variation is the process behaving " +
        "normally, not something to investigate.";
  }

  function bindChartTabs() {
    var tabs = el("chartTabs");
    tabs.addEventListener("click", function (event) {
      var button = event.target.closest("button");
      if (!button) return;
      Array.prototype.forEach.call(tabs.querySelectorAll("button"), function (other) {
        other.setAttribute("aria-pressed", String(other === button));
      });
      renderChart(button.dataset.series);
    });
    renderChart("breachRate");
  }

  // ---------------------------------------------------- 3. breach calculator
  function buildControls() {
    var wrap = el("controls");
    var html = COX.covariates.map(function (covariate) {
      var meta = LABELS[covariate.name] || { label: covariate.name, help: "" };
      if (covariate.binary) return "";
      return '<div class="field"><label for="cv-' + covariate.name + '">' +
        escapeHtml(meta.label) + '<span class="value" id="val-' + covariate.name + '"></span>' +
        "</label>" +
        '<input type="range" id="cv-' + covariate.name + '" min="' + covariate.min +
        '" max="' + covariate.max + '" step="0.01" value="' + covariate.mean + '">' +
        (meta.help ? '<div style="font-size:11.5px;color:var(--muted);margin-top:3px">' +
          escapeHtml(meta.help) + "</div>" : "") +
        "</div>";
    }).join("");

    var switches = COX.covariates.filter(function (c) { return c.binary; })
      .map(function (covariate) {
        var meta = LABELS[covariate.name] || { label: covariate.name };
        return '<label class="switch" data-for="' + covariate.name + '">' +
          '<input type="checkbox" id="cv-' + covariate.name + '">' +
          "<span>" + escapeHtml(meta.label) + "</span></label>";
      }).join("");

    var targets = DATA.sla.slaTargets.map(function (hours) {
      return '<option value="' + hours + '"' + (hours === 64 ? " selected" : "") + ">" +
        hours + " working hours</option>";
    }).join("");

    wrap.innerHTML = html +
      '<div class="field"><label>Case attributes</label><div class="switches">' +
      switches + "</div></div>" +
      '<div class="field"><label for="slaTarget">SLA target</label>' +
      '<select id="slaTarget">' + targets + "</select></div>";

    el("presets").innerHTML = PRESETS.map(function (preset, index) {
      return '<button data-preset="' + index + '">' + escapeHtml(preset.name) + "</button>";
    }).join("");

    wrap.addEventListener("input", readControls);
    wrap.addEventListener("change", readControls);
    el("presets").addEventListener("click", function (event) {
      var button = event.target.closest("button");
      if (button) applyPreset(PRESETS[Number(button.dataset.preset)]);
    });
  }

  function applyPreset(preset) {
    COX.covariates.forEach(function (covariate) {
      var input = el("cv-" + covariate.name);
      var value = preset.values[covariate.name];
      if (covariate.binary) input.checked = Boolean(value);
      else input.value = String(value);
    });
    el("slaTarget").value = String(preset.sla);
    readControls();
  }

  function readControls() {
    COX.covariates.forEach(function (covariate) {
      var input = el("cv-" + covariate.name);
      if (covariate.binary) {
        state.covariates[covariate.name] = input.checked ? 1 : 0;
        input.closest(".switch").classList.toggle("on", input.checked);
      } else {
        var value = Number(input.value);
        state.covariates[covariate.name] = value;
        el("val-" + covariate.name).textContent = value.toFixed(2);
      }
    });
    state.sla = Number(el("slaTarget").value);
    renderRisk();
  }

  function vector() {
    return NAMES.map(function (name) { return state.covariates[name] || 0; });
  }

  function renderRisk() {
    var x = vector();
    var probability = MODEL.breachProbability(COX, x, state.sla);
    el("risk").textContent = percent(probability);
    el("riskBar").style.width = (100 * probability).toFixed(1) + "%";
    el("riskBar").style.background = probability > 0.5 ? "var(--danger)"
      : probability > 0.25 ? "var(--warn)" : "var(--accent)";
    var median = MODEL.quantile(COX, x, 0.5);
    el("verdict").innerHTML = "chance this case is still open after <b>" + state.sla +
      "</b> working hours" + (median === null
        ? "" : " &middot; half of cases like it clear within <b>" + median.toFixed(0) + "h</b>");

    renderCurve(x);
    renderDrivers(x);
  }

  function renderCurve(x) {
    var km = DATA.sla.km;
    var horizon = Math.max(state.sla * 3, 240);
    var f = frame(560, 250, [0, horizon], [0, 1], { top: 14, right: 16, bottom: 38, left: 46 });

    var caseCurve = MODEL.survivalCurve(COX, x, horizon, 180).map(function (pair) {
      return [f.x(pair[0]), f.y(pair[1])];
    });

    var overall = [];
    for (var i = 0; i < km.times.length; i++) {
      if (km.times[i] > horizon) break;
      overall.push([f.x(km.times[i]), f.y(km.survival[i])]);
    }

    var body = [axes(f,
      [0, 0.25, 0.5, 0.75, 1],
      ticks(0, horizon, 5).map(function (t) { return { at: t, label: t.toFixed(0) + "h" }; }),
      function (v) { return (100 * v).toFixed(0) + "%"; })];
    if (overall.length) body.push('<polyline class="ghost" points="' + points(overall) + '"/>');
    body.push('<polyline class="series" points="' + points(caseCurve) + '"/>');

    var targetX = f.x(state.sla);
    var probability = MODEL.breachProbability(COX, x, state.sla);
    body.push('<line class="ref" x1="' + targetX.toFixed(2) + '" y1="' + f.m.top +
      '" x2="' + targetX.toFixed(2) + '" y2="' + (f.height - f.m.bottom) + '"/>');
    body.push('<circle class="pt sig" cx="' + targetX.toFixed(2) + '" cy="' +
      f.y(probability).toFixed(2) + '" r="4.5"/>');
    body.push('<text class="tick" x="' + (targetX + 7).toFixed(2) + '" y="' +
      (f.y(probability) - 8).toFixed(2) + '" style="fill:var(--warn)">' +
      percent(probability) + " still open</text>");

    el("curve").innerHTML = svg(f.width, f.height, body.join(""),
      "Survival curve for the selected case");
  }

  function renderDrivers(x) {
    var rows = MODEL.contributions(COX, x).filter(function (row) {
      return Math.abs(row.contribution) > 1e-9;
    });
    el("drivers").innerHTML = rows.length ? rows.map(function (row) {
      var meta = LABELS[row.name] || { label: row.name };
      /* A positive contribution raises the resolution hazard, so the case
       * clears sooner and the breach risk falls. */
      var raisesRisk = row.contribution < 0;
      return "<li><span class='name'>" + escapeHtml(meta.label) + "</span>" +
        "<span class='chip " + (raisesRisk ? "up" : "down") + "'>" +
        (raisesRisk ? "+" : "−") + " risk " +
        Math.abs(row.contribution).toFixed(2) + "</span></li>";
    }).join("") : "<li><span class='name'>Every covariate is at zero.</span></li>";
  }

  // ---------------------------------------------------------------- 4. lint
  function renderLint(severity) {
    var findings = DATA.daxlint.findings.filter(function (finding) {
      return severity === "all" || finding.severity === severity;
    });
    el("lintBody").innerHTML = findings.slice(0, 30).map(function (finding) {
      return "<tr><td><code>" + escapeHtml(finding.code) + "</code></td>" +
        '<td><span class="sev ' + escapeHtml(finding.severity) + '">' +
        escapeHtml(finding.severity) + "</span></td>" +
        "<td><code>" + escapeHtml(finding.object) +
        (finding.line ? ":" + finding.line : "") + "</code></td>" +
        "<td>" + escapeHtml(finding.message) + "</td></tr>";
    }).join("");
    var extra = findings.length > 30 ? " Showing the first 30." : "";
    el("lintNote").innerHTML = "<b>" + findings.length + "</b> finding" +
      (findings.length === 1 ? "" : "s") + " over " + DATA.daxlint.measures +
      " measures, " + DATA.daxlint.tables + " tables and " +
      DATA.daxlint.relationships + " relationships, from " + DATA.daxlint.ruleCount +
      " rules." + extra + " The linter exits non-zero on an error, so it drops " +
      "straight into CI.";
  }

  function bindLintTabs() {
    var counts = { error: 0, warning: 0, info: 0 };
    DATA.daxlint.findings.forEach(function (finding) { counts[finding.severity]++; });
    var options = [["all", "All " + DATA.daxlint.findings.length]]
      .concat(["error", "warning", "info"].filter(function (severity) {
        return counts[severity] > 0;
      }).map(function (severity) {
        return [severity, severity + " " + counts[severity]];
      }));

    var tabs = el("lintTabs");
    tabs.innerHTML = options.map(function (option, index) {
      return '<button data-severity="' + option[0] + '" aria-pressed="' +
        (index === 0) + '">' + escapeHtml(option[1]) + "</button>";
    }).join("");
    tabs.addEventListener("click", function (event) {
      var button = event.target.closest("button");
      if (!button) return;
      Array.prototype.forEach.call(tabs.querySelectorAll("button"), function (other) {
        other.setAttribute("aria-pressed", String(other === button));
      });
      renderLint(button.dataset.severity);
    });
    renderLint("all");
  }

  // -------------------------------------------------------------- 5. footer
  function renderFooter() {
    var truth = COX.groundTruth || {};
    var covered = 0, total = 0;
    NAMES.forEach(function (name, index) {
      if (!(name in truth)) return;
      total++;
      var interval = COX.confidenceIntervals[index];
      if (interval[0] <= truth[name] && truth[name] <= interval[1]) covered++;
    });

    el("coxNote").innerHTML = "Concordance index <b>" + COX.concordance +
      "</b> over " + thousands(COX.comparablePairs) + " comparable pairs." +
      (total ? " The data was generated from a Weibull proportional-hazards model with " +
        "published coefficients, so the fit can be checked: <b>" + covered + " of " + total +
        "</b> generating values fall inside their 95% interval — an occasional miss is " +
        "the interval behaving as advertised, not a defect." : "");

    el("footer").innerHTML = "Fitted with opslab " + escapeHtml(DATA.version) +
      " on " + escapeHtml(DATA.generated) + ", pure standard library. " +
      '<a href="report.html">The full static report</a> &middot; ' +
      '<a href="https://github.com/pr1317/opslab">Source and documentation on GitHub</a>.';
  }

  renderSummary();
  renderMap();
  bindChartTabs();
  buildControls();
  applyPreset(PRESETS[0]);
  bindLintTabs();
  renderFooter();
})();
