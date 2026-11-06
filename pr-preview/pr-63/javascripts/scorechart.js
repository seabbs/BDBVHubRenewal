// Score chart: one summary table as lines or grouped bars, a tab per
// metric and a checkbox per model (toggles.js), drawn as SVG in the site's
// theme. A metric with `log` is drawn on a log axis, bars from its ref.
// Other scripts build one with bdbvScoreChart.build and give it new rows
// with update, keeping the tab, ticks and zoom.
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var T = window.bdbvToggles;
  var Z = window.bdbvZoom;
  var h = T.h;

  function el(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(n);
    return n;
  }
  var nice = Z.nice;

  // Ticks of 1, 2 and 5 times a power of 10 between 10^lo and 10^hi, of
  // powers of 10 alone past 3 decades, or round steps where the range is
  // too narrow to hold two of them.
  function logTicks(lo, hi) {
    var out = [];
    var steps = hi - lo > 3 ? [1] : [1, 2, 5];
    for (var p = Math.floor(lo); p <= Math.ceil(hi); p++) {
      steps.forEach(function (f) {
        var v = f * Math.pow(10, p);
        if (Math.log10(v) >= lo && Math.log10(v) <= hi) out.push(v);
      });
    }
    if (out.length >= 2) return out;
    var a = Math.pow(10, lo), b = Math.pow(10, hi), step = nice(b - a);
    out = [];
    for (var v = Math.ceil(a / step) * step; v <= b; v += step) out.push(v);
    return out;
  }

  // onDraw(metric index, ticks), if given, runs after each draw.
  function build(root, d, onDraw) {
    root.textContent = "";
    var state = { metric: 0 };
    var tabs = h("div", { class: "scorechart-tabs", role: "tablist" }, root);
    var buttons = d.metrics.map(function (m, i) {
      var b = h("button", { type: "button", role: "tab" }, tabs, m.label);
      b.addEventListener("click", function () { state.metric = i; draw(); });
      return b;
    });
    var zoom = Z.create(tabs, function () { draw(); });
    var keys = T.legend(root, d.models, function () { draw(); });
    state.on = keys.on;
    var figure = h("div", { class: "scorechart-figure" }, root);

    function draw() {
      keys.paint();
      buttons.forEach(function (b, i) {
        b.setAttribute("aria-selected", i === state.metric ? "true" : "false");
      });
      var metric = d.metrics[state.metric];
      var col = 3 + state.metric;
      // On a log axis, values are drawn as their log10 and non-positive
      // ones are left out.
      var log = !!metric.log;
      var tf = log ? Math.log10 : function (v) { return v; };
      var rows = d.rows.filter(function (r) {
        return state.on[r[0]] && r[col] !== null && (!log || r[col] > 0);
      });
      var nx = d.x.length;
      var xr = Z.xRange(zoom, -0.5, nx - 0.5);
      var vals = rows.filter(function (r) { return Z.visible(zoom, r[1]); })
        .map(function (r) { return tf(r[col]); });
      if (metric.ref !== null) vals.push(tf(metric.ref));
      var lo = Math.min.apply(null, vals.concat(log ? [] : [0]));
      var hi = Math.max.apply(null, vals.concat([lo + 1e-9]));
      var step = nice(hi - lo || 1);
      if (log) {
        var pad = Math.max((hi - lo) * 0.05, 0.05);
        lo -= pad;
        hi += pad;
      } else {
        lo = Math.floor(lo / step) * step;
        hi = Math.ceil(hi * 1.02 / step) * step;
      }
      var yr = Z.yRange(zoom, lo, hi, metric.key);
      if (yr[0] !== lo || yr[1] !== hi) {
        lo = yr[0];
        hi = yr[1];
        step = nice(hi - lo);
      }
      var ticks = [];
      if (log) {
        ticks = logTicks(lo, hi);
      } else {
        for (var v = Math.ceil(lo / step) * step; v <= hi + step / 2; v += step) ticks.push(v);
      }
      var many = d.x.filter(function (_, i) { return Z.visible(zoom, i); }).length > 12;
      var W = 760, H = many ? 380 : 320, L = 52, R = 10, T = 10, B = many ? 100 : 40;
      var bw = (W - L - R) / (xr[1] - xr[0]);
      function X(i) { return L + (i - xr[0]) * bw; }
      function Y(v) { return T + (1 - (v - lo) / (hi - lo)) * (H - T - B); }
      figure.textContent = "";
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
        "aria-label": metric.label + " by " + d.x_label.toLowerCase() }, figure);
      ticks.forEach(function (v) {
        el("line", { x1: L, x2: W - R, y1: Y(tf(v)), y2: Y(tf(v)), class: "scorechart-grid" }, svg);
        var t = el("text", { x: L - 6, y: Y(tf(v)) + 4, "text-anchor": "end",
          class: "scorechart-tick" }, svg);
        t.textContent = +v.toFixed(6);
      });
      d.x.forEach(function (x, i) {
        if (!Z.visible(zoom, i)) return;
        var tx = many ?
          el("text", { x: X(i), y: H - B + 12, "text-anchor": "end", class: "scorechart-tick",
            transform: "rotate(-60 " + X(i) + " " + (H - B + 12) + ")" }, svg) :
          el("text", { x: X(i), y: H - 22, "text-anchor": "middle",
            class: "scorechart-tick" }, svg);
        tx.textContent = x;
      });
      var lab = el("text", { x: L + (W - L - R) / 2, y: H - 4, "text-anchor": "middle",
        class: "scorechart-tick" }, svg);
      lab.textContent = d.x_label;
      var plot = Z.clip(svg, { L: L, R: R, T: T, B: B, W: W, H: H });
      if (metric.ref !== null) {
        el("line", { x1: L, x2: W - R, y1: Y(tf(metric.ref)), y2: Y(tf(metric.ref)),
          class: "scorechart-ref" }, plot);
      }
      var shown = d.models.map(function (_, i) { return i; })
        .filter(function (i) { return state.on[i]; });
      function tip(r) {
        return d.models[r[0]].name + ", " + d.x_label.toLowerCase() + " " + d.x[r[1]] +
          ": " + metric.label + " " + (+r[col].toFixed(3)) + ", n " + r[2];
      }
      shown.forEach(function (m, k) {
        var c = keys.colour(m);
        var fill = keys.dash(m) ? T.hatch(svg, c) : c;
        var own = rows.filter(function (r) { return r[0] === m; })
          .sort(function (a, b) { return a[1] - b[1]; });
        if (d.kind === "bar") {
          var w = bw * 0.8 / shown.length;
          own.forEach(function (r) {
            var x = X(r[1]) - bw * 0.4 + k * w;
            var base = log ? tf(metric.ref) : 0;
            var y0 = Y(Math.min(Math.max(lo, base), hi)), y1 = Y(tf(r[col]));
            var bar = el("rect", { x: x, y: Math.min(y0, y1), width: w,
              height: Math.abs(y0 - y1), fill: fill, stroke: c }, plot);
            el("title", {}, bar).textContent = tip(r);
          });
        } else {
          el("polyline", { fill: "none", stroke: c, "stroke-width": 2,
            "stroke-dasharray": keys.dash(m),
            points: own.map(function (r) { return X(r[1]) + "," + Y(tf(r[col])); }).join(" ") }, plot);
          own.forEach(function (r) {
            var p = el("circle", { cx: X(r[1]), cy: Y(tf(r[col])), r: 3.5, fill: c }, plot);
            el("title", {}, p).textContent = tip(r);
          });
        }
      });
      Z.attach(zoom, svg, { L: L, R: R, T: T, B: B, W: W, H: H, x: xr, y: [lo, hi] },
        metric.key);
      if (onDraw) onDraw(state.metric, state.on);
    }
    draw();
    T.onTheme(draw);
    return {
      update: function (rows, x) {
        d.rows = rows;
        d.x = x;
        draw();
      },
    };
  }

  window.bdbvScoreChart = { build: build };
})();
