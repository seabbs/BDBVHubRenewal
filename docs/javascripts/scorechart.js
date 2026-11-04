// Score chart: one summary table as lines or grouped bars, a tab per
// metric and a checkbox per model (toggles.js), drawn as SVG in the site's
// theme.
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var T = window.bdbvToggles;
  var h = T.h;

  function el(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(n);
    return n;
  }
  function nice(span) {
    var p = Math.pow(10, Math.floor(Math.log10(span / 5)));
    var f = span / 5 / p;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
  }

  function init(root) {
    fetch(root.dataset.src)
      .then(function (r) { return r.json(); })
      .then(function (d) { build(root, d); })
      .catch(function () { root.textContent = "The chart data could not be loaded."; });
  }

  function build(root, d) {
    root.textContent = "";
    var state = { metric: 0 };
    var tabs = h("div", { class: "scorechart-tabs", role: "tablist" }, root);
    var buttons = d.metrics.map(function (m, i) {
      var b = h("button", { type: "button", role: "tab" }, tabs, m.label);
      b.addEventListener("click", function () { state.metric = i; draw(); });
      return b;
    });
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
      var rows = d.rows.filter(function (r) { return state.on[r[0]] && r[col] !== null; });
      var vals = rows.map(function (r) { return r[col]; });
      if (metric.ref !== null) vals.push(metric.ref);
      var lo = Math.min.apply(null, vals.concat([0]));
      var hi = Math.max.apply(null, vals.concat([lo + 1e-9]));
      var step = nice(hi - lo || 1);
      lo = Math.floor(lo / step) * step;
      hi = Math.ceil(hi * 1.02 / step) * step;
      var W = 760, H = 320, L = 52, R = 10, T = 10, B = 40;
      var nx = d.x.length;
      var bw = (W - L - R) / Math.max(nx, 1);
      function X(i) { return L + (i + 0.5) * bw; }
      function Y(v) { return T + (1 - (v - lo) / (hi - lo)) * (H - T - B); }
      figure.textContent = "";
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
        "aria-label": metric.label + " by " + d.x_label.toLowerCase() }, figure);
      for (var v = lo; v <= hi + step / 2; v += step) {
        el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: "scorechart-grid" }, svg);
        var t = el("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end",
          class: "scorechart-tick" }, svg);
        t.textContent = +v.toFixed(6);
      }
      d.x.forEach(function (x, i) {
        var tx = el("text", { x: X(i), y: H - 22, "text-anchor": "middle",
          class: "scorechart-tick" }, svg);
        tx.textContent = x;
      });
      var lab = el("text", { x: L + (W - L - R) / 2, y: H - 4, "text-anchor": "middle",
        class: "scorechart-tick" }, svg);
      lab.textContent = d.x_label;
      if (metric.ref !== null) {
        el("line", { x1: L, x2: W - R, y1: Y(metric.ref), y2: Y(metric.ref),
          class: "scorechart-ref" }, svg);
      }
      var shown = d.models.map(function (_, i) { return i; })
        .filter(function (i) { return state.on[i]; });
      function tip(r) {
        return d.models[r[0]].name + ", " + d.x_label.toLowerCase() + " " + d.x[r[1]] +
          ": " + metric.label + " " + (+r[col].toFixed(3)) + ", n " + r[2];
      }
      shown.forEach(function (m, k) {
        var c = keys.colour(m);
        var own = rows.filter(function (r) { return r[0] === m; })
          .sort(function (a, b) { return a[1] - b[1]; });
        if (d.kind === "bar") {
          var w = bw * 0.8 / shown.length;
          own.forEach(function (r) {
            var x = X(r[1]) - bw * 0.4 + k * w;
            var y0 = Y(Math.max(lo, 0)), y1 = Y(r[col]);
            var bar = el("rect", { x: x, y: Math.min(y0, y1), width: w,
              height: Math.abs(y0 - y1), fill: c }, svg);
            el("title", {}, bar).textContent = tip(r);
          });
        } else {
          el("polyline", { fill: "none", stroke: c, "stroke-width": 2,
            points: own.map(function (r) { return X(r[1]) + "," + Y(r[col]); }).join(" ") }, svg);
          own.forEach(function (r) {
            var p = el("circle", { cx: X(r[1]), cy: Y(r[col]), r: 3.5, fill: c }, svg);
            el("title", {}, p).textContent = tip(r);
          });
        }
      });
    }
    draw();
    T.onTheme(draw);
  }

  function start() {
    document.querySelectorAll(".scorechart[data-src]").forEach(init);
  }
  if (typeof document$ !== "undefined") document$.subscribe(start);
  else document.addEventListener("DOMContentLoaded", start);
})();
