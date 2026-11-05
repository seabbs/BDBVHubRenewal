// Nowcast chart: pick a scale and a place (for a zone, its province first)
// to see each model's nowcast and forecast over the reported counts, as
// bands or trajectories, with its scores (summarise.js) for the rounds
// chosen. Model tick boxes come from toggles.js.
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var DAY = 864e5;
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
    "Oct", "Nov", "Dec"];
  var T = window.bdbvToggles;
  var Z = window.bdbvZoom;
  var h = T.h;

  function el(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(n);
    return n;
  }
  function day(s) { return Date.parse(s + "T00:00:00Z"); }
  function fmt(v, d) { return v === null || isNaN(v) ? "" : v.toFixed(d); }

  function init(root) {
    fetch(root.dataset.src)
      .then(function (r) { return r.json(); })
      .then(function (d) {
        return fetch(root.dataset.summary).then(function (r) { return r.json(); })
          .then(function (summary) { build(root, d, summary); });
      })
      .catch(function () { root.textContent = "The nowcast data could not be loaded."; });
  }

  function build(root, d, summary) {
    root.textContent = "";
    var controls = h("div", { class: "explorer-controls" }, root);
    var scales = [["national", "National"], ["provinces", "Provinces"], ["zones", "Health zones"]]
      .filter(function (s) { return d.places.some(function (p) { return p.scale === s[0]; }); });
    var selScale = h("select", {}, h("label", {}, controls, "Scale "));
    scales.forEach(function (s) { h("option", { value: s[0] }, selScale, s[1]); });
    var provLabel = h("label", {}, controls, "Province ");
    var selProv = h("select", {}, provLabel);
    d.places.filter(function (p) { return p.scale === "provinces"; }).forEach(function (p) {
      if (d.places.some(function (z) { return z.province === p.id; })) {
        h("option", { value: p.id }, selProv, p.name);
      }
    });
    var selZone = h("select", {}, h("label", {}, controls, "Place "));
    var selRound = h("select", {}, h("label", {}, controls, "Scores for "));
    h("option", { value: "all" }, selRound, "All rounds");
    summary.origins.forEach(function (o, i) { h("option", { value: i }, selRound, o); });
    var view = T.viewSelect(controls, true, function () { draw(); });
    var zoom = Z.create(controls, function () { draw(); });
    var keys = T.legend(root, d.models, function () { draw(); });
    var note = h("p", { class: "explorer-note" }, root);
    var figure = h("div", { class: "explorer-figure" }, root);
    var table = h("div", { class: "explorer-table" }, root);

    function fillZones() {
      var zones = selScale.value === "zones";
      provLabel.hidden = !zones;
      selZone.textContent = "";
      d.places.filter(function (p) {
        return p.scale === selScale.value && (!zones || p.province === selProv.value);
      }).forEach(function (p) {
        h("option", { value: p.id }, selZone, zones ? p.id : p.name);
      });
    }
    selScale.addEventListener("change", function () { fillZones(); draw(); });
    selProv.addEventListener("change", function () { fillZones(); draw(); });
    selZone.addEventListener("change", draw);
    selRound.addEventListener("change", draw);

    function ticked() {
      return d.models.map(function (_, i) { return i; })
        .filter(function (i) { return keys.on[i]; });
    }

    function draw() {
      keys.paint();
      var zone = selZone.value;
      var data = d.data[zone] || [];
      var all = d.q[zone] || [];
      var on = ticked();
      var none = d.models.filter(function (m) { return !m.fan; })
        .map(function (m) { return m.name; });
      note.textContent = none.length ?
        "No forecast at the " + d.release + " release from " + none.join(", ") + "." : "";
      var q = all.filter(function (r) { return keys.on[r[0]]; });
      var ref = day(d.reference_date);
      var x0 = day(data[0][0]) - 4 * DAY;
      var x1 = ref + 4 * DAY;
      q.forEach(function (r) { x1 = Math.max(x1, day(r[1]) + 4 * DAY); });
      var xr = Z.xRange(zoom, x0, x1);
      x0 = xr[0];
      x1 = xr[1];
      var ymax = 1;
      data.forEach(function (p) {
        if (Z.visible(zoom, day(p[0]))) ymax = Math.max(ymax, p[1]);
      });
      var traj = view.value !== "traj" ? [] : (d.traj[zone] || []).filter(function (t) {
        return keys.on[t[0]];
      });
      if (view.value === "traj") {
        traj.forEach(function (t) {
          for (var k = 3; k < t.length; k++) {
            if (Z.visible(zoom, day(t[2]) + (k - 3) * 7 * DAY)) ymax = Math.max(ymax, t[k]);
          }
        });
        q.forEach(function (r) {
          if (Z.visible(zoom, day(r[1]))) ymax = Math.max(ymax, r[4]);
        });
      } else {
        q.forEach(function (r) {
          if (Z.visible(zoom, day(r[1]))) ymax = Math.max(ymax, r[6]);
        });
      }
      var yr = Z.yRange(zoom, 0, ymax * 1.05, "");
      var y0 = yr[0], y1 = yr[1];
      var W = 760, H = 300, L = 44, R = 10, Tp = 10, B = 28;
      function X(t) { return L + (t - x0) / (x1 - x0) * (W - L - R); }
      function Y(v) { return Tp + (1 - (v - y0) / (y1 - y0)) * (H - Tp - B); }
      figure.textContent = "";
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
        "aria-label": "Nowcast and forecast for " + zone }, figure);
      var span = y1 - y0;
      // Whole counts, so the step is at least 1.
      var step = Math.max(1, Z.nice(span));
      for (var v = Math.ceil(y0 / step) * step; v <= y1; v += step) {
        el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: "explorer-grid" }, svg);
        el("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end",
          class: "explorer-tick" }, svg).textContent = v;
      }
      var tstep = (x1 - x0) > 70 * DAY ? 28 * DAY : 7 * DAY;
      var tfirst = ref - Math.floor((ref - x0) / tstep) * tstep;
      for (var t = tfirst; t <= x1; t += tstep) {
        var dt = new Date(t);
        el("text", { x: X(t), y: H - 8, "text-anchor": "middle", class: "explorer-tick" },
          svg).textContent = dt.getUTCDate() + " " + MONTHS[dt.getUTCMonth()];
      }
      var plot = Z.clip(svg, { L: L, R: R, T: Tp, B: B, W: W, H: H });
      on.forEach(function (m) {
        var own = q.filter(function (r) { return r[0] === m; });
        if (!own.length) return;
        var c = keys.colour(m);
        function band(lo, hi, op) {
          var top = own.map(function (r) { return X(day(r[1])) + "," + Y(r[hi]); });
          var bot = own.slice().reverse().map(function (r) { return X(day(r[1])) + "," + Y(r[lo]); });
          el("polygon", { points: top.concat(bot).join(" "), fill: c, "fill-opacity": op }, plot);
        }
        if (view.value === "traj") {
          traj.filter(function (t) { return t[0] === m; }).forEach(function (t) {
            var pts = [];
            for (var k = 3; k < t.length; k++) {
              pts.push(X(day(t[2]) + (k - 3) * 7 * DAY) + "," + Y(t[k]));
            }
            el("polyline", { points: pts.join(" "), fill: "none", stroke: c,
              "stroke-width": 0.8, "stroke-opacity": 0.18 }, plot);
          });
        } else {
          band(2, 6, 0.12);
          band(3, 5, 0.22);
        }
        el("polyline", { fill: "none", stroke: c, "stroke-width": 2,
          "stroke-dasharray": keys.dash(m),
          points: own.map(function (r) { return X(day(r[1])) + "," + Y(r[4]); }).join(" ") }, plot);
        own.forEach(function (r) {
          var p = el("circle", { cx: X(day(r[1])), cy: Y(r[4]), r: 6, fill: "transparent" }, plot);
          el("title", {}, p).textContent = d.models[m].name + ", week ending " + r[1] +
            ": median " + r[4] + ", 50% " + r[3] + " to " + r[5] + ", 90% " + r[2] + " to " + r[6];
        });
      });
      el("polyline", { class: "explorer-data", fill: "none",
        points: data.map(function (p) { return X(day(p[0])) + "," + Y(p[1]); }).join(" ") }, plot);
      data.forEach(function (p) {
        var c = el("circle", { cx: X(day(p[0])), cy: Y(p[1]), r: 2.8, class: "explorer-point" }, plot);
        el("title", {}, c).textContent = p[0] + ": " + p[1] + " reported (" + d.release + ")";
      });
      el("line", { x1: X(ref), x2: X(ref), y1: Tp, y2: H - B, class: "explorer-ref" }, plot);
      Z.attach(zoom, svg, { L: L, R: R, T: Tp, B: B, W: W, H: H, x: [x0, x1], y: [y0, y1] });
      drawTable(zone, on);
    }

    function drawTable(zone, on) {
      table.textContent = "";
      var place = summary.places.map(function (p) { return p.id; }).indexOf(zone);
      var round = selRound.value === "all" ? null : +selRound.value;
      var res = window.bdbvSummarise(summary, function (r) {
        return r[3] === place && (round === null || r[1] === round);
      });
      var names = d.models.map(function (m) { return m.name; });
      var ticked = function (m) {
        var i = names.indexOf(summary.models[m].name);
        return i < 0 || on.indexOf(i) >= 0;
      };
      var scored = res.map(function (r) { return summary.models[r.model].name; });
      var missing = on.map(function (i) { return names[i]; })
        .filter(function (n) { return scored.indexOf(n) < 0; });
      if (missing.length) {
        h("p", { class: "explorer-note" }, table,
          "No forecast scored here for: " + missing.join(", ") + ".");
      }
      window.bdbvSummaryTable(table, summary, res, ticked);
    }
    fillZones();
    draw();
    T.onTheme(draw);
  }

  function start() {
    document.querySelectorAll(".nowcastchart[data-src]").forEach(init);
  }
  if (typeof document$ !== "undefined") document$.subscribe(start);
  else document.addEventListener("DOMContentLoaded", start);
})();
