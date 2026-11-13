// Forecast explorer: draws SVG from a JSON file, in the site's theme. With
// data-targets (target to summary URL), the table summarises the page's
// per-forecast scores (summarise.js) for the target, location, rounds and
// models chosen, relative to the baseline picked.
(function () {
  "use strict";
  var DAY = 864e5;
  var T = window.bdbvToggles;
  var h = T.h;
  var el = T.el;

  function day(s) { return Date.parse(s + "T00:00:00Z"); }

  function init(root) {
    var targets = root.dataset.targets ? JSON.parse(root.dataset.targets) : null;
    Promise.all([T.getJSON(root.dataset.src),
      targets ? T.getJSON(targets[Object.keys(targets)[0]]) : null])
      .then(function (both) { build(root, both[0], both[1], targets); })
      .catch(function () { root.textContent = "The explorer data could not be loaded."; });
  }

  function build(root, d, summary, targets) {
    root.textContent = "";
    var state = {
      loc: d.locations[0].id,
      round: root.dataset.round === "all" ? "all" : d.rounds.length - 1,
    };
    var controls = h("div", { class: "explorer-controls" }, root);
    var lab1 = h("label", {}, controls, "Location ");
    var selLoc = h("select", {}, lab1);
    (d.levels || ["National", "Province", "Zone"]).forEach(function (level) {
      var items = d.locations.filter(function (l) { return l.level === level; });
      if (!items.length) return;
      var g = h("optgroup", { label: level }, selLoc);
      items.forEach(function (l) {
        h("option", { value: l.id }, g, l.name + " (" + l.id + ")");
      });
    });
    var lab2 = h("label", {}, controls, "Round ");
    var selRound = h("select", {}, lab2);
    d.rounds.forEach(function (r, i) { h("option", { value: i }, selRound, r); });
    h("option", { value: "all" }, selRound, "All rounds");
    selRound.value = String(state.round);
    var hasTraj = d.models.some(function (m) { return m.trajectories; });
    var view = T.viewSelect(controls, hasTraj, function () { draw(); });
    var zoom = window.bdbvZoom.create(controls, function () { draw(); });
    T.fullscreen(controls, root);
    var rel = null;
    if (targets) {
      T.targetSelect(controls, targets, Object.keys(targets)[0], function (k) {
        T.getJSON(targets[k]).then(function (s) {
          summary = s;
          rel.fill(summary, rel.name());
          draw();
        });
      });
      rel = T.baselineSelect(controls, function () { draw(); });
      rel.fill(summary);
    }
    var Z = window.bdbvZoom;
    var legend = T.legend(root, d.models, draw);
    var note = h("p", { class: "explorer-note" }, root);
    var figure = h("div", { class: "explorer-figure" }, root);
    var table = h("div", { class: "explorer-table" }, root);

    selLoc.addEventListener("change", function () { state.loc = selLoc.value; draw(); });
    selRound.addEventListener("change", function () {
      state.round = selRound.value === "all" ? "all" : +selRound.value;
      draw();
    });

    function draw() {
      legend.paint();
      var rounds = state.round === "all"
        ? d.rounds.map(function (_, i) { return i; }) : [state.round];
      var models = d.models.map(function (_, i) { return i; })
        .filter(function (i) { return legend.on[i]; });
      var rows = d.q.filter(function (r) {
        return r[2] === state.loc && rounds.indexOf(r[1]) >= 0 && legend.on[r[0]];
      });
      var data = d.data[state.loc] || [];
      var first = day(d.rounds[rounds[0]]) - 8 * 7 * DAY;
      var lastWeek = rows.reduce(function (a, r) { return Math.max(a, day(r[3])); },
        day(d.rounds[rounds[rounds.length - 1]]));
      var pts = data.filter(function (p) {
        return day(p[0]) >= first && day(p[0]) <= lastWeek;
      });
      var xr = Z.xRange(zoom, first - 3 * DAY, lastWeek + 4 * DAY);
      var x0 = xr[0], x1 = xr[1];
      var ymax = 1;
      pts.forEach(function (p) {
        if (Z.visible(zoom, day(p[0]))) ymax = Math.max(ymax, p[1]);
      });
      rows.forEach(function (r) {
        if (Z.visible(zoom, day(r[3]))) ymax = Math.max(ymax, r[8]);
      });
      // The data as the round saw it, when one round is shown.
      var seen = rounds.length !== 1 || !d.asof ? [] : d.asof.filter(function (a) {
        return a[0] === rounds[0] && a[1] === state.loc;
      });
      seen.forEach(function (a) {
        if (Z.visible(zoom, day(a[2]))) ymax = Math.max(ymax, a[3]);
      });
      var traj = view.value !== "traj" ? [] : d.traj.filter(function (t) {
        return t[2] === state.loc && rounds.indexOf(t[1]) >= 0 && legend.on[t[0]];
      });
      traj.forEach(function (t) {
        for (var k = 5; k < t.length; k++) {
          if (Z.visible(zoom, day(t[4]) + (k - 5) * 7 * DAY)) ymax = Math.max(ymax, t[k]);
        }
      });
      var bandsOnly = d.models.filter(function (m, i) {
        return legend.on[i] && !m.trajectories;
      }).map(function (m) { return m.name; });
      var have = {};
      rows.forEach(function (r) { have[r[0]] = true; });
      var absent = d.models.filter(function (m, i) { return legend.on[i] && !have[i]; })
        .map(function (m) { return m.name; });
      var notes = [];
      if (absent.length) {
        notes.push("No forecast here for the rounds chosen from: " + absent.join(", ") + ".");
      }
      if (view.value === "traj" && bandsOnly.length) {
        notes.push("Bands only, as no samples are embedded: " + bandsOnly.join(", ") + ".");
      }
      note.textContent = notes.join(" ");
      var yr = Z.yRange(zoom, 0, ymax * 1.05, "");
      var y0 = yr[0], y1 = yr[1];
      var W = 760, H = 340, L = 52, R = 10, T = 10, B = 28;
      function X(t) { return L + (t - x0) / (x1 - x0) * (W - L - R); }
      function Y(v) { return T + (1 - (v - y0) / (y1 - y0)) * (H - T - B); }
      figure.textContent = "";
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
        "aria-label": "Forecasts by week for the selected location" }, figure);
      var plot = Z.clip(svg, { L: L, R: R, T: T, B: B, W: W, H: H });
      var cut = day(d.complete_through);
      if (cut < x1) {
        var sx = Math.max(X(cut + 3.5 * DAY), L);
        el("rect", { x: sx, y: T, width: W - R - sx, height: H - T - B,
          class: "explorer-shade" }, plot);
      }
      var span = y1 - y0;
      // Whole counts, so the step is at least 1.
      var step = Math.max(1, Z.nice(span));
      for (var v = Math.ceil(y0 / step) * step; v <= y1; v += step) {
        el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: "explorer-grid" }, svg);
        var t = el("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end",
          class: "explorer-tick" }, svg);
        t.textContent = v;
      }
      var tstep = (x1 - x0) > 70 * DAY ? 14 * DAY : 7 * DAY;
      for (var tt = x0; tt <= x1; tt += tstep) {
        var dt = new Date(tt);
        var tx = el("text", { x: X(tt), y: H - 8, "text-anchor": "middle",
          class: "explorer-tick" }, svg);
        tx.textContent = dt.getUTCDate() + " " +
          ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][dt.getUTCMonth()];
      }
      var slots = Math.max(models.length * rounds.length, 1);
      var width = Math.min(5.5 * DAY, 5.5 * DAY) / slots;
      var asTraj = function (m) { return view.value === "traj" && d.models[m].trajectories; };
      traj.forEach(function (t) {
        var pts = [];
        for (var k = 5; k < t.length; k++) {
          pts.push(X(day(t[4]) + (k - 5) * 7 * DAY) + "," + Y(t[k]));
        }
        el("polyline", { points: pts.join(" "), fill: "none", stroke: legend.colour(t[0]),
          "stroke-width": 0.8, "stroke-opacity": rounds.length > 1 ? 0.1 : 0.18 }, plot);
      });
      models.forEach(function (m) {
        if (!asTraj(m)) return;
        rounds.forEach(function (ri) {
          var own = rows.filter(function (r) { return r[0] === m && r[1] === ri; })
            .sort(function (a, b) { return day(a[3]) - day(b[3]); });
          el("polyline", { fill: "none", stroke: legend.colour(m), "stroke-width": 2,
            "stroke-dasharray": legend.dash(m), points: own.map(function (r) { return X(day(r[3])) + "," + Y(r[6]); }).join(" ") },
            plot);
        });
      });
      rows.forEach(function (r) {
        if (asTraj(r[0])) return;
        var slot = models.indexOf(r[0]) * rounds.length + rounds.indexOf(r[1]);
        var t0 = day(r[3]) + (slot - (slots - 1) / 2) * width;
        var xx = X(t0);
        var c = legend.colour(r[0]);
        var dash = legend.dash(r[0]);
        var op = rounds.length > 1 ? 0.7 : 1;
        var tip = d.models[r[0]].name + ", round " + d.rounds[r[1]] + ", week ending " + r[3] +
          ": median " + r[6] + ", 50% " + r[5] + " to " + r[7] + ", " + (d.outer || "90%") + " " + r[4] + " to " + r[8];
        var g = el("g", { opacity: op }, plot);
        el("title", {}, g).textContent = tip;
        el("line", { x1: xx, x2: xx, y1: Y(r[4]), y2: Y(r[8]), stroke: c, "stroke-width": 1,
          "stroke-dasharray": dash }, g);
        el("line", { x1: xx, x2: xx, y1: Y(r[5]), y2: Y(r[7]), stroke: c, "stroke-width": 3,
          "stroke-dasharray": dash }, g);
        // A dashed model's median is hollow, so it differs from its parent.
        el("circle", { cx: xx, cy: Y(r[6]), r: 2.5, fill: dash ? "none" : c, stroke: c }, g);
      });
      if (seen.length) {
        var sg = el("polyline", { class: "explorer-asof", fill: "none",
          points: seen.map(function (a) { return X(day(a[2])) + "," + Y(a[3]); }).join(" ") }, plot);
        el("title", {}, sg).textContent = "Data as of " + d.rounds[rounds[0]];
      }
      if (pts.length) {
        el("polyline", { class: "explorer-data", fill: "none",
          points: pts.map(function (p) { return X(day(p[0])) + "," + Y(p[1]); }).join(" ") }, plot);
        pts.forEach(function (p) {
          var done = day(p[0]) <= cut;
          var c = el("circle", { cx: X(day(p[0])), cy: Y(p[1]), r: 2.8,
            class: done ? "explorer-point" : "explorer-point hollow" }, plot);
          el("title", {}, c).textContent = p[0] + ": " + p[1] + " " + (d.unit || "cases") + " (" + (d.data_label || d.truth_vintage) + ")";
        });
      }
      rounds.forEach(function (ri) {
        var rx = X(day(d.rounds[ri]));
        if (rx >= L && rx <= W - R) {
          el("line", { x1: rx, x2: rx, y1: T, y2: H - B, class: "explorer-ref" }, plot);
        }
      });
      Z.attach(zoom, svg, { L: L, R: R, T: T, B: B, W: W, H: H, x: [x0, x1], y: [y0, y1] });
      drawTable(rounds);
    }

    function drawTable(rounds) {
      table.textContent = "";
      if (!summary) return;
      var place = summary.places.map(function (p) { return p.id; }).indexOf(state.loc);
      var origins = {};
      rounds.forEach(function (ri) {
        var o = summary.origins.indexOf(d.rounds[ri]);
        if (o >= 0) origins[o] = true;
      });
      var keep = function (r) { return r[3] === place && origins[r[1]]; };
      var s = window.bdbvSummarise(summary, keep, rel.index());
      var names = d.models.map(function (m) { return m.name; });
      window.bdbvSummaryTable(table, summary, s, function (m) {
        var i = names.indexOf(summary.models[m].name);
        return i < 0 || legend.on[i];
      }, [], null, window.bdbvEmptyNote(summary, rel.index(), keep));
    }
    draw();
    T.onTheme(draw);
  }

  function start() {
    document.querySelectorAll(".explorer[data-src]").forEach(init);
  }
  if (typeof document$ !== "undefined") document$.subscribe(start);
  else document.addEventListener("DOMContentLoaded", start);
})();
