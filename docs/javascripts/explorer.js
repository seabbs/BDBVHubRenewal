// Forecast explorer: draws SVG from a JSON file, in the site's theme.
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var PALETTE = [
    "#4c78d8", "#e8743b", "#2ca58d", "#c0509a",
    "#9a7fd1", "#d9a521", "#5ab0d6", "#a0522d",
  ];
  var DAY = 864e5;

  function el(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(n);
    return n;
  }
  function h(tag, attrs, parent, text) {
    var n = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (text !== undefined) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  }
  function day(s) { return Date.parse(s + "T00:00:00Z"); }
  function mean(a) { return a.reduce(function (x, y) { return x + y; }, 0) / a.length; }
  function fmt(v, d) { return v === null || isNaN(v) ? "" : v.toFixed(d); }

  function init(root) {
    fetch(root.dataset.src)
      .then(function (r) { return r.json(); })
      .then(function (d) { build(root, d); })
      .catch(function () { root.textContent = "The explorer data could not be loaded."; });
  }

  function build(root, d) {
    root.textContent = "";
    var state = {
      loc: d.locations[0].id,
      round: d.rounds.length - 1,
      off: {},
    };
    var controls = h("div", { class: "explorer-controls" }, root);
    var lab1 = h("label", {}, controls, "Location ");
    var selLoc = h("select", {}, lab1);
    ["National", "Province", "Zone"].forEach(function (level) {
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
    var legend = h("div", { class: "explorer-legend" }, root);
    var figure = h("div", { class: "explorer-figure" }, root);
    var table = h("div", { class: "explorer-table" }, root);

    d.models.forEach(function (m, i) {
      var l = h("label", { class: "explorer-key" }, legend);
      var cb = h("input", { type: "checkbox", checked: "" }, l);
      var sw = h("span", { class: "explorer-swatch" }, l);
      sw.style.background = PALETTE[i % PALETTE.length];
      h("span", {}, l, m.label);
      cb.addEventListener("change", function () {
        state.off[i] = !cb.checked;
        draw();
      });
    });
    selLoc.addEventListener("change", function () { state.loc = selLoc.value; draw(); });
    selRound.addEventListener("change", function () {
      state.round = selRound.value === "all" ? "all" : +selRound.value;
      draw();
    });

    function draw() {
      var rounds = state.round === "all"
        ? d.rounds.map(function (_, i) { return i; }) : [state.round];
      var models = d.models.map(function (_, i) { return i; })
        .filter(function (i) { return !state.off[i]; });
      var rows = d.q.filter(function (r) {
        return r[2] === state.loc && rounds.indexOf(r[1]) >= 0 && !state.off[r[0]];
      });
      var data = d.data[state.loc] || [];
      var first = day(d.rounds[rounds[0]]) - 8 * 7 * DAY;
      var lastWeek = rows.reduce(function (a, r) { return Math.max(a, day(r[3])); },
        day(d.rounds[rounds[rounds.length - 1]]));
      var pts = data.filter(function (p) {
        return day(p[0]) >= first && day(p[0]) <= lastWeek;
      });
      var ymax = 1;
      pts.forEach(function (p) { ymax = Math.max(ymax, p[1]); });
      rows.forEach(function (r) { ymax = Math.max(ymax, r[8]); });
      ymax *= 1.05;
      var W = 760, H = 340, L = 52, R = 10, T = 10, B = 28;
      var x0 = first - 3 * DAY, x1 = lastWeek + 4 * DAY;
      function X(t) { return L + (t - x0) / (x1 - x0) * (W - L - R); }
      function Y(v) { return T + (1 - v / ymax) * (H - T - B); }
      figure.textContent = "";
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
        "aria-label": "Forecasts by week for the selected location" }, figure);
      var cut = day(d.complete_through);
      if (cut < x1) {
        var sx = Math.max(X(cut + 3.5 * DAY), L);
        el("rect", { x: sx, y: T, width: W - R - sx, height: H - T - B,
          class: "explorer-shade" }, svg);
      }
      var step = ymax > 1000 ? 500 : ymax > 400 ? 100 : ymax > 100 ? 50 : ymax > 40 ? 10 : 5;
      for (var v = 0; v <= ymax; v += step) {
        el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: "explorer-grid" }, svg);
        var t = el("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end",
          class: "explorer-tick" }, svg);
        t.textContent = v;
      }
      for (var tt = x0; tt <= x1; tt += 14 * DAY) {
        var dt = new Date(tt);
        var tx = el("text", { x: X(tt), y: H - 8, "text-anchor": "middle",
          class: "explorer-tick" }, svg);
        tx.textContent = dt.getUTCDate() + " " +
          ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][dt.getUTCMonth()];
      }
      var slots = Math.max(models.length * rounds.length, 1);
      var width = Math.min(5.5 * DAY, 5.5 * DAY) / slots;
      rows.forEach(function (r) {
        var slot = models.indexOf(r[0]) * rounds.length + rounds.indexOf(r[1]);
        var t0 = day(r[3]) + (slot - (slots - 1) / 2) * width;
        var xx = X(t0);
        var c = PALETTE[r[0] % PALETTE.length];
        var op = rounds.length > 1 ? 0.7 : 1;
        var tip = d.models[r[0]].label + ", round " + d.rounds[r[1]] + ", week ending " + r[3] +
          ": median " + r[6] + ", 50% " + r[5] + " to " + r[7] + ", 90% " + r[4] + " to " + r[8];
        var g = el("g", { opacity: op }, svg);
        el("title", {}, g).textContent = tip;
        el("line", { x1: xx, x2: xx, y1: Y(r[4]), y2: Y(r[8]), stroke: c, "stroke-width": 1 }, g);
        el("line", { x1: xx, x2: xx, y1: Y(r[5]), y2: Y(r[7]), stroke: c, "stroke-width": 3 }, g);
        el("circle", { cx: xx, cy: Y(r[6]), r: 2.5, fill: c }, g);
      });
      if (pts.length) {
        el("polyline", { class: "explorer-data", fill: "none",
          points: pts.map(function (p) { return X(day(p[0])) + "," + Y(p[1]); }).join(" ") }, svg);
        pts.forEach(function (p) {
          var done = day(p[0]) <= cut;
          var c = el("circle", { cx: X(day(p[0])), cy: Y(p[1]), r: 2.8,
            class: done ? "explorer-point" : "explorer-point hollow" }, svg);
          el("title", {}, c).textContent = p[0] + ": " + p[1] + " cases (" + d.truth_vintage + ")";
        });
      }
      rounds.forEach(function (ri) {
        var rx = X(day(d.rounds[ri]));
        if (rx >= L && rx <= W - R) {
          el("line", { x1: rx, x2: rx, y1: T, y2: H - B, class: "explorer-ref" }, svg);
        }
      });
      drawTable(rounds);
    }

    function drawTable(rounds) {
      table.textContent = "";
      var s = d.scores.filter(function (r) {
        return r[2] === state.loc && rounds.indexOf(r[1]) >= 0;
      });
      if (!s.length) {
        h("p", { class: "explorer-note" }, table,
          "No scores yet for this selection. Rounds are scored once later data are published.");
        return;
      }
      var by = {};
      s.forEach(function (r) { (by[r[0]] = by[r[0]] || []).push(r); });
      var key = function (r) { return r[1] + "|" + r[3]; };
      var base = {};
      (by[d.baseline] || []).forEach(function (r) { base[key(r)] = r[4]; });
      var tbl = h("table", {}, table);
      var head = h("tr", {}, h("thead", {}, tbl));
      ["Model", "Weeks", "WIS", "Relative WIS", "50% coverage", "90% coverage", "Bias"]
        .forEach(function (c) { h("th", {}, head, c); });
      var body = h("tbody", {}, tbl);
      Object.keys(by).map(Number).forEach(function (m) {
        if (state.off[m]) return;
        var rs = by[m];
        var rel = "";
        if (d.baseline !== null && m !== d.baseline) {
          var both = rs.filter(function (r) { return key(r) in base; });
          if (both.length) {
            rel = fmt(mean(both.map(function (r) { return r[4]; })) /
              mean(both.map(function (r) { return base[key(r)]; })), 2);
          }
        }
        var tr = h("tr", {}, body);
        h("td", {}, tr, d.models[m].label);
        h("td", {}, tr, String(rs.length));
        h("td", {}, tr, fmt(mean(rs.map(function (r) { return r[4]; })), 1));
        h("td", {}, tr, rel);
        h("td", {}, tr, fmt(mean(rs.map(function (r) { return r[5]; })), 2));
        h("td", {}, tr, fmt(mean(rs.map(function (r) { return r[6]; })), 2));
        h("td", {}, tr, fmt(mean(rs.map(function (r) { return r[7]; })), 2));
      });
    }
    draw();
  }

  function start() {
    document.querySelectorAll(".explorer[data-src]").forEach(init);
  }
  if (typeof document$ !== "undefined") document$.subscribe(start);
  else document.addEventListener("DOMContentLoaded", start);
})();
