// Zone chart: pick a province then a zone to see its nowcast and forecast
// fan over the reported counts, with its scores by release.
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var DAY = 864e5;
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
    "Oct", "Nov", "Dec"];

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
  function fmt(v, d) { return v === null || isNaN(v) ? "" : v.toFixed(d); }

  function init(root) {
    fetch(root.dataset.src)
      .then(function (r) { return r.json(); })
      .then(function (d) { build(root, d); })
      .catch(function () { root.textContent = "The zone data could not be loaded."; });
  }

  function build(root, d) {
    root.textContent = "";
    var controls = h("div", { class: "explorer-controls" }, root);
    var selProv = h("select", {}, h("label", {}, controls, "Province "));
    d.provinces.forEach(function (p) {
      h("option", { value: p.id }, selProv, p.name + " (" + p.id + ")");
    });
    var selZone = h("select", {}, h("label", {}, controls, "Zone "));
    var figure = h("div", { class: "explorer-figure" }, root);
    var table = h("div", { class: "explorer-table" }, root);

    function fillZones() {
      selZone.textContent = "";
      d.zones.filter(function (z) { return z.province === selProv.value; })
        .forEach(function (z) { h("option", { value: z.id }, selZone, z.id); });
    }
    selProv.addEventListener("change", function () { fillZones(); draw(); });
    selZone.addEventListener("change", draw);

    function draw() {
      var zone = selZone.value;
      var data = d.data[zone] || [];
      var q = d.q[zone] || [];
      var ref = day(d.reference_date);
      var x0 = Math.min(day(data[0][0]), day(q[0][0])) - 4 * DAY;
      var x1 = day(q[q.length - 1][0]) + 4 * DAY;
      var ymax = 1;
      data.forEach(function (p) { ymax = Math.max(ymax, p[1]); });
      q.forEach(function (r) { ymax = Math.max(ymax, r[5]); });
      ymax *= 1.05;
      var W = 760, H = 300, L = 44, R = 10, T = 10, B = 28;
      function X(t) { return L + (t - x0) / (x1 - x0) * (W - L - R); }
      function Y(v) { return T + (1 - v / ymax) * (H - T - B); }
      figure.textContent = "";
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
        "aria-label": "Nowcast and forecast for " + zone }, figure);
      var step = ymax > 200 ? 50 : ymax > 80 ? 20 : ymax > 40 ? 10 : ymax > 15 ? 5 : ymax > 6 ? 2 : 1;
      for (var v = 0; v <= ymax; v += step) {
        el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: "explorer-grid" }, svg);
        el("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end",
          class: "explorer-tick" }, svg).textContent = v;
      }
      for (var t = ref; t >= x0; t -= 28 * DAY) {
        var dt = new Date(t);
        el("text", { x: X(t), y: H - 8, "text-anchor": "middle", class: "explorer-tick" },
          svg).textContent = dt.getUTCDate() + " " + MONTHS[dt.getUTCMonth()];
      }
      function band(lo, hi, cls) {
        var top = q.map(function (r) { return X(day(r[0])) + "," + Y(r[hi]); });
        var bot = q.slice().reverse().map(function (r) { return X(day(r[0])) + "," + Y(r[lo]); });
        el("polygon", { points: top.concat(bot).join(" "), class: cls }, svg);
      }
      band(1, 5, "zonechart-band90");
      band(2, 4, "zonechart-band50");
      el("polyline", { fill: "none", class: "zonechart-median",
        points: q.map(function (r) { return X(day(r[0])) + "," + Y(r[3]); }).join(" ") }, svg);
      q.forEach(function (r) {
        var c = el("circle", { cx: X(day(r[0])), cy: Y(r[3]), r: 6, fill: "transparent" }, svg);
        el("title", {}, c).textContent = "Week ending " + r[0] + ": median " + r[3] +
          ", 50% " + r[2] + " to " + r[4] + ", 90% " + r[1] + " to " + r[5];
      });
      el("polyline", { class: "explorer-data", fill: "none",
        points: data.map(function (p) { return X(day(p[0])) + "," + Y(p[1]); }).join(" ") }, svg);
      data.forEach(function (p) {
        var c = el("circle", { cx: X(day(p[0])), cy: Y(p[1]), r: 2.8, class: "explorer-point" }, svg);
        el("title", {}, c).textContent = p[0] + ": " + p[1] + " reported (" + d.release + ")";
      });
      el("line", { x1: X(ref), x2: X(ref), y1: T, y2: H - B, class: "explorer-ref" }, svg);
      drawTable(zone);
    }

    function drawTable(zone) {
      table.textContent = "";
      var s = d.scores[zone] || [];
      if (!s.length) {
        h("p", { class: "explorer-note" }, table, "No release scores for this zone yet.");
        return;
      }
      var tbl = h("table", {}, table);
      var head = h("tr", {}, h("thead", {}, tbl));
      ["Origin", "Model", "n", "Relative WIS", "50% coverage", "90% coverage"]
        .forEach(function (c) { h("th", {}, head, c); });
      var body = h("tbody", {}, tbl);
      s.forEach(function (r) {
        var tr = h("tr", {}, body);
        h("td", {}, tr, r[0]);
        h("td", {}, tr, r[1]);
        h("td", {}, tr, String(r[2]));
        h("td", {}, tr, fmt(r[3], 2));
        h("td", {}, tr, fmt(r[4], 2));
        h("td", {}, tr, fmt(r[5], 2));
      });
    }
    fillZones();
    draw();
  }

  function start() {
    document.querySelectorAll(".zonechart[data-src]").forEach(init);
  }
  if (typeof document$ !== "undefined") document$.subscribe(start);
  else document.addEventListener("DOMContentLoaded", start);
})();
