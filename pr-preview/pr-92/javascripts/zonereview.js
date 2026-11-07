// Zone review: every place's recent reported level beside one model's
// latest nowcast and forecast weeks, from the nowcast chart's data, so
// implausible places are easy to find. The ratio is
// log((median + 1) / (recent + 1)) for the week chosen, with recent the
// mean of the last two complete weeks. Rows sort and filter with
// tablefilter.js, and a place opens in the nowcast chart whose element id
// is `data-chart` (nowcastchart.js).
// A map of the zones in `data-map` (GeoJSON with `location`) is coloured by
// the same ratio, blue for falls and red for rises, clamped at 2.
(function () {
  "use strict";
  var T = window.bdbvToggles;
  var h = T.h;
  var SCALES = { national: "National", provinces: "Province", zones: "Health zone" };
  var NS = "http://www.w3.org/2000/svg";
  var CLAMP = 2;

  function el(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(n);
    return n;
  }
  function mode() {
    return document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "light";
  }
  // Diverging colour for a ratio: from the theme's neutral to blue or red.
  function colour(v) {
    var mid = mode() === "dark" ? [58, 60, 70] : [238, 238, 238];
    var end = v < 0 ? [57, 135, 229] : [214, 69, 65];
    var t = Math.min(Math.abs(v) / CLAMP, 1);
    return "rgb(" + mid.map(function (c, i) { return Math.round(c + t * (end[i] - c)); }).join(",") + ")";
  }

  function init(root) {
    var get = function (src) { return fetch(src).then(function (r) { return r.json(); }); };
    Promise.all([get(root.dataset.src), root.dataset.map ? get(root.dataset.map) : null])
      .then(function (both) { build(root, both[0], both[1]); })
      .catch(function () { root.textContent = "The zone review data could not be loaded."; });
  }

  // SVG paths of every feature, projected with longitude scaled by
  // cos(latitude) at the middle of the map.
  function project(geo, W) {
    var pts = [];
    geo.features.forEach(function (f) {
      JSON.stringify(f.geometry.coordinates).replace(/-?[\d.]+,-?[\d.]+/g, function (m) {
        pts.push(m.split(",").map(Number));
        return m;
      });
    });
    var xs = pts.map(function (p) { return p[0]; }), ys = pts.map(function (p) { return p[1]; });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var k = Math.cos((y0 + y1) / 2 * Math.PI / 180);
    var s = W / ((x1 - x0) * k);
    var H = (y1 - y0) * s;
    function ring(r) {
      return "M" + r.map(function (p) {
        return ((p[0] - x0) * k * s).toFixed(1) + "," + ((y1 - p[1]) * s).toFixed(1);
      }).join("L") + "Z";
    }
    return {
      H: H,
      paths: geo.features.map(function (f) {
        var g = f.geometry;
        var polys = g.type === "Polygon" ? [g.coordinates] : g.coordinates;
        return { id: f.properties.location,
          d: polys.map(function (p) { return p.map(ring).join(""); }).join("") };
      }),
    };
  }

  function build(root, d, geo) {
    // Open a place in the linked chart, once that chart has loaded.
    function show(id) {
      var charts = window.bdbvNowcastCharts || {};
      if (charts[root.dataset.chart]) charts[root.dataset.chart](id);
    }
    root.textContent = "";
    var complete = {};
    Object.keys(d.data).forEach(function (k) {
      d.data[k].forEach(function (p) { if (p[0] <= d.complete_through) complete[p[0]] = true; });
    });
    var last = Object.keys(complete).sort().slice(-2);
    // Mean of the last two complete weeks, a missing week counting as 0.
    function level(id) {
      var weeks = {};
      (d.data[id] || []).forEach(function (p) { weeks[p[0]] = p[1]; });
      return last.reduce(function (a, w) { return a + (weeks[w] || 0); }, 0) / last.length;
    }
    var names = {};
    d.places.forEach(function (p) { names[p.id] = p.name; });

    var controls = h("div", { class: "explorer-controls" }, root);
    var selModel = h("select", {}, h("label", {}, controls, "Model "));
    d.models.forEach(function (m, i) {
      if (m.fan) h("option", { value: i }, selModel, m.name);
    });
    var selWeek = h("select", {}, h("label", {}, controls, "Ratio for "));
    var mapDiv = geo ? h("div", { class: "zonereview-map" }, root) : null;
    var shapes = geo ? project(geo, 760) : null;
    var holder = h("div", {}, root);
    var places = {};
    d.places.forEach(function (p) { places[p.id] = p; });

    // The ratio at a place for model m and week w, with its median.
    function ratioAt(id, m, w) {
      var r = (d.q[id] || []).filter(function (x) { return x[0] === m && x[1] === w; })[0];
      if (!r) return null;
      var base = level(id);
      return { median: r[4], recent: base, ratio: Math.log((r[4] + 1) / (base + 1)) };
    }

    function drawMap() {
      if (!geo) return;
      mapDiv.textContent = "";
      var m = +selModel.value, w = selWeek.value;
      var W = 760, pad = 34;
      var svg = el("svg", { viewBox: "0 0 " + W + " " + (shapes.H + pad), role: "img",
        "aria-label": "Health zones by the ratio of the median to the recent level" }, mapDiv);
      shapes.paths.forEach(function (s) {
        var r = ratioAt(s.id, m, w);
        var p = places[s.id];
        var path = el("path", { d: s.d, class: "zonereview-zone",
          fill: r ? colour(r.ratio) : "none" }, svg);
        el("title", {}, path).textContent = (p ? p.name + " (" + s.id + ")" : s.id) +
          (r ? ": median " + Math.round(r.median) + ", recent " + r.recent.toFixed(1) +
            ", ratio " + r.ratio.toFixed(2) : ": no forecast");
        path.addEventListener("click", function () {
          show(s.id);
        });
      });
      // Key: the colour scale from -2 to 2.
      var y = shapes.H + 10, x0 = W / 2 - 120;
      for (var i = 0; i < 40; i++) {
        el("rect", { x: x0 + i * 6, y: y, width: 6, height: 8,
          fill: colour(-CLAMP + i * 2 * CLAMP / 39) }, svg);
      }
      [-2, -1, 0, 1, 2].forEach(function (v) {
        el("text", { x: x0 + (v + CLAMP) / (2 * CLAMP) * 240, y: y + 22, "text-anchor": "middle",
          class: "explorer-tick" }, svg).textContent = String(v);
      });
    }

    // The model's latest nowcast week and every week from the reference date.
    function weeksOf(m) {
      var all = {};
      Object.keys(d.q).forEach(function (k) {
        d.q[k].forEach(function (r) { if (r[0] === m) all[r[1]] = true; });
      });
      var sorted = Object.keys(all).sort();
      var before = sorted.filter(function (w) { return w < d.reference_date; });
      return before.slice(-1).concat(sorted.filter(function (w) { return w >= d.reference_date; }));
    }
    function fillWeeks() {
      var weeks = weeksOf(+selModel.value);
      selWeek.textContent = "";
      weeks.forEach(function (w) {
        h("option", { value: w }, selWeek,
          (w < d.reference_date ? "nowcast, week ending " : "week ending ") + w);
      });
      return weeks;
    }

    function draw() {
      var m = +selModel.value;
      var weeks = weeksOf(m);
      var old = holder.querySelector("input[type=search]");
      var query = old ? old.value : "";
      holder.textContent = "";
      var block = h("div", { class: "tablefilter zonereview-table", "data-filters": "Scale,Province" }, holder);
      var table = h("table", {}, block);
      var head = h("tr", {}, h("thead", {}, table));
      ["Place", "Scale", "Province", "Recent"].concat(weeks.map(function (w) {
        return (w < d.reference_date ? "Nowcast " : "") + w;
      })).concat(["Ratio"]).forEach(function (c) { h("th", {}, head, c); });
      var body = h("tbody", {}, table);
      d.places.forEach(function (p) {
        var rows = (d.q[p.id] || []).filter(function (r) { return r[0] === m; });
        if (!rows.length) return;
        var by = {};
        rows.forEach(function (r) { by[r[1]] = r; });
        var base = level(p.id);
        var tr = h("tr", {}, body);
        var b = h("button", { type: "button", class: "zonereview-link" },
          h("td", {}, tr), p.scale === "zones" ? p.name + " (" + p.id + ")" : p.name);
        b.addEventListener("click", function () {
          show(p.id);
        });
        h("td", {}, tr, SCALES[p.scale]);
        h("td", {}, tr, p.province ? names[p.province] || p.province : "");
        h("td", {}, tr, base.toFixed(1));
        weeks.forEach(function (w) {
          var r = by[w];
          var td = h("td", r ? { "data-sort": String(r[4]) } : {}, tr,
            r ? Math.round(r[4]) + " (" + Math.round(r[2]) + " to " + Math.round(r[6]) + ")" : "");
          if (!r) td.dataset.sort = "";
        });
        var r = by[selWeek.value];
        var ratio = r ? Math.log((r[4] + 1) / (base + 1)) : NaN;
        var cls = isNaN(ratio) ? "" : ratio >= Math.LN2 ? "zonereview-up" :
          ratio <= -Math.LN2 ? "zonereview-down" : "";
        h("td", { "data-sort": isNaN(ratio) ? "" : String(ratio), class: cls }, tr,
          isNaN(ratio) ? "" : ratio.toFixed(2));
      });
      window.bdbvTableFilter(block);
      drawMap();
      var search = block.querySelector("input[type=search]");
      if (query && search) {
        search.value = query;
        search.dispatchEvent(new Event("input"));
      }
    }
    selModel.addEventListener("change", function () { fillWeeks(); draw(); });
    selWeek.addEventListener("change", draw);
    fillWeeks();
    draw();
    T.onTheme(drawMap);
  }

  function start() {
    document.querySelectorAll(".zonereview[data-src]").forEach(init);
  }
  if (typeof document$ !== "undefined") document$.subscribe(start);
  else document.addEventListener("DOMContentLoaded", start);
})();
