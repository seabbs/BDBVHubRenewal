// Place map shared by the interactive charts: health zones from a GeoJSON
// file (features keyed by `location`), coloured by one value per zone or,
// at province level, per province, on a diverging scale around a centre.
// Clicking a zone picks it, and the picked place is outlined. Also the
// ratio of a nowcast to the recent count that the Nowcast page maps.
(function () {
  "use strict";
  var W = 600;
  var h = window.bdbvToggles.h;
  var el = window.bdbvToggles.el;

  function mode() {
    return document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "light";
  }
  // From the theme's neutral at 0 to blue at -1 or red at 1.
  function colour(t) {
    var mid = mode() === "dark" ? [58, 60, 70] : [238, 238, 238];
    var end = t < 0 ? [57, 135, 229] : [214, 69, 65];
    var a = Math.min(Math.abs(t), 1);
    return "rgb(" + mid.map(function (c, i) { return Math.round(c + a * (end[i] - c)); }).join(",") + ")";
  }
  function province(id) { return id.split("-").slice(0, 2).join("-"); }

  // SVG paths of every zone, with longitude scaled by cos(latitude) at the
  // middle of the map.
  function project(geo) {
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
    function ring(r) {
      return "M" + r.map(function (p) {
        return ((p[0] - x0) * k * s).toFixed(1) + "," + ((y1 - p[1]) * s).toFixed(1);
      }).join("L") + "Z";
    }
    return {
      H: (y1 - y0) * s,
      zones: geo.features.map(function (f) {
        var g = f.geometry;
        var polys = g.type === "Polygon" ? [g.coordinates] : g.coordinates;
        return { id: f.properties.location,
          d: polys.map(function (p) { return p.map(ring).join(""); }).join("") };
      }),
    };
  }

  // A map in parent. onPick(zone id) runs on a click. draw(o) colours it:
  // o.level ("zones" or "provinces"), o.values (by zone or province id),
  // o.centre, o.span (the distances below and above the centre at full
  // colour), o.log
  // (compare log values), o.label(id, value) for the tooltip, o.picked (a
  // zone or province id) and o.key (the scale's name).
  function create(parent, geo, onPick) {
    var shapes = project(geo);
    var root = h("div", { class: "placemap" }, parent);
    function draw(o) {
      root.textContent = "";
      var svg = el("svg", { viewBox: "0 0 " + W + " " + (shapes.H + 50), role: "img",
        "aria-label": "Map of health zones coloured by " + o.key }, root);
      var tf = o.log ? Math.log : function (v) { return v; };
      function shade(v) {
        if (v === undefined || v === null || isNaN(v) || (o.log && v <= 0)) return "none";
        var x = tf(v) - tf(o.centre);
        return colour(x / (x < 0 ? o.span[0] : o.span[1]));
      }
      var picked = [];
      shapes.zones.forEach(function (z) {
        var id = o.level === "zones" ? z.id : province(z.id);
        var v = o.values[id];
        var path = el("path", { d: z.d, class: "placemap-zone", fill: shade(v) }, svg);
        el("title", {}, path).textContent = o.label(id, v);
        path.addEventListener("click", function () { onPick(z.id); });
        if (id === o.picked) picked.push({ d: z.d, fill: shade(v) });
      });
      // The picked place's outer border: a wide stroke under its own fill,
      // so the borders between its zones do not show.
      picked.forEach(function (z) { el("path", { d: z.d, class: "placemap-picked" }, svg); });
      picked.forEach(function (z) {
        el("path", { d: z.d, fill: z.fill === "none" ? "var(--md-default-bg-color)" : z.fill,
          "pointer-events": "none" }, svg);
      });
      var y = shapes.H + 8, x0 = W / 2 - 120;
      for (var i = 0; i < 40; i++) {
        el("rect", { x: x0 + i * 6, y: y, width: 6, height: 8, fill: colour(-1 + i * 2 / 39) }, svg);
      }
      [-1, 0, 1].forEach(function (t) {
        var x = t * (t < 0 ? o.span[0] : o.span[1]);
        var v = o.log ? Math.exp(tf(o.centre) + x) : o.centre + x;
        el("text", { x: x0 + (t + 1) * 120, y: y + 22, "text-anchor": "middle",
          class: "explorer-tick" }, svg).textContent = (t < 0 ? "≤ " : t > 0 ? "≥ " : "") +
          +v.toPrecision(2);
      });
      el("text", { x: W / 2, y: y + 34, "text-anchor": "middle", class: "explorer-tick" },
        svg).textContent = o.key;
    }
    return { draw: draw };
  }

  // The nowcast against the recent count, from the nowcast chart's data:
  // recent is the mean of the last two complete weeks (a missing week is
  // 0) and the ratio is log((median + 1) / (recent + 1)).
  function ratios(d) {
    var complete = {};
    Object.keys(d.data).forEach(function (k) {
      d.data[k].forEach(function (p) { if (p[0] <= d.complete_through) complete[p[0]] = true; });
    });
    var last = Object.keys(complete).sort().slice(-2);
    function recent(id) {
      var weeks = {};
      (d.data[id] || []).forEach(function (p) { weeks[p[0]] = p[1]; });
      return last.reduce(function (a, w) { return a + (weeks[w] || 0); }, 0) / last.length;
    }
    // Model m's latest nowcast week and every week from the reference date.
    function weeks(m) {
      var all = {};
      Object.keys(d.q).forEach(function (k) {
        d.q[k].forEach(function (r) { if (r[0] === m) all[r[1]] = true; });
      });
      var sorted = Object.keys(all).sort();
      return sorted.filter(function (w) { return w < d.reference_date; }).slice(-1)
        .concat(sorted.filter(function (w) { return w >= d.reference_date; }));
    }
    // Quantiles (q5 to q95), recent count and ratio at a place.
    function at(id, m, w) {
      var r = (d.q[id] || []).filter(function (x) { return x[0] === m && x[1] === w; })[0];
      if (!r) return null;
      var base = recent(id);
      return { q: r.slice(2), recent: base, ratio: Math.log((r[4] + 1) / (base + 1)) };
    }
    return { last: last, recent: recent, weeks: weeks, at: at };
  }

  window.bdbvPlaceMap = { create: create, ratios: ratios, province: province };
})();
