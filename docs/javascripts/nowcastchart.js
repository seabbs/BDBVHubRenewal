// Nowcast chart: pick a scale and a place (for a zone, its province first)
// to see each model's latest nowcast and forecast over the reported counts,
// as bands or trajectories. Model tick boxes come from toggles.js.
// data-scale is the scale shown first. bdbvNowcastCharts[element id](place)
// picks a place in that chart and scrolls it into view. The map of
// `data-map` (placemap.js), shown at first with data-map-on, colours places
// by one model's nowcast against the recent count, and beside it a panel
// gives the numbers for the place picked.
(function () {
  "use strict";
  var DAY = 864e5;
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
    "Oct", "Nov", "Dec"];
  var T = window.bdbvToggles;
  var Z = window.bdbvZoom;
  var PM = window.bdbvPlaceMap;
  var h = T.h;
  var el = T.el;

  function day(s) { return Date.parse(s + "T00:00:00Z"); }
  function fmt(v, d) { return v === null || isNaN(v) ? "" : v.toFixed(d); }

  function init(root) {
    T.getJSON(root.dataset.src)
      .then(function (d) { build(root, d); })
      .catch(function () { root.textContent = "The nowcast data could not be loaded."; });
  }

  function build(root, d) {
    root.textContent = "";
    var controls = h("div", { class: "explorer-controls" }, root);
    var scales = [["national", "National"], ["provinces", "Provinces"], ["zones", "Health zones"]]
      .filter(function (s) { return d.places.some(function (p) { return p.scale === s[0]; }); });
    var selScale = h("select", {}, h("label", {}, controls, "Scale "));
    scales.forEach(function (s) { h("option", { value: s[0] }, selScale, s[1]); });
    if (root.dataset.scale) selScale.value = root.dataset.scale;
    var provLabel = h("label", {}, controls, "Province ");
    var selProv = h("select", {}, provLabel);
    d.places.filter(function (p) { return p.scale === "provinces"; }).forEach(function (p) {
      if (d.places.some(function (z) { return z.province === p.id; })) {
        h("option", { value: p.id }, selProv, p.name);
      }
    });
    var selZone = h("select", {}, h("label", {}, controls, "Place "));
    var view = T.viewSelect(controls, true, function () { draw(); });
    var zoom = Z.create(controls, function () { draw(); });
    T.fullscreen(controls, root);
    var mapOn = T.mapToggle(controls, root.dataset.mapOn === "true", function () { draw(); });
    var map = null;
    var keys = T.legend(root, d.models, function () { draw(); });
    var note = h("p", { class: "explorer-note" }, root);
    var panes = h("div", { class: "chart-map" }, root);
    var figure = h("div", { class: "explorer-figure" }, panes);
    var side = h("div", { class: "chart-map-side" }, panes);
    var mapControls = h("div", { class: "explorer-controls" }, side);
    var mapModel = h("select", {}, h("label", {}, mapControls, "Map "));
    d.models.forEach(function (m, i) { if (m.fan) h("option", { value: i }, mapModel, m.name); });
    var mapWeek = h("select", {}, h("label", {}, mapControls, "Week "));
    var R = PM.ratios(d);
    var mapHolder = h("div", {}, side);
    var info = h("dl", { class: "chart-map-info" }, side);
    var names = {};
    d.places.forEach(function (p) { names[p.id] = p; });
    function fillWeeks() {
      mapWeek.textContent = "";
      R.weeks(+mapModel.value).forEach(function (w) {
        h("option", { value: w }, mapWeek, (w < d.reference_date ? "nowcast, " : "") + w);
      });
    }
    mapModel.addEventListener("change", function () { fillWeeks(); draw(); });
    mapWeek.addEventListener("change", draw);

    function fillZones() {
      var zones = selScale.value === "zones";
      provLabel.hidden = !zones;
      selZone.textContent = "";
      d.places.filter(function (p) {
        return p.scale === selScale.value && (!zones || p.province === selProv.value);
      }).forEach(function (p) {
        h("option", { value: p.id }, selZone, p.name);
      });
    }
    selScale.addEventListener("change", function () { fillZones(); draw(); });
    selProv.addEventListener("change", function () { fillZones(); draw(); });
    selZone.addEventListener("change", draw);

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
      // A place with no cases has no data, so its axis starts 12 weeks back.
      var x0 = (data.length ? day(data[0][0]) : ref - 84 * DAY) - 4 * DAY;
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
      var W = 760, H = 300, L = 44, mRight = 10, Tp = 10, B = 28;
      function X(t) { return L + (t - x0) / (x1 - x0) * (W - L - mRight); }
      function Y(v) { return Tp + (1 - (v - y0) / (y1 - y0)) * (H - Tp - B); }
      figure.textContent = "";
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
        "aria-label": "Nowcast and forecast for " + zone }, figure);
      var span = y1 - y0;
      // Whole counts, so the step is at least 1.
      var step = Math.max(1, Z.nice(span));
      for (var v = Math.ceil(y0 / step) * step; v <= y1; v += step) {
        el("line", { x1: L, x2: W - mRight, y1: Y(v), y2: Y(v), class: "explorer-grid" }, svg);
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
      var plot = Z.clip(svg, { L: L, R: mRight, T: Tp, B: B, W: W, H: H });
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
      Z.attach(zoom, svg, { L: L, R: mRight, T: Tp, B: B, W: W, H: H, x: [x0, x1], y: [y0, y1] });
      drawMap(zone);
    }

    // The map at zone level for zones, else at province level, and the
    // numbers for the place picked.
    function drawMap(place) {
      side.hidden = !mapOn.checked || !map;
      if (side.hidden) return;
      var m = +mapModel.value, w = mapWeek.value;
      var level = selScale.value === "zones" ? "zones" : "provinces";
      var values = {};
      d.places.forEach(function (p) {
        var r = p.scale === level && R.at(p.id, m, w);
        if (r) values[p.id] = r.ratio;
      });
      map.draw({ level: level, values: values, centre: 0, span: [2, 2], log: false,
        picked: place, key: "log((median + 1) / (recent + 1)), " + d.models[m].name + ", " + w,
        label: function (id, v) {
          return (names[id] ? names[id].name : id) + (v === undefined ? ": no forecast" :
            ": ratio " + v.toFixed(2));
        } });
      info.textContent = "";
      var p = names[place], r = R.at(place, m, w);
      var rows = [["Place", p.name + (p.scale === "zones" ? " (" + p.id + ")" : "")]];
      if (p.province) rows.push(["Province", names[p.province] ? names[p.province].name : p.province]);
      rows.push(["Recent", R.recent(place).toFixed(1) + " (weeks ending " + R.last.join(" and ") + ")"]);
      rows.push(r ? ["Median, " + w, Math.round(r.q[2]) + " (90% " + Math.round(r.q[0]) +
        " to " + Math.round(r.q[4]) + ")"] : ["Median, " + w, "no forecast"]);
      if (r) rows.push(["Ratio", r.ratio.toFixed(2)]);
      rows.forEach(function (row) {
        h("dt", {}, info, row[0]);
        h("dd", {}, info, row[1]);
      });
    }

    function show(id) {
      var p = d.places.filter(function (x) { return x.id === id; })[0];
      if (!p) return;
      selScale.value = p.scale;
      if (p.province) selProv.value = p.province;
      fillZones();
      selZone.value = id;
      draw();
    }
    window.bdbvNowcastCharts = window.bdbvNowcastCharts || {};
    window.bdbvNowcastCharts[root.id] = function (id) {
      show(id);
      root.scrollIntoView({ behavior: "smooth", block: "start" });
    };
    fillZones();
    fillWeeks();
    draw();
    // The map loads on its own, so the chart works without it.
    T.getJSON(root.dataset.map).then(function (geo) {
      map = PM.create(mapHolder, geo, function (zone) {
        show(selScale.value === "zones" ? zone : PM.province(zone));
      });
      draw();
    }).catch(function () { mapOn.parentNode.hidden = true; });
    T.onTheme(draw);
  }

  function start() {
    document.querySelectorAll(".nowcastchart[data-src]").forEach(init);
  }
  if (typeof document$ !== "undefined") document$.subscribe(start);
  else document.addEventListener("DOMContentLoaded", start);
})();
