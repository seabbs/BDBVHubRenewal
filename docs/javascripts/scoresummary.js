// Score summary: per-forecast scores (summary_rows) summarised in the
// browser for the selected scale, round and models, as a score chart
// (scorechart.js) and a table. data-view="overall" draws one group of bars
// for the scale; data-view="place" draws one group per place, and a place
// picker narrows the table; data-view="horizon" draws a line per model by
// horizon.
(function () {
  "use strict";
  var T = window.bdbvToggles;
  var h = T.h;
  var summarise = window.bdbvSummarise;
  var summariseEnergy = window.bdbvSummariseEnergy;

  function fmt(v, dp) { return v === null || v === undefined || isNaN(v) ? "–" : v.toFixed(dp); }
  // Lower-case the first letter of a label, but not of an acronym (WIS).
  function lower(s) {
    return /^[A-Z]{2}/.test(s) ? s : s.charAt(0).toLowerCase() + s.slice(1);
  }

  // Tabulate summaries s of data d in container, for the models isOn(m):
  // n, the mean of each metric and, for relative metrics, the relative one.
  // extra columns ({label, dp, get(e)}) read extraFor(model), if given.
  function summaryTable(container, d, s, isOn, extra, extraFor) {
    extra = extra || [];
    if (!s.length) {
      h("p", { class: "explorer-note" }, container,
        "No scores yet for this selection. Rounds are scored once later data are published.");
      return;
    }
    var cols = [];
    d.metrics.forEach(function (m) {
      cols.push({ label: m.label, get: function (x) { return x.mean[m.key]; } });
      if (m.relative) {
        cols.push({ label: "Relative " + lower(m.label),
          get: function (x) { return x.rel[m.key]; } });
      }
    });
    var tbl = h("table", {}, container);
    var head = h("tr", {}, h("thead", {}, tbl));
    ["Model", "n"].concat(cols.map(function (c) { return c.label; }))
      .concat(extra.map(function (c) { return c.label; }))
      .forEach(function (c) { h("th", {}, head, c); });
    var body = h("tbody", {}, tbl);
    s.forEach(function (r) {
      if (!isOn(r.model)) return;
      var tr = h("tr", {}, body);
      h("td", {}, tr, d.models[r.model].name);
      h("td", {}, tr, String(r.n));
      cols.forEach(function (c) { h("td", {}, tr, fmt(c.get(r), 3)); });
      var e = extra.length ? extraFor(r.model) : null;
      extra.forEach(function (c) { h("td", {}, tr, e ? fmt(c.get(e), c.dp) : "–"); });
    });
  }
  window.bdbvSummaryTable = summaryTable;

  function init(root) {
    var get = function (src) { return fetch(src).then(function (r) { return r.json(); }); };
    Promise.all([get(root.dataset.src), root.dataset.map ? get(root.dataset.map) : null])
      .then(function (both) { build(root, both[0], root.dataset.view || "overall", both[1]); })
      .catch(function () { root.textContent = "The score data could not be loaded."; });
  }

  // geo, for the place view, adds a map of the chosen metric for one model.
  function build(root, d, view, geo) {
    root.textContent = "";
    var byPlace = view === "place";
    var byHorizon = view === "horizon";
    var controls = h("div", { class: "explorer-controls" }, root);
    var selScale = h("select", {}, h("label", {}, controls, "Scale "));
    d.pools.forEach(function (p) {
      if (byPlace && p.id === "national") return;
      h("option", { value: p.id }, selScale, p.label);
    });
    var selRound = h("select", {}, h("label", {}, controls, "Round "));
    h("option", { value: "all" }, selRound, "All rounds");
    d.origins.forEach(function (o, i) { h("option", { value: i }, selRound, o); });
    var selPlace = null;
    if (byPlace) selPlace = h("select", {}, h("label", {}, controls, "Table for "));
    T.fullscreen(controls, root);
    var mapOn = geo ? T.mapToggle(controls, false, function () { drawMap(); }) : null;
    h("p", { class: "explorer-note" }, root, "Relative scores divide by " +
      d.models[d.baseline].name + " on the same forecasts.");
    var panes = h("div", { class: "chart-map" }, root);
    var chartDiv = h("div", {}, panes);
    var side = h("div", { class: "chart-map-side" }, panes);
    side.hidden = true;
    var mapModel = null, map = null;
    if (geo) {
      mapModel = h("select", {}, h("label", {}, h("div", { class: "explorer-controls" }, side), "Map "));
      d.models.forEach(function (m, i) {
        if (i !== d.baseline) h("option", { value: i }, mapModel, m.name);
      });
      // The first model ticked at first.
      var first = d.models.map(function (m, i) { return i; })
        .filter(function (i) { return i !== d.baseline && d.models[i].on; })[0];
      if (first !== undefined) mapModel.value = first;
      mapModel.addEventListener("change", function () { drawMap(); });
      map = window.bdbvPlaceMap.create(side, geo, function (zone) {
        var id = selScale.value === "zones" ? zone : window.bdbvPlaceMap.province(zone);
        var x = places().filter(function (y) { return y.p.id === id; })[0];
        if (!x) return;
        selPlace.value = x.i;
        update();
      });
    }
    var table = h("div", { class: "explorer-table" }, root);

    // Chart metrics: a relative metric shows the model over the baseline,
    // on a log axis so that places with near-zero scores do not set it.
    var metrics = d.metrics.map(function (m) {
      return m.relative ?
        { key: "rel_" + m.key, label: "Relative " + lower(m.label), ref: 1, log: true } :
        { key: m.key, label: m.label, ref: m.ref };
    });
    function value(s, m) { return m.relative ? s.rel[m.key] : s.mean[m.key]; }
    // The energy score is defined only for a whole scale, so it is left
    // out by place and by horizon.
    var energy = view === "overall" && d.energy ? d.energy.metrics : [];
    energy.forEach(function (m) {
      metrics.push({ key: "rel_" + m.key, label: "Relative " + lower(m.label), ref: 1, log: true });
    });
    function joint() {
      var pool = d.pools.map(function (p) { return p.id; }).indexOf(selScale.value);
      var round = selRound.value === "all" ? null : +selRound.value;
      return summariseEnergy(d, function (r) {
        return r[3] === pool && (round === null || r[1] === round);
      });
    }

    function places() {
      return d.places.map(function (p, i) { return { i: i, p: p }; })
        .filter(function (x) { return x.p.pool === selScale.value; });
    }
    function keep(placeSet, horizon) {
      var round = selRound.value === "all" ? null : +selRound.value;
      return function (r, i) {
        return placeSet[r[3]] && (round === null || r[1] === round) &&
          (horizon === undefined || d.horizons[i] === horizon);
      };
    }
    var horizons = d.horizons.filter(function (v, i, a) { return a.indexOf(v) === i; })
      .sort(function (a, b) { return a - b; });
    function set(list) {
      var s = {};
      list.forEach(function (x) { s[x.i] = true; });
      return s;
    }
    function fillPlaces() {
      if (!byPlace) return;
      selPlace.textContent = "";
      h("option", { value: "all" }, selPlace, "All places");
      places().forEach(function (x) { h("option", { value: x.i }, selPlace, x.p.name); });
    }

    // Rows for the chart: [model, x index, n, ...one value per metric].
    function chartRows() {
      if (byHorizon) {
        var all = set(places());
        var hrows = [];
        horizons.forEach(function (hz, xi) {
          summarise(d, keep(all, hz)).forEach(function (s) {
            hrows.push([s.model, xi, s.n].concat(d.metrics.map(function (m) {
              return value(s, m);
            })));
          });
        });
        return { rows: hrows, x: horizons };
      }
      var groups = byPlace ? places().map(function (x) { return [x]; }) : [places()];
      var x = byPlace ? groups.map(function (g) { return g[0].p.name; }) :
        [selScale.options[selScale.selectedIndex].text];
      var rows = [];
      var j = energy.length ? joint() : {};
      groups.forEach(function (g, xi) {
        summarise(d, keep(set(g))).forEach(function (s) {
          rows.push([s.model, xi, s.n].concat(d.metrics.map(function (m) {
            return value(s, m);
          }), energy.map(function (m) {
            return j[s.model] ? j[s.model].rel[m.key] : null;
          })));
        });
      });
      return { rows: rows, x: x };
    }

    var jointCols = [{ label: "n energy", dp: 0, get: function (e) { return e.n; } }];
    energy.forEach(function (m) {
      jointCols.push({ label: m.label, dp: 3, get: function (e) { return e.mean[m.key]; } });
      jointCols.push({ label: "Relative " + lower(m.label), dp: 3, get: function (e) { return e.rel[m.key]; } });
    });
    // The map of the metric shown, for the model chosen, at the scale chosen.
    var shown = 0;
    function drawMap() {
      if (!geo) return;
      side.hidden = !mapOn.checked;
      if (!mapOn.checked) return;
      var metric = metrics[shown];
      var m = +mapModel.value;
      // Full colour at 4 times or a quarter of the reference for a
      // relative score, 0.5 from it for coverage and 1 from it for bias,
      // so no one place sets the scale.
      var span = metric.log ? Math.log(4) : metric.ref === 0 ? 1 : 0.5;
      var values = {};
      places().forEach(function (x) {
        var s = summarise(d, keep(set([x]))).filter(function (r) { return r.model === m; })[0];
        var v = s ? value(s, d.metrics[shown]) : null;
        if (v === null || v === undefined || isNaN(v) || (metric.log && v <= 0)) return;
        values[x.p.id] = v;
      });
      var picked = selPlace.value === "all" ? null : d.places[+selPlace.value].id;
      map.draw({ level: selScale.value === "zones" ? "zones" : "provinces", values: values,
        centre: metric.ref, span: span, log: metric.log, picked: picked,
        key: metric.label + ", " + d.models[m].name,
        label: function (id, v) {
          var p = d.places.filter(function (y) { return y.id === id; })[0];
          return (p ? p.name : id) + (v === undefined ? ": no score" : ": " + (+v.toFixed(3)));
        } });
    }

    function drawTable(k, on) {
      shown = k;
      drawMap();
      var chosen = byPlace && selPlace.value !== "all" ?
        [{ i: +selPlace.value }] : places();
      var j = energy.length ? joint() : null;
      table.textContent = "";
      summaryTable(table, d, summarise(d, keep(set(chosen))),
        function (m) { return on[m]; },
        j ? jointCols : [], function (m) { return j && j[m]; });
    }

    fillPlaces();
    var first = chartRows();
    var chart = window.bdbvScoreChart.build(chartDiv, {
      kind: byHorizon ? "line" : "bar",
      x_label: byHorizon ? "Horizon" : byPlace ? "Place" : "Scale",
      x: first.x,
      models: d.models,
      metrics: metrics,
      rows: first.rows,
    }, drawTable);
    function update() {
      var c = chartRows();
      chart.update(c.rows, c.x);
    }
    selScale.addEventListener("change", function () { fillPlaces(); update(); });
    selRound.addEventListener("change", update);
    if (byPlace) selPlace.addEventListener("change", update);
  }

  function start() {
    document.querySelectorAll(".scoresummary[data-src]").forEach(init);
  }
  if (typeof document$ !== "undefined") document$.subscribe(start);
  else document.addEventListener("DOMContentLoaded", start);
})();
