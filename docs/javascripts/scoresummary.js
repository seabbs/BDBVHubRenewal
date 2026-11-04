// Score summary: per-forecast scores (summary_rows) summarised in the
// browser for the selected scale, round and models, as a score chart
// (scorechart.js) and a table. data-view="overall" draws one group of bars
// for the scale; data-view="place" draws one group per place, and a place
// picker narrows the table.
(function () {
  "use strict";
  var h = window.bdbvToggles.h;
  var summarise = window.bdbvSummarise;
  var summariseEnergy = window.bdbvSummariseEnergy;

  function fmt(v, dp) { return v === undefined || isNaN(v) ? "" : v.toFixed(dp); }
  // Lower-case the first letter of a label, but not of an acronym (WIS).
  function lower(s) {
    return /^[A-Z]{2}/.test(s) ? s : s.charAt(0).toLowerCase() + s.slice(1);
  }

  function init(root) {
    fetch(root.dataset.src)
      .then(function (r) { return r.json(); })
      .then(function (d) { build(root, d, root.dataset.view || "overall"); })
      .catch(function () { root.textContent = "The score data could not be loaded."; });
  }

  function build(root, d, view) {
    root.textContent = "";
    var byPlace = view === "place";
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
    var chartDiv = h("div", {}, root);
    var table = h("div", { class: "explorer-table" }, root);

    // Chart metrics: a relative metric shows the model over the baseline.
    var metrics = d.metrics.map(function (m) {
      return m.relative ?
        { key: "rel_" + m.key, label: "Relative " + lower(m.label), ref: 1 } :
        { key: m.key, label: m.label, ref: m.ref };
    });
    function value(s, m) { return m.relative ? s.rel[m.key] : s.mean[m.key]; }
    // Energy scores: one per scale, so overall view only.
    var energy = !byPlace && d.energy ? d.energy.metrics : [];
    energy.forEach(function (m) {
      metrics.push({ key: "rel_" + m.key, label: "Relative " + lower(m.label), ref: 1 });
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
    function keep(placeSet) {
      var round = selRound.value === "all" ? null : +selRound.value;
      return function (r) {
        return placeSet[r[3]] && (round === null || r[1] === round);
      };
    }
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

    var cols = [];
    d.metrics.forEach(function (m) {
      cols.push({ label: m.label, get: function (s) { return s.mean[m.key]; } });
      if (m.relative) {
        cols.push({ label: "Relative " + lower(m.label), get: function (s) { return s.rel[m.key]; } });
      }
    });
    var jointCols = [{ label: "n energy", dp: 0, get: function (e) { return e.n; } }];
    energy.forEach(function (m) {
      jointCols.push({ label: m.label, dp: 3, get: function (e) { return e.mean[m.key]; } });
      jointCols.push({ label: "Relative " + lower(m.label), dp: 3, get: function (e) { return e.rel[m.key]; } });
    });
    function drawTable(_, on) {
      var chosen = byPlace && selPlace.value !== "all" ?
        [{ i: +selPlace.value }] : places();
      var s = summarise(d, keep(set(chosen)));
      var j = energy.length ? joint() : null;
      table.textContent = "";
      if (!s.length) {
        h("p", { class: "explorer-note" }, table, "No scores for this selection.");
        return;
      }
      var tbl = h("table", {}, table);
      var head = h("tr", {}, h("thead", {}, tbl));
      ["Model", "n"].concat(cols.map(function (c) { return c.label; }))
        .concat(j ? jointCols.map(function (c) { return c.label; }) : [])
        .forEach(function (c) { h("th", {}, head, c); });
      var body = h("tbody", {}, tbl);
      s.forEach(function (r) {
        if (!on[r.model]) return;
        var tr = h("tr", {}, body);
        h("td", {}, tr, d.models[r.model].name);
        h("td", {}, tr, String(r.n));
        cols.forEach(function (c) { h("td", {}, tr, fmt(c.get(r), 3)); });
        if (j) {
          jointCols.forEach(function (c) {
            h("td", {}, tr, j[r.model] ? fmt(c.get(j[r.model]), c.dp) : "");
          });
        }
      });
    }

    fillPlaces();
    var first = chartRows();
    var chart = window.bdbvScoreChart.build(chartDiv, {
      kind: "bar",
      x_label: byPlace ? "Place" : "Scale",
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
