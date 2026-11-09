// Checks chart: the backtest scores by place type or flag (the report's
// checks JSON), drawn with scorechart.js, with a target (data-targets), a
// scale, a split and a baseline picker. The JSON has a set per baseline,
// scale and split.
(function () {
  "use strict";
  var T = window.bdbvToggles;
  var h = T.h;

  function build(root, d, targets, target, base) {
    root.textContent = "";
    var controls = h("div", { class: "explorer-controls" }, root);
    T.targetSelect(controls, targets, target, function (k) { load(root, k, rel.name()); });
    var selScale = h("select", {}, h("label", {}, controls, "Scale "));
    d.scales.forEach(function (s) { h("option", { value: s }, selScale, s); });
    var selSplit = h("select", {}, h("label", {}, controls, "Split by "));
    d.splits.forEach(function (s) { h("option", { value: s }, selSplit, s); });
    var rel = T.baselineSelect(controls, function () { update(); });
    rel.fill(d, base);
    var note = h("p", { class: "explorer-note" }, root);
    var chartDiv = h("div", {}, root);
    function pick() {
      note.textContent = "Relative WIS divides by " + rel.name() + " on the same scored weeks.";
      var k = rel.name() + "|" + selScale.value + "|" + selSplit.value;
      return d.sets[k] || { x: [], rows: [] };
    }
    var first = pick();
    var chart = window.bdbvScoreChart.build(chartDiv, {
      kind: "bar", x_label: "Group", x: first.x, models: d.models,
      metrics: d.metrics, rows: first.rows,
    });
    function update() {
      var s = pick();
      chart.update(s.rows, s.x);
    }
    selScale.addEventListener("change", update);
    selSplit.addEventListener("change", update);
  }

  // Load the checks of one target (data-targets) and draw them.
  function load(root, target, base) {
    var targets = JSON.parse(root.dataset.targets);
    T.getJSON(targets[target])
      .then(function (d) { build(root, d, targets, target, base); })
      .catch(function () { root.textContent = "The checks data could not be loaded."; });
  }
  function init(root) { load(root, Object.keys(JSON.parse(root.dataset.targets))[0]); }

  function start() {
    document.querySelectorAll(".checkschart[data-targets]").forEach(init);
  }
  if (typeof document$ !== "undefined") document$.subscribe(start);
  else document.addEventListener("DOMContentLoaded", start);
})();
