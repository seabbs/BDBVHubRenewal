// Checks chart: the backtest scores by place type or flag (the report's
// checks JSON), drawn with scorechart.js, with a scale and a split picker.
(function () {
  "use strict";
  var h = window.bdbvToggles.h;

  function build(root, d) {
    root.textContent = "";
    var controls = h("div", { class: "explorer-controls" }, root);
    var selScale = h("select", {}, h("label", {}, controls, "Scale "));
    d.scales.forEach(function (s) { h("option", { value: s }, selScale, s); });
    var selSplit = h("select", {}, h("label", {}, controls, "Split by "));
    d.splits.forEach(function (s) { h("option", { value: s }, selSplit, s); });
    h("p", { class: "explorer-note" }, root,
      "Relative WIS divides by randomwalk on the same scored weeks.");
    var chartDiv = h("div", {}, root);
    function pick() {
      return d.sets[selScale.value + "|" + selSplit.value] || { x: [], rows: [] };
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

  function init(root) {
    fetch(root.dataset.src)
      .then(function (r) { return r.json(); })
      .then(function (d) { build(root, d); })
      .catch(function () { root.textContent = "The checks data could not be loaded."; });
  }

  function start() {
    document.querySelectorAll(".checkschart[data-src]").forEach(init);
  }
  if (typeof document$ !== "undefined") document$.subscribe(start);
  else document.addEventListener("DOMContentLoaded", start);
})();
