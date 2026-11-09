// Checks chart: the backtest scores by place type or flag (the report's
// checks JSON), drawn with scorechart.js, with a target (data-targets), a
// scale, a split and a baseline picker. The JSON has a set per baseline,
// scale and split. The score history (scripts/score_history.py) reuses it
// with `kind`, `x_label`, one split, whose picker is hidden, and `marks`,
// the x positions scorechart.js draws a vertical line at.
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
    var splitLabel = h("label", {}, controls, "Split by ");
    var selSplit = h("select", {}, splitLabel);
    if (d.splits.length < 2) splitLabel.hidden = true;
    d.splits.forEach(function (s) { h("option", { value: s }, selSplit, s); });
    var rel = T.baselineSelect(controls, function () { update(); });
    rel.fill(d, base);
    var note = h("p", { class: "explorer-note" }, root);
    var chartDiv = h("div", {}, root);
    // Relative WIS is the first metric, missing where the baseline has no
    // forecast at the scale.
    function pick() {
      var k = rel.name() + "|" + selScale.value + "|" + selSplit.value;
      var set = d.sets[k] || { x: [], rows: [] };
      var none = !set.rows.some(function (r) { return r[3] !== null; });
      note.textContent = "Relative WIS divides by " + rel.name() + " on the same scored weeks." +
        (none ? " " + rel.name() + " has no forecasts at this scale, so there is none here." : "");
      return set;
    }
    var first = pick();
    var chart = window.bdbvScoreChart.build(chartDiv, {
      kind: d.kind || "bar", x_label: d.x_label || "Group", x: first.x,
      models: d.models, metrics: d.metrics, rows: first.rows, marks: first.marks,
    });
    function update() {
      var s = pick();
      chart.update(s.rows, s.x, s.marks);
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
