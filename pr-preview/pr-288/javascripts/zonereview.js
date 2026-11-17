// Zone review: every place's recent reported level beside one model's
// latest nowcast and forecast weeks, from the nowcast chart's data, so
// implausible places are easy to find. Recent and the ratio
// log((median + 1) / (recent + 1)) for the week chosen are as the place
// map takes them (placemap.js). Rows sort and filter with tablefilter.js,
// and a place opens in the nowcast chart whose element id is `data-chart`
// (nowcastchart.js).
(function () {
  "use strict";
  var T = window.bdbvToggles;
  var h = T.h;
  var SCALES = { national: "National", provinces: "Province", zones: "Health zone" };

  function init(root) {
    fetch(root.dataset.src)
      .then(function (r) { return r.json(); })
      .then(function (d) { build(root, d); })
      .catch(function () { root.textContent = "The zone review data could not be loaded."; });
  }

  function build(root, d) {
    // Open a place in the linked chart, once that chart has loaded.
    function show(id) {
      var charts = window.bdbvNowcastCharts || {};
      if (charts[root.dataset.chart]) charts[root.dataset.chart](id);
    }
    root.textContent = "";
    var R = window.bdbvPlaceMap.ratios(d);
    var names = {};
    d.places.forEach(function (p) { names[p.id] = p.name; });

    var controls = h("div", { class: "explorer-controls" }, root);
    var selModel = h("select", {}, h("label", {}, controls, "Model "));
    d.models.forEach(function (m, i) {
      if (m.fan) h("option", { value: i }, selModel, m.name);
    });
    var selWeek = h("select", {}, h("label", {}, controls, "Ratio for "));
    var holder = h("div", {}, root);

    function fillWeeks() {
      selWeek.textContent = "";
      R.weeks(+selModel.value).forEach(function (w) {
        h("option", { value: w }, selWeek,
          (w < d.reference_date ? "nowcast, week ending " : "week ending ") + w);
      });
    }

    function draw() {
      var m = +selModel.value;
      var weeks = R.weeks(m);
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
        var at = weeks.map(function (w) { return R.at(p.id, m, w); });
        if (!at.some(Boolean)) return;
        var tr = h("tr", {}, body);
        var b = h("button", { type: "button", class: "zonereview-link" },
          h("td", {}, tr), p.scale === "zones" ? p.name + " (" + p.id + ")" : p.name);
        b.addEventListener("click", function () { show(p.id); });
        h("td", {}, tr, SCALES[p.scale]);
        h("td", {}, tr, p.province ? names[p.province] || p.province : "");
        h("td", {}, tr, R.recent(p.id).toFixed(1));
        at.forEach(function (r) {
          h("td", { "data-sort": r ? String(r.q[2]) : "" }, tr, r ? Math.round(r.q[2]) +
            " (" + Math.round(r.q[0]) + " to " + Math.round(r.q[4]) + ")" : "");
        });
        var r = R.at(p.id, m, selWeek.value);
        var cls = !r ? "" : r.ratio >= Math.LN2 ? "zonereview-up" :
          r.ratio <= -Math.LN2 ? "zonereview-down" : "";
        h("td", { "data-sort": r ? String(r.ratio) : "", class: cls }, tr,
          r ? r.ratio.toFixed(2) : "");
      });
      window.bdbvTableFilter(block);
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
  }

  function start() {
    document.querySelectorAll(".zonereview[data-src]").forEach(init);
  }
  if (typeof document$ !== "undefined") document$.subscribe(start);
  else document.addEventListener("DOMContentLoaded", start);
})();
