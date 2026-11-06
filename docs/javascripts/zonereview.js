// Zone review: every place's recent reported level beside one model's
// latest nowcast and forecast weeks, from the nowcast chart's data, so
// implausible places are easy to find. The ratio is
// log((median + 1) / (recent + 1)) for the week chosen, with recent the
// mean of the last two complete weeks. Rows sort and filter with
// tablefilter.js, and a place opens in the nowcast chart (nowcastchart.js).
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
    var holder = h("div", {}, root);

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
          if (window.bdbvNowcastShow) window.bdbvNowcastShow(p.id);
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
