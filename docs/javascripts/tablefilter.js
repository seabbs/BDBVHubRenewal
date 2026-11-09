// Table filter: a text search, a select per named column and sortable
// headers for each table inside a `.tablefilter` block. data-defaults
// (column to value) sets the value a select opens on. A cell's
// `data-sort` value, where given, is what it sorts by. Without JavaScript
// the table is a plain table. Scripts that build a table call
// bdbvTableFilter(block).
(function () {
  "use strict";
  var h = window.bdbvToggles.h;

  function text(cell) { return cell.textContent.trim(); }
  function key(cell) {
    return cell.dataset.sort !== undefined ? cell.dataset.sort : text(cell);
  }
  function num(s) { return s === "" ? NaN : Number(s); }

  function init(root) {
    var table = root.querySelector("table");
    if (!table || root.dataset.ready) return;
    root.dataset.ready = "1";
    var heads = Array.prototype.slice.call(table.querySelectorAll("thead th"));
    var names = heads.map(text);
    var body = table.tBodies[0];
    var rows = Array.prototype.slice.call(body.rows);
    var controls = h("div", { class: "explorer-controls tablefilter-controls" });
    root.insertBefore(controls, root.firstChild);
    var search = h("input", { type: "search", placeholder: "Search" },
      h("label", {}, controls, "Search "));
    var filters = (root.dataset.filters || "").split(",").filter(Boolean)
      .map(function (name) {
        var col = names.indexOf(name);
        var sel = h("select", {}, h("label", {}, controls, name + " "));
        h("option", { value: "" }, sel, "All");
        var seen = [];
        rows.forEach(function (r) {
          var v = text(r.cells[col]);
          if (seen.indexOf(v) < 0) seen.push(v);
        });
        seen.sort().forEach(function (v) { h("option", { value: v }, sel, v); });
        sel.addEventListener("change", apply);
        return { col: col, sel: sel };
      });
    var defaults = root.dataset.defaults ? JSON.parse(root.dataset.defaults) : {};
    filters.forEach(function (f) {
      var v = defaults[names[f.col]];
      var has = Array.prototype.some.call(f.sel.options, function (o) { return o.value === v; });
      if (has) f.sel.value = v;
    });
    var count = h("span", { class: "tablefilter-count" }, controls);
    search.addEventListener("input", apply);

    function apply() {
      var q = search.value.toLowerCase();
      var shown = 0;
      rows.forEach(function (r) {
        var ok = (!q || r.textContent.toLowerCase().indexOf(q) >= 0) &&
          filters.every(function (f) {
            return !f.sel.value || text(r.cells[f.col]) === f.sel.value;
          });
        r.hidden = !ok;
        if (ok) shown += 1;
      });
      count.textContent = shown + " of " + rows.length + " rows";
    }

    var sorted = { col: -1, dir: 1 };
    heads.forEach(function (th, col) {
      th.classList.add("tablefilter-sort");
      th.setAttribute("aria-sort", "none");
      th.addEventListener("click", function () {
        sorted.dir = sorted.col === col ? -sorted.dir : 1;
        sorted.col = col;
        heads.forEach(function (o) { o.setAttribute("aria-sort", "none"); });
        th.setAttribute("aria-sort", sorted.dir > 0 ? "ascending" : "descending");
        rows.sort(function (a, b) {
          var x = key(a.cells[col]), y = key(b.cells[col]);
          var nx = num(x), ny = num(y);
          var c = isNaN(nx) || isNaN(ny) ? x.localeCompare(y) : nx - ny;
          return c * sorted.dir;
        });
        rows.forEach(function (r) { body.appendChild(r); });
      });
    });
    apply();
  }

  window.bdbvTableFilter = init;

  function start() {
    document.querySelectorAll(".tablefilter").forEach(init);
  }
  if (typeof document$ !== "undefined") document$.subscribe(start);
  else document.addEventListener("DOMContentLoaded", start);
})();
