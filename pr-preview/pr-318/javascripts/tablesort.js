// Table sort: a click on a column header orders the rows by that column,
// ascending, then descending on a second click. Numbers sort as numbers,
// text as text and missing cells (empty, nan, NA or a dash) go last either
// way. Ties keep the order the table opened in. A cell's `data-sort`, where
// given, is what it sorts by. Tables in a `.tablesort` block are made
// sortable on load; scripts that build a table call
// bdbvTableSort.sortable(table, state), with state ({col, dir}) kept by the
// caller so a rebuilt table keeps the viewer's order.
(function (root) {
  "use strict";

  var MISSING = ["", "nan", "na", "–", "—", "-"];

  // A key as [kind, value]: 0 a number, 1 text, 2 missing.
  function classify(s) {
    var t = String(s).trim();
    if (MISSING.indexOf(t.toLowerCase()) >= 0) return [2, null];
    var n = Number(t.replace(/,/g, ""));
    if (t !== "" && !isNaN(n)) return [0, n];
    return [1, t];
  }

  // The row order for keys (one per row) in direction dir (1 or -1).
  function order(keys, dir) {
    var ks = keys.map(classify);
    var idx = keys.map(function (_, i) { return i; });
    idx.sort(function (a, b) {
      var x = ks[a], y = ks[b];
      if (x[0] === 2 || y[0] === 2) {
        if (x[0] !== y[0]) return x[0] === 2 ? 1 : -1;
        return a - b;
      }
      var c;
      if (x[0] !== y[0]) c = x[0] - y[0];
      else if (x[0] === 0) c = x[1] - y[1];
      else c = x[1].localeCompare(y[1]);
      return c * dir || a - b;
    });
    return idx;
  }

  function key(cell) {
    if (!cell) return "";
    return cell.dataset.sort !== undefined ? cell.dataset.sort : cell.textContent;
  }

  // Make the headers of table sort its body rows. state ({col, dir}), if
  // given, is updated on each click and applied now if it names a column.
  function sortable(table, state) {
    if (!table || table.dataset.sortable) return;
    table.dataset.sortable = "1";
    state = state || { col: -1, dir: 1 };
    var heads = Array.prototype.slice.call(table.querySelectorAll("thead th"));
    var body = table.tBodies[0];
    if (!body) return;
    var rows = Array.prototype.slice.call(body.rows);
    function apply() {
      heads.forEach(function (o, i) {
        o.setAttribute("aria-sort", i !== state.col ? "none" :
          state.dir > 0 ? "ascending" : "descending");
      });
      if (state.col < 0 || state.col >= heads.length) return;
      var keys = rows.map(function (r) { return key(r.cells[state.col]); });
      order(keys, state.dir).forEach(function (i) { body.appendChild(rows[i]); });
    }
    heads.forEach(function (th, col) {
      th.classList.add("tablesort-head");
      th.title = "Sort by " + th.textContent.trim();
      th.addEventListener("click", function () {
        state.dir = state.col === col ? -state.dir : 1;
        state.col = col;
        apply();
      });
    });
    apply();
  }

  var api = { classify: classify, order: order, sortable: sortable };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
    return;
  }
  root.bdbvTableSort = api;

  function start() {
    document.querySelectorAll(".tablesort table").forEach(function (t) { sortable(t); });
  }
  if (typeof document$ !== "undefined") document$.subscribe(start);
  else document.addEventListener("DOMContentLoaded", start);
})(this);
