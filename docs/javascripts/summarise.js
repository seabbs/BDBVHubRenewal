// Summarise per-forecast scores for a selection, as relative_wis does in
// Python: each model over the forecasts the baseline also made (same
// origin, release, place and week), with n, the mean of every metric and,
// for relative metrics, the model's mean over the baseline's. Metric
// columns are used as embedded and only averaged and divided.
(function (root) {
  "use strict";

  // d.rows: [model, origin, release, place, week, ...one value per metric]
  // d.metrics: [{key, label, ref, relative}]
  // keep(row) selects the rows; returns one entry per model with rows.
  function summarise(d, keep) {
    var nm = d.metrics.length;
    var base = {};
    function key(r) { return r[1] + "|" + r[2] + "|" + r[3] + "|" + r[4]; }
    d.rows.forEach(function (r) {
      if (r[0] === d.baseline && keep(r)) base[key(r)] = r;
    });
    var acc = {};
    d.rows.forEach(function (r) {
      if (!keep(r)) return;
      var b = base[key(r)];
      if (!b) return;
      var a = acc[r[0]];
      if (!a) {
        a = acc[r[0]] = { n: 0, sum: [], bsum: [] };
        for (var j = 0; j < nm; j++) { a.sum.push(0); a.bsum.push(0); }
      }
      a.n += 1;
      for (var i = 0; i < nm; i++) {
        a.sum[i] += r[5 + i];
        a.bsum[i] += b[5 + i];
      }
    });
    return Object.keys(acc).map(Number).sort(function (x, y) { return x - y; })
      .map(function (m) {
        var a = acc[m];
        var out = { model: m, n: a.n, mean: {}, rel: {} };
        d.metrics.forEach(function (met, i) {
          out.mean[met.key] = a.sum[i] / a.n;
          // A zero baseline gives no relative score, as in Python.
          if (met.relative) out.rel[met.key] = a.bsum[i] > 0 ? a.sum[i] / a.bsum[i] : null;
        });
        return out;
      });
  }

  if (typeof module !== "undefined" && module.exports) module.exports = summarise;
  else root.bdbvSummarise = summarise;
})(this);
