// Summarise the embedded energy scores for a pool and round, as
// relative_energy does in Python: each model over the origins and releases
// the baseline also has, with n, the mean of each energy score and the
// model's mean over the baseline's. There is one score per model, origin,
// release and pool, so this serves the overall view only.
(function (root) {
  "use strict";

  // d.energy.rows: [model, origin, release, pool, ...one value per metric]
  // keep(row) selects the rows; returns one entry per model with rows.
  function summariseEnergy(d, keep) {
    var e = d.energy;
    var nm = e.metrics.length;
    var base = {};
    function key(r) { return r[1] + "|" + r[2] + "|" + r[3]; }
    e.rows.forEach(function (r) {
      if (r[0] === d.baseline && keep(r)) base[key(r)] = r;
    });
    var acc = {};
    e.rows.forEach(function (r) {
      var b = base[key(r)];
      if (!keep(r) || !b) return;
      var a = acc[r[0]];
      if (!a) {
        a = acc[r[0]] = { n: 0, sum: [], bsum: [] };
        for (var j = 0; j < nm; j++) { a.sum.push(0); a.bsum.push(0); }
      }
      a.n += 1;
      for (var i = 0; i < nm; i++) {
        a.sum[i] += r[4 + i];
        a.bsum[i] += b[4 + i];
      }
    });
    var out = {};
    Object.keys(acc).forEach(function (m) {
      var a = acc[m];
      var s = { n: a.n, mean: {}, rel: {} };
      e.metrics.forEach(function (met, i) {
        s.mean[met.key] = a.sum[i] / a.n;
        s.rel[met.key] = a.sum[i] / a.bsum[i];
      });
      out[m] = s;
    });
    return out;
  }

  if (typeof module !== "undefined" && module.exports) module.exports = summariseEnergy;
  else root.bdbvSummariseEnergy = summariseEnergy;
})(this);
