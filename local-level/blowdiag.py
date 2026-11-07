import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, sys.argv[1]); import checks
from bdbvhubrenewal.forecasters import chain
d = list(chain.Settings.__init__.__defaults__); d[-2], d[-1] = 0.05, 5.0
chain.Settings.__init__.__defaults__ = tuple(d)
from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.data.tidy import load_daily_zones
from bdbvhubrenewal.data.zones import hub_zones, weekly_zone_counts
from bdbvhubrenewal.evaluation.backtest import make_tasks, load_config
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.forecasters.randomwalk import _reliable
from bdbvhubrenewal.forecasters.zones import zone_task, reconcile_zones
pd.set_option("display.width", 250)
cfg = load_config("config/backtest.toml"); t = load_target(); real = t[t.kind == "real"]
daily = load_daily(); dz = load_daily_zones(); delays = load_delays(); zones = hub_zones(Path("data/hub-config/tasks.json"))
rows = []
for v in ["2026-10-17", "2026-10-24"]:
    task = make_tasks(real[real.vintage == v], delays, cfg, daily=daily)[0]
    counts = weekly_zone_counts(dz, v)
    active = tuple(z for z in zones if z in set(counts.location))
    zt = zone_task(task, counts, active, pd.concat([daily, dz]))
    m = get_model("randomwalknowcast")
    direct = m.forecast(zt); prov = m.forecast(task)
    rec = reconcile_zones(direct, prov, task, counts, active)
    o = m.origin_at(zt, errors=True); snap = o.snap; L = len(snap.locations)
    cw = snap.cut_week(o.o); nd = snap.n_weeks
    rel = _reliable(o) & (np.arange(nd) <= cw)
    w0 = np.where(rel.any(1), nd - 1 - np.argmax(rel[:, ::-1], 1), o.start); w0 = np.where(o.silent, o.start, w0)
    ref = task.reference_date.isoformat()
    cz = counts.assign(vintage=v)
    q = rec[rec.target_end_date == ref].groupby("location").value.quantile([.5, .95]).unstack()
    qd = direct[direct.target_end_date == ref].groupby("location").value.quantile([.5, .95]).unstack()
    qq = pd.DataFrame({"model": "randomwalknowcast", "origin": v, "location": q.index, "q50": q[.5].values, "q95": q[.95].values})
    fl = checks.flags(cz, qq).set_index("location").flag
    refi = list(snap.week_ends).index(ref) if ref in snap.week_ends else nd
    for i, z in enumerate(snap.locations):
        if z.count("-") != 2: continue
        rep = o.reported[i]
        rows.append(dict(origin=v, zone=z, flag=fl.get(z, "none"), silent=bool(o.silent[i]), fallback=bool(not rel[i].any()),
            w0=snap.week_ends[w0[i]], weeks_back=int(cw - w0[i]), partial_scaled=bool(w0[i] == cw), rep_w0=rep[w0[i]],
            point_w0=round(o.point[i, w0[i]], 1), sigma=round(o.sigma[i], 2), steps=int(refi - w0[i]),
            recent_max=rep[max(0, cw - 4):cw + 1].max(), d50=qd.loc[z, .5] if z in qd.index else np.nan, d95=qd.loc[z, .95] if z in qd.index else np.nan,
            r50=q.loc[z, .5] if z in q.index else np.nan, r95=q.loc[z, .95] if z in q.index else np.nan))
df = pd.DataFrame(rows); df.to_csv(sys.argv[2], index=False)
b = df[df.flag == "blow-up"]
print("zone forecasts", len(df), "blow-ups", len(b), "collapse", (df.flag == "collapse").sum())
print("blow-ups: silent", b.silent.sum(), "fallback (no week with 5% and 5 cases)", b.fallback.sum(), "partial-week scaled", b.partial_scaled.sum())
print(b[["weeks_back", "steps", "sigma", "rep_w0", "point_w0", "recent_max", "d50", "d95", "r50", "r95"]].describe().round(2).to_string())
print(b.sort_values("r95", ascending=False).head(12).to_string(index=False))
