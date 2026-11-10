"""Edge-case checks on sample files at the 2026-11-14 round.

Usage: edge.py <S> <data dir> model=samples.csv [model=samples.csv ...]

Writes `<S>/edge.csv`, one row per model, place and check:
- below reported: a data week (up to 11-07) with draws below the count
  already reported (share of draws);
- zones vs province: zone draws do not sum to the province draw;
- national vs provinces: national draw differs from the provinces' sum by
  more than the largest unassigned count seen (5 cases);
- week jump: a median step between consecutive forecast weeks (11-07 to
  12-05) of more than 2x, and 5 or more cases;
- never seeded: a zone with no case ever reported that is above 0 in more
  than 5% of draws in some forecast week;
- quantile order: 5% above the median or median above 95% (cannot happen
  with samples, kept as a check on the files).
"""

import sys
from pathlib import Path

import pandas as pd

S, DATA = Path(sys.argv[1]), Path(sys.argv[2])
FILES = dict(a.split("=", 1) for a in sys.argv[3:])
ts = pd.read_csv(S / "hub/ts.csv")
rep = ts[ts.as_of == "2026-11-08"].set_index(
    ["location", "target_end_date"]
).observation
ever = ts[ts.as_of == "2026-11-08"].groupby("location").observation.sum()
STEPS = ["2026-11-07", "2026-11-14", "2026-11-21", "2026-11-28",
         "2026-12-05"]
FORECAST = STEPS[1:]

rows = []
for model, path in FILES.items():
    d = pd.read_csv(path)
    if "output_type_id" in d.columns:
        d = d.rename(columns={"output_type_id": "sample"})
    d = d[["location", "target_end_date", "sample", "value"]]
    d["depth"] = d.location.str.count("-")
    # below reported
    data = d[d.target_end_date <= "2026-11-07"].copy()
    data["rep"] = [rep.get(k, 0) for k in
                   zip(data.location, data.target_end_date)]
    low = data.assign(b=data.value < data.rep).groupby(
        ["location", "target_end_date"]
    ).agg(share=("b", "mean"), rep=("rep", "first"), lo=("value", "min"))
    for (loc, w), r in low[low.share > 0].iterrows():
        rows.append(dict(model=model, location=loc, week=w,
                         check="below reported",
                         detail=f"{r.share:.0%} of draws below {r.rep:.0f}"
                                f" (min {r.lo:.0f})"))
    # zones vs province
    z = d[d.depth == 2].assign(prov=lambda x: x.location.str[:5])
    zs = z.groupby(["prov", "target_end_date", "sample"]).value.sum()
    pv = d[d.depth == 1].set_index(
        ["location", "target_end_date", "sample"]
    ).value
    pv.index.names = zs.index.names
    gap = (zs - pv.reindex(zs.index)).dropna()
    bad = gap[gap.abs() > 0.5]
    for (prov, w), g in bad.groupby(level=[0, 1]):
        rows.append(dict(model=model, location=prov, week=w,
                         check="zones vs province",
                         detail=f"{len(g)} draws off, up to {g.abs().max():.0f}"))
    # national vs provinces
    ps = d[d.depth == 1].groupby(["target_end_date", "sample"]).value.sum()
    nat = d[d.depth == 0].set_index(["target_end_date", "sample"]).value
    gap = (nat - ps.reindex(nat.index)).dropna()
    for w, g in gap.groupby(level=0):
        if g.abs().max() > 5:
            rows.append(dict(model=model, location="CD", week=w,
                             check="national vs provinces",
                             detail=f"national minus provinces: median "
                                    f"{g.median():.0f}, range "
                                    f"{g.min():.0f} to {g.max():.0f}"))
    q = d.groupby(["location", "target_end_date"]).value.quantile(
        [0.05, 0.5, 0.95]
    ).unstack()
    q.columns = ["q05", "q50", "q95"]
    for loc, g in q.groupby(level=0):
        g = g.droplevel(0)
        m = g.q50.reindex(STEPS)
        for a, b in zip(STEPS[:-1], STEPS[1:], strict=True):
            lo, hi = sorted([m[a], m[b]])
            if hi > 2 * max(lo, 1) and hi - lo >= 5:
                rows.append(dict(model=model, location=loc, week=b,
                                 check="week jump",
                                 detail=f"median {m[a]:.0f} -> {m[b]:.0f}"))
        if ((g.q05 > g.q50) | (g.q50 > g.q95)).any():
            rows.append(dict(model=model, location=loc, week="",
                             check="quantile order", detail="crossing"))
    # never seeded
    fc = d[(d.depth == 2) & d.target_end_date.isin(FORECAST)]
    pos = fc.assign(p=fc.value > 0).groupby(
        ["location", "target_end_date"]
    ).agg(share=("p", "mean"), top=("value", "max"))
    for (loc, w), r in pos.iterrows():
        if ever.get(loc, 0) == 0 and r.share > 0.05:
            rows.append(dict(model=model, location=loc, week=w,
                             check="never seeded",
                             detail=f"{r.share:.0%} of draws above 0, "
                                    f"max {r.top:.0f}"))

out = pd.DataFrame(rows)
out.to_csv(S / "edge.csv", index=False)
print(out.groupby(["model", "check"]).size().unstack(fill_value=0)
      .to_string())
