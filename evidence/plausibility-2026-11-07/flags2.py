"""Plausibility flags for the 2026-11-07 round, calibrated to Sam's places.

Three flags, each a forecast at odds with the zone's own data:
- collapse: median at 11-07 at or below a third of the recent level, the
  level being at least 3 (recent level: the larger of the mean of weeks
  10-10 and 10-17 and the count already reported for 10-24);
- blow-up: median at 11-07 above three times the largest count reported
  since 10-03 (at least 1) and 5 or more above it, or 95% quantile above
  ten times it and 20 or more above it;
- no cases ever: no case ever reported in the zone, yet 3.5 or more cases
  expected (sample mean) over the four forecast weeks.
"""

import sys
from pathlib import Path

import pandas as pd

S = Path(sys.argv[1])
FORECAST = ["2026-11-07", "2026-11-14", "2026-11-21", "2026-11-28"]
ts = pd.read_csv(S / "hub/ts.csv")
names = ts.drop_duplicates("location").set_index("location").location_name
rep = ts[ts.as_of == "2026-11-01"].pivot_table(
    index="location", columns="target_end_date", values="observation"
).fillna(0)
level = pd.concat(
    [rep[["2026-10-10", "2026-10-17"]].mean(axis=1), rep["2026-10-24"]], axis=1
).max(axis=1)
recent = rep[[c for c in rep.columns if c >= "2026-10-03"]].max(axis=1).clip(lower=1)
ever = rep.sum(axis=1)
sys.path.insert(0, str(S))
from flags_files import FILES  # noqa: E402

means = {}
for model, path in FILES.items():
    d = pd.read_csv(path)
    d = d[d.target_end_date.isin(FORECAST)]
    means[model] = d.groupby("location").value.sum() / d.output_type_id.nunique()
q = pd.read_csv(S / "quantiles.csv").set_index(["model", "location", "target_end_date"])

rows = []
for (model, loc), g in q.groupby(level=[0, 1]):
    g = g.droplevel([0, 1])
    m0, hi0 = g.q50["2026-11-07"], g.q95["2026-11-07"]
    lv, mx = level.get(loc, 0), recent.get(loc, 1)
    out = []
    if lv >= 3 and m0 <= lv / 3:
        out.append("collapse")
    if (m0 > 3 * mx and m0 - mx >= 5) or (hi0 > 10 * mx and hi0 - mx >= 20):
        out.append("blow-up")
    if ever.get(loc, 0) == 0 and means[model].get(loc, 0) >= 3.5:
        out.append("no cases ever")
    for f in out:
        rows.append(dict(
            model=model, location=loc, name=names.get(loc), flag=f,
            level=lv, recent_max=rep.loc[loc, [c for c in rep.columns if c >= "2026-10-03"]].max(),
            rep_1024=rep.loc[loc, "2026-10-24"], rep_1031=rep.loc[loc, "2026-10-31"],
            med=m0, lo=g.q05["2026-11-07"], hi=hi0,
            med_h3=g.q50["2026-11-28"], hi_h3=g.q95["2026-11-28"],
        ))
flags = pd.DataFrame(rows)
flags.to_csv(S / "flags2.csv", index=False)
for (m, f), g in flags.groupby(["model", "flag"]):
    print(m, f, len(g), ", ".join(g.name))
