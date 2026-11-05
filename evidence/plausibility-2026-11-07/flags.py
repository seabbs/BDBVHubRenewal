"""Plausibility flags for the 2026-11-07 round, per model and place."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

S = Path(sys.argv[1])
ART = S / "art"
FILES = {
    "randomwalk": ART
    / "inputs/candidates/randomwalk/submission/model-output/seabbs-randomwalk"
    / "2026-11-07-seabbs-randomwalk.csv",
    "randomwalknowcast": ART
    / "inputs/candidates/randomwalknowcast/submission/model-output"
    / "seabbs-rwnowcast/2026-11-07-seabbs-rwnowcast.csv",
    "pprandomwalknowcast": S
    / "pprw/model-output/seabbs-pprw/2026-11-07-seabbs-pprw.csv",
    "renewal": ART
    / "candidate-main/submission/model-output/seabbs-renewal"
    / "2026-11-07-seabbs-renewal.csv",
    "renewalsp": ART
    / "candidate-renewalsp/submission/model-output/seabbs-renewal"
    / "2026-11-07-seabbs-renewal.csv",
    "renewalbvd": ART
    / "candidate-bvd/submission/model-output/seabbs-renewal"
    / "2026-11-07-seabbs-renewal.csv",
}
AS_OF = "2026-11-01"
COMPLETE = ["2026-10-10", "2026-10-17"]  # reported at lag of 2+ weeks
RECENT = ["2026-09-19", "2026-09-26", "2026-10-03", "2026-10-10",
          "2026-10-17", "2026-10-24", "2026-10-31"]
H0, H1, H3 = "2026-11-07", "2026-11-14", "2026-11-28"
FORECAST = ["2026-11-07", "2026-11-14", "2026-11-21", "2026-11-28"]

ts = pd.read_csv(S / "hub/ts.csv")
names = ts[["location", "location_name", "location_type"]].drop_duplicates(
    "location"
)
rep = ts[ts.as_of == AS_OF].pivot_table(
    index="location", columns="target_end_date", values="observation"
)
rep = rep.reindex(columns=sorted(rep.columns)).fillna(0)
level = rep[COMPLETE].mean(axis=1)
# reported counts are lower bounds, so the latest weeks raise the bar
floor = rep[["2026-10-24", "2026-10-31"]].max(axis=1)
rmax = rep[RECENT].max(axis=1)
rsd = rep[RECENT[:-2]].std(axis=1)
trend = rep["2026-10-17"] - rep["2026-10-03"]

rows, summ = [], []
for model, path in FILES.items():
    d = pd.read_csv(path, dtype={"output_type_id": int})
    q = d.groupby(["location", "target_end_date"]).value.quantile(
        [0.05, 0.5, 0.95, 0.975]
    ).unstack()
    q.columns = ["q05", "q50", "q95", "q975"]
    q = q.reset_index()
    q["model"] = model
    summ.append(q)
    # zone sums against the province, per sample
    z = d[d.location.str.count("-") == 2].copy()
    z["province"] = z.location.str[:5]
    zs = z.groupby(["province", "target_end_date", "output_type_id"]).value.sum()
    pv = d[d.location.str.count("-") == 1].set_index(
        ["location", "target_end_date", "output_type_id"]
    ).value
    pv.index.names = zs.index.names
    zp = pd.DataFrame({"zsum": zs, "prov": pv}).dropna()
    zp = zp.groupby(["province", "target_end_date"]).median()
    for loc, g in q.groupby("location"):
        g = g.set_index("target_end_date")
        lv, mx, sd = level.get(loc, 0), rmax.get(loc, 0), rsd.get(loc, 0)
        scale = (
            "national" if loc == "CD"
            else "province" if loc.count("-") == 1 else "zone"
        )
        base = dict(model=model, location=loc, scale=scale, level=lv,
                    rmax=mx, med_h0=g.q50[H0], lo_h0=g.q05[H0],
                    hi_h0=g.q95[H0], med_h3=g.q50[H3], lo_h3=g.q05[H3],
                    hi_h3=g.q95[H3], q975_max=g.q975[FORECAST].max())
        meds = g.q50[FORECAST]
        hi = max(lv, floor.get(loc, 0))
        if any(m > 3 * hi and m - hi >= 3 for m in meds):
            rows.append(base | dict(flag="median high"))
        if any(m < lv / 3 and lv - m >= 3 for m in meds):
            rows.append(base | dict(flag="median low"))
        top = g.q95[FORECAST].max()
        if top > 5 * max(mx, 2) and top - mx >= 10:
            rows.append(base | dict(flag="upper tail"))
        width = (g.q95 - g.q05)[H1]
        if sd >= 2 and width < sd:
            rows.append(base | dict(flag="interval narrow"))
        m0, m3 = g.q50[H0], g.q50[H3]
        steep = (m3 > 2 * max(m0, 1) or m3 < m0 / 2) and abs(m3 - m0) >= 5
        drift = (m3 > 2 * hi or m3 < lv / 2) and abs(m3 - hi) >= 10
        tr = np.sign(trend.get(loc, 0))
        if (steep and np.sign(m3 - m0) != tr) or (
            drift and np.sign(m3 - hi) != tr
        ):
            rows.append(base | dict(flag="trend unsupported"))
        if scale == "province" and (loc, H0) in zp.index:
            zsum, prov = zp.loc[(loc, H0)]
            if (zsum > 1.5 * prov or zsum < prov / 1.5) and abs(zsum - prov) >= 5:
                rows.append(base | dict(flag=f"zones sum {zsum:.0f}"))

flags = pd.DataFrame(rows).merge(names, on="location", how="left")
flags.to_csv(S / "flags.csv", index=False)
pd.concat(summ).to_csv(S / "quantiles.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
print(flags.groupby(["model", "flag"]).size().unstack(fill_value=0))
cols = ["model", "location_name", "scale", "flag", "level", "rmax",
        "med_h0", "lo_h0", "hi_h0", "med_h3", "hi_h3"]
print(flags[cols].round(1).to_string())
