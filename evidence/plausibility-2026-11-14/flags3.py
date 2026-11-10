"""Plausibility flags for the 2026-11-14 round (data as of 2026-11-08).

Usage: flags3.py <S> <data dir> model=samples.csv [model=samples.csv ...]

Each samples file is long (location, target_end_date, sample, value), or a
hub file (output_type_id is the sample). The rules are those of the
2026-11-07 round (`flags.py` and `flags2.py` there), moved on one week, plus
an uplift flag and a wide-interval flag.

Weeks: the release's latest onset is 2026-11-05, so week 11-07 is partial and
week 10-31 is one week old. The complete weeks are those at least two weeks
old (10-10, 10-17, 10-24). The level is the mean of 10-17 and 10-24.
The first forecast week is 11-14 (H0) and the last 12-05 (H3).

Sam's flags (as `flags2.py`):
- collapse: median at H0 at or below a third of the recent level, the level
  being at least 3 (recent level: the larger of the mean of 10-17 and 10-24
  and the count already reported for 10-31);
- blow-up: median at H0 above three times the largest count reported since
  10-10 (at least 1) and 5 or more above it, or 95% quantile above ten times
  it and 20 or more above it;
- no cases ever: no case ever reported, yet 3.5 or more cases expected
  (sample mean) over the four forecast weeks.

General flags (as `flags.py`): median high, median low, upper tail, interval
narrow, trend unsupported.

New:
- uplift: the three complete weeks are flat or falling (10-24 at most 1.1
  times the mean of 10-10 and 10-17), yet the median rises over the forecast
  horizon: at H3 it is more than 1.2 times the median at H0, and 5 or more
  cases above it;
- step-up: the same flat or falling weeks, yet the median at H0 is more than
  1.3 times their mean and 5 or more above it. The complete weeks are not
  final (10-24 has grown by 3% to 31% after age 2), so this allows about 10%
  for later reports and 20% on top;
- interval wide: the 90% interval at H3 is more than four times the larger of
  the median at H3 and the level, and 20 or more wide;
- interval narrow at H3: the 90% interval at H3 is under half the level, the
  level being at least 10.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from names import location_names  # noqa: E402

S, DATA = Path(sys.argv[1]), Path(sys.argv[2])
FILES = dict(a.split("=", 1) for a in sys.argv[3:])
AS_OF = "2026-11-08"
FLAT = ["2026-10-10", "2026-10-17", "2026-10-24"]
COMPLETE = ["2026-10-17", "2026-10-24"]
LATEST = ["2026-10-31", "2026-11-07"]
RECENT = ["2026-09-26", "2026-10-03", "2026-10-10", "2026-10-17",
          "2026-10-24", "2026-10-31", "2026-11-07"]
SINCE = "2026-10-10"
FORECAST = ["2026-11-14", "2026-11-21", "2026-11-28", "2026-12-05"]
H0, H1, H3 = FORECAST[0], FORECAST[1], FORECAST[3]

ts = pd.read_csv(S / "hub/ts.csv")
names = location_names(S, DATA)
rep = ts[ts.as_of == AS_OF].pivot_table(
    index="location", columns="target_end_date", values="observation"
)
rep = rep.reindex(columns=sorted(rep.columns)).fillna(0)
level = rep[COMPLETE].mean(axis=1)
floor = rep[LATEST].max(axis=1)
rmax = rep[RECENT].max(axis=1)
rsd = rep[RECENT[:-2]].std(axis=1)
trend = rep["2026-10-24"] - rep["2026-10-10"]
level2 = pd.concat([level, rep["2026-10-31"]], axis=1).max(axis=1)
since = rep[[c for c in rep.columns if c >= SINCE]].max(axis=1)
ever = rep.sum(axis=1)
flat_level = rep[FLAT].mean(axis=1)
flat = rep[FLAT[2]] <= 1.1 * rep[FLAT[:2]].mean(axis=1)


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "output_type_id" in d.columns:
        d = d.rename(columns={"output_type_id": "sample"})
    return d[["location", "target_end_date", "sample", "value"]]


rows, summ = [], []
for model, path in FILES.items():
    d = load(path)
    nsamp = d["sample"].nunique()
    fc = d[d.target_end_date.isin(FORECAST)]
    means = fc.groupby("location").value.sum() / nsamp
    q = d.groupby(["location", "target_end_date"]).value.quantile(
        [0.05, 0.25, 0.5, 0.75, 0.95, 0.975]
    ).unstack()
    q.columns = ["q05", "q25", "q50", "q75", "q95", "q975"]
    q = q.reset_index()
    q["model"] = model
    summ.append(q)
    for loc, g in q.groupby("location"):
        g = g.set_index("target_end_date")
        lv, mx, sd = level.get(loc, 0), rmax.get(loc, 0), rsd.get(loc, 0)
        scale = (
            "national" if loc == "CD"
            else "province" if loc.count("-") == 1 else "zone"
        )
        base = dict(
            model=model, location=loc, name=names.get(loc, loc), scale=scale,
            level=lv, rmax=mx, flat_level=flat_level.get(loc, 0),
            **{f"rep {c[5:]}": rep.loc[loc, c] if loc in rep.index else 0
               for c in FLAT + LATEST},
            med_h0=g.q50[H0], lo_h0=g.q05[H0], hi_h0=g.q95[H0],
            med_h3=g.q50[H3], lo_h3=g.q05[H3], hi_h3=g.q95[H3],
        )
        out = []
        meds = g.q50[FORECAST]
        hi = max(lv, floor.get(loc, 0))
        # Sam's flags
        lv2, mx2 = level2.get(loc, 0), max(since.get(loc, 0), 1)
        m0, hi0 = g.q50[H0], g.q95[H0]
        if lv2 >= 3 and m0 <= lv2 / 3:
            out.append("collapse")
        if (m0 > 3 * mx2 and m0 - mx2 >= 5) or (
            hi0 > 10 * mx2 and hi0 - mx2 >= 20
        ):
            out.append("blow-up")
        if ever.get(loc, 0) == 0 and means.get(loc, 0) >= 3.5:
            out.append("no cases ever")
        # general flags
        if any(m > 3 * hi and m - hi >= 3 for m in meds):
            out.append("median high")
        if any(m < lv / 3 and lv - m >= 3 for m in meds):
            out.append("median low")
        top = g.q95[FORECAST].max()
        if top > 5 * max(mx, 2) and top - mx >= 10:
            out.append("upper tail")
        width = (g.q95 - g.q05)[H1]
        if sd >= 2 and width < sd:
            out.append("interval narrow")
        m3 = g.q50[H3]
        steep = (m3 > 2 * max(m0, 1) or m3 < m0 / 2) and abs(m3 - m0) >= 5
        drift = (m3 > 2 * hi or m3 < lv / 2) and abs(m3 - hi) >= 10
        tr = np.sign(trend.get(loc, 0))
        if (steep and np.sign(m3 - m0) != tr) or (
            drift and np.sign(m3 - hi) != tr
        ):
            out.append("trend unsupported")
        # new flags
        fl = flat_level.get(loc, 0)
        if flat.get(loc, True):
            if m3 > 1.2 * m0 and m3 - m0 >= 5:
                out.append("uplift")
            if m0 > 1.3 * fl and m0 - fl >= 5:
                out.append("step-up")
        w3 = g.q95[H3] - g.q05[H3]
        if w3 > 4 * max(m3, lv) and w3 >= 20:
            out.append("interval wide")
        if lv >= 10 and w3 < lv / 2:
            out.append("interval narrow at H3")
        rows += [base | dict(flag=f) for f in out]

flags = pd.DataFrame(rows)
flags.to_csv(S / "flags3.csv", index=False)
pd.concat(summ).to_csv(S / "quantiles.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
print(flags.groupby(["model", "scale", "flag"]).size().unstack(
    ["scale"], fill_value=0
).to_string())
