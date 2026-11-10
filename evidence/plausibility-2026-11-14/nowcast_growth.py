"""Each model's nowcast of weeks 10-24 and 10-31 against past growth.

Usage: nowcast_growth.py <S> <data dir>

For CD and the provinces with at least 20 cases a week, the growth from a
week's count at age 1 (or 2) to its count in the 2026-11-08 release, over
the weeks that have since aged by at least 2 more weeks, gives a range.
The count reported at 11-08 times that range is where the final count
should fall (a lower bound, as 11-08 is not final either). A nowcast is
flagged when its 90% interval misses that range or when its 95% quantile
is above 1.5 times the top of it.
Reads `<S>/quantiles.csv` from `flags3.py`.
"""

import sys
from pathlib import Path

import pandas as pd

S, DATA = Path(sys.argv[1]), Path(sys.argv[2])
t = pd.read_csv(DATA / "target.csv")
t["age"] = (
    (pd.to_datetime(t.cutoff) - pd.to_datetime(t.target_end_date)).dt.days
    + 6
) // 7
last = t[t.vintage == "2026-11-08"].set_index(
    ["location", "target_end_date"]
).total
t["final"] = [last.get(k) for k in zip(t.location, t.target_end_date)]
old = t[(t.vintage < "2026-11-08") & (t.vintage >= "2026-10-09")]
q = pd.read_csv(S / "quantiles.csv").set_index(
    ["model", "location", "target_end_date"]
)
rows = []
for loc in sorted(t.location.unique()):
    if loc.count("-") > 1:
        continue
    for week, age in [("2026-10-24", 2), ("2026-10-31", 1)]:
        rep = last.get((loc, week), 0)
        if rep < 20:
            continue
        h = old[(old.location == loc) & (old.age == age) & (old.total > 0)
                & (old.target_end_date <= "2026-10-17")]
        g = (h.final / h.total)
        lo, hi = rep * g.min(), rep * g.max()
        for model in q.index.get_level_values(0).unique():
            if (model, loc, week) not in q.index:
                continue
            r = q.loc[(model, loc, week)]
            flag = ""
            if r.q95 < lo or r.q05 > hi:
                flag = "misses"
            elif r.q95 > 1.5 * hi:
                flag = "tail"
            rows.append(dict(
                model=model, location=loc, week=week, age=age, reported=rep,
                growth=f"{g.min():.2f} to {g.max():.2f}",
                range=f"{lo:.0f} to {hi:.0f}",
                nowcast=f"{r.q50:.0f} [{r.q05:.0f}, {r.q95:.0f}]",
                q95_over_top=round(r.q95 / hi, 2), flag=flag,
            ))
out = pd.DataFrame(rows)
out.to_csv(S / "nowcast_growth.csv", index=False)
pd.set_option("display.width", 250)
print(out.to_string(index=False))
