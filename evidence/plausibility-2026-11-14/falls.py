"""How often and how far reported week counts fall in later releases.

Usage: falls.py <S> <data dir>

For each place and week reported at age a (weeks from the week holding the
release's cut-off) in every earlier release, it compares the count
with the same week in the 2026-11-08 release (the latest).
It writes `<S>/falls.csv` (one row per place, week and release) and prints,
by level and age, the share of place-weeks whose count later fell, the
median and largest fall, and the cases lost as a share of those reported.
For the provinces it also splits the falls into the observed and imputed
streams.
"""

import sys
from pathlib import Path

import pandas as pd

S, DATA = Path(sys.argv[1]), Path(sys.argv[2])
t = pd.read_csv(DATA / "target.csv")
z = pd.read_csv(DATA / "health_zones.csv.gz")
t["level"] = t.location.map(lambda x: "national" if x == "CD" else "province")
z["level"] = "zone"
cols = ["level", "location", "vintage", "cutoff", "target_end_date", "total"]
d = pd.concat([t[cols + ["observed", "imputed"]], z[cols]])
d["age"] = (
    (pd.to_datetime(d.cutoff) - pd.to_datetime(d.target_end_date)).dt.days
    + 6
) // 7
latest = d.vintage.max()
last = d[d.vintage == latest].set_index(["location", "target_end_date"])
d = d[d.vintage < latest]
key = list(zip(d.location, d.target_end_date, strict=True))
# a zone week missing from the latest release has 0 cases there
d["final"] = [last.total.get(k, 0) for k in key]
for s in ["observed", "imputed"]:
    d[f"final_{s}"] = [last[s].get(k) if s in last else None for k in key]
d = d[d.total > 0]
d["fall"] = (d.total - d.final).clip(lower=0)
d["fell"] = d.fall > 0
d.to_csv(S / "falls.csv", index=False)

pd.set_option("display.width", 200)
g = d[d.age.between(0, 6)].groupby(["level", "age"])
tab = g.agg(
    n=("fell", "size"), share_fell=("fell", "mean"),
    reported=("total", "sum"), lost=("fall", "sum"),
    max_fall=("fall", "max"),
)
tab["lost_share"] = tab.lost / tab.reported
tab["median_fall_when_fell"] = g.apply(
    lambda x: x.loc[x.fell, "fall"].median() if x.fell.any() else 0,
    include_groups=False,
)
tab["rel_fall_when_fell"] = g.apply(
    lambda x: (x.loc[x.fell, "fall"] / x.loc[x.fell, "total"]).median()
    if x.fell.any() else 0,
    include_groups=False,
)
print(tab.round(3).to_string())
p = d[(d.level == "province") & d.fell]
p = p.assign(
    obs_fall=(p.observed - p.final_observed).clip(lower=0),
    imp_fall=(p.imputed - p.final_imputed).clip(lower=0),
)
print("\nProvince falls split by stream (sum of falls):",
      p[["fall", "obs_fall", "imp_fall"]].sum().to_dict())
print("\nLargest falls:")
print(d.sort_values("fall").tail(12)[
    ["level", "location", "target_end_date", "vintage", "age", "total",
     "final"]
].to_string(index=False))
