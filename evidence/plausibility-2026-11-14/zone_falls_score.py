"""Zone backtest scores, main against #163, split by whether counts fell.

Usage: zone_falls_score.py <data dir> <out dir> name=scores.csv ...

The scores are `zone-backtest` outputs (every zone it scores, every
origin with a later release). A cell is one model, origin, truth release,
zone and week, on the reported target. It "fell" when the count the truth
release shows is below the count reported at the origin. A zone "has
falls" when any of its cells fell. Only cells every run scores are kept.
Writes `zone_falls_score.csv` and prints the tables.
"""

import sys
from pathlib import Path

import pandas as pd

data, out = sys.argv[1:3]
RUNS = dict(a.split("=", 1) for a in sys.argv[3:])
KEY = ["model", "origin", "truth_vintage", "location", "target_end_date"]
hz = pd.read_csv(Path(data) / "health_zones.csv.gz")
rep = hz.set_index(["vintage", "location", "target_end_date"]).total
cut = hz.drop_duplicates("vintage").set_index("vintage").cutoff


def load(scores: str, name: str) -> pd.DataFrame:
    s = pd.read_csv(scores)
    # " (summed to provinces)" is the reconciled form, which is submitted
    s = s[(s.target == "reported") & s.model.str.contains("summed")]
    s["reported"] = [
        rep.get(k, 0.0)
        for k in zip(s.origin, s.location, s.target_end_date, strict=True)
    ]
    s["weeks_back"] = (
        (
            pd.to_datetime(s.origin.map(cut))
            - pd.to_datetime(s.target_end_date)
        ).dt.days
        + 6
    ) // 7
    return s.assign(run=name)


d = pd.concat([load(f, n) for n, f in RUNS.items()])
both = d.groupby(KEY).run.nunique()
d = d.set_index(KEY).loc[lambda x: x.index.isin(both[both == len(RUNS)].index)]
d = d.reset_index()
d["fell"] = d.truth < d.reported
zones = d.groupby("location").fell.any()
d["zone_has_falls"] = d.location.map(zones)
d["age"] = d.weeks_back
rows = []
for by in [["model", "fell"], ["model", "zone_has_falls"], ["model"]]:
    g = (
        d.groupby([*by, "run"])
        .agg(
            n=("wis", "size"),
            wis=("wis", "mean"),
            log_wis=("log_wis", "mean"),
            cov50=("coverage_50", "mean"),
            cov90=("coverage_90", "mean"),
        )
        .reset_index()
    )
    rows.append(g.assign(group=" + ".join(by)))
    pd.set_option("display.width", 200)
    print(g.round(4).to_string(index=False), "\n")
pd.concat(rows).to_csv(Path(out) / "zone_falls_score.csv", index=False)
# data weeks only (weeks with a count at the origin), by age
dw = d[d.age >= 0]
g = (
    dw.groupby(["model", "age", "run"])
    .agg(
        n=("wis", "size"),
        fell=("fell", "mean"),
        wis=("wis", "mean"),
        cov90=("coverage_90", "mean"),
    )
    .reset_index()
)
print(g[g.age <= 6].round(4).to_string(index=False))
g.to_csv(Path(out) / "zone_falls_score_by_age.csv", index=False)
