"""Draws below reported counts against the falls in the release history.

Usage: below_vs_data.py <S> model=samples.csv ...

For each model, level (national, province, zone) and age at the 2026-11-08
release (weeks 10-03 to 11-07, ages 5 to 0), over place-weeks with a
reported count above 0:
- p_below: share of draws below the reported count (the model's chance
  that the week's final count ends below it);
- lost: mean shortfall over all draws as a share of the reported count;
- p_beyond: share of draws further below than the largest fall the data
  show at that level and age (`falls.py`).
The data columns come from `<S>/falls.csv`: share of place-weeks that
fell (data_p) and cases lost as a share of those reported (data_lost).
Writes `<S>/below_vs_data.csv`.
"""

import sys
from pathlib import Path

import pandas as pd

S = Path(sys.argv[1])
falls = pd.read_csv(S / "falls.csv")
data = falls[falls.age.between(0, 6)].groupby(["level", "age"]).agg(
    data_n=("fell", "size"), data_p=("fell", "mean"),
    rep=("total", "sum"), lostc=("fall", "sum"), data_max=("fall", "max"),
)
data["data_lost"] = data.lostc / data.rep
ts = pd.read_csv(S / "hub/ts.csv")
cur = ts[ts.as_of == "2026-11-08"].set_index(
    ["location", "target_end_date"]
).observation
AGE = {"2026-11-07": 0, "2026-10-31": 1, "2026-10-24": 2, "2026-10-17": 3,
       "2026-10-10": 4, "2026-10-03": 5}
rows = []
for arg in sys.argv[2:]:
    model, path = arg.split("=", 1)
    d = pd.read_csv(path)
    if "output_type_id" in d.columns:
        d = d.rename(columns={"output_type_id": "sample"})
    d = d[d.target_end_date.isin(AGE)].copy()
    d["rep"] = [cur.get(k, 0) for k in zip(d.location, d.target_end_date)]
    d = d[d.rep > 0]
    d["age"] = d.target_end_date.map(AGE)
    d["level"] = d.location.str.count("-").map(
        {0: "national", 1: "province", 2: "zone"}
    )
    d["short"] = (d.rep - d.value).clip(lower=0)
    mx = data.data_max.reindex(
        list(zip(d.level, d.age))
    ).fillna(0).to_numpy()
    d["beyond"] = d.short > mx
    g = d.groupby(["level", "age"]).agg(
        place_weeks=("location", "nunique"), p_below=("short", lambda s:
                                                      (s > 0).mean()),
        short=("short", "sum"), rep=("rep", "sum"),
        p_beyond=("beyond", "mean"),
    )
    g["lost"] = g.short / g.rep
    rows.append(g.drop(columns=["short", "rep"]).assign(model=model))
out = pd.concat(rows).reset_index().merge(
    data[["data_n", "data_p", "data_lost", "data_max"]].reset_index(),
    on=["level", "age"], how="left",
)
out.to_csv(S / "below_vs_data.csv", index=False)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)
print(out.round(3).to_string(index=False))
