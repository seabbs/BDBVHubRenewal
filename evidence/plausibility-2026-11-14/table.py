"""Markdown tables of the flagged places, one per model.

Usage: table.py <S> <model>
"""

import sys
from pathlib import Path

import pandas as pd

S, model = Path(sys.argv[1]), sys.argv[2]
f = pd.read_csv(S / "flags3.csv")
f = f[f.model == model]
q = pd.read_csv(S / "quantiles.csv")
q = q[q.model == model].set_index(["location", "target_end_date"])
GROUP = {
    "collapse": "Sam's", "blow-up": "Sam's", "no cases ever": "Sam's",
    "uplift": "uplift", "step-up": "uplift",
    "upper tail": "width", "interval wide": "width",
    "interval narrow": "width", "interval narrow at H3": "width",
}


def mi(loc: str, w: str) -> str:
    r = q.loc[(loc, w)]
    return f"{r.q50:.0f} ({r.q05:.0f} to {r.q95:.0f})"


rows = []
for (loc, name, scale), g in f.groupby(["location", "name", "scale"]):
    r = g.iloc[0]
    rows.append(dict(
        place=f"{name} ({loc})", scale=scale,
        flags=", ".join(g.flag),
        groups=set(GROUP.get(x, "other") for x in g.flag),
        reported=", ".join(f"{r[c]:.0f}" for c in
                           ["rep 10-17", "rep 10-24", "rep 10-31",
                            "rep 11-07"]),
        h0=mi(loc, "2026-11-14"), h3=mi(loc, "2026-12-05"),
        order=("national", "province", "zone").index(scale),
    ))
t = pd.DataFrame(rows).sort_values(["order", "place"])
print(f"{model}: {len(t)} places;",
      t.scale.value_counts().to_dict(),
      f.flag.value_counts().to_dict())
head = ("| place | flag | reported 10-17, 10-24, 10-31, 11-07 "
        "| median (90%) 11-14 | median (90%) 12-05 |\n| --- | --- | --- "
        "| --- | --- |")
print(head)
for r in t.itertuples():
    print(f"| {r.place} | {r.flags} | {r.reported} | {r.h0} | {r.h3} |")
