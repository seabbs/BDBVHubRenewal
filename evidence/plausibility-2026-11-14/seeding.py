"""Never-seeded zones: how many each draw seeds, against the data.

Usage: seeding.py <S> model=samples.csv ...
"""

import sys
from pathlib import Path

import pandas as pd

S = Path(sys.argv[1])
FORECAST = ["2026-11-14", "2026-11-21", "2026-11-28", "2026-12-05"]
ts = pd.read_csv(S / "hub/ts.csv")
cur = ts[(ts.as_of == "2026-11-08") & (ts.observation > 0)]
seen = set(cur.location)
first = cur[cur.location.str.count("-") == 2].groupby("location") \
    .target_end_date.min()
print("zones first seeded per week since 09-12:",
      first[first >= "2026-09-12"].value_counts().sort_index().to_dict())
rows = []
for arg in sys.argv[2:]:
    model, path = arg.split("=", 1)
    d = pd.read_csv(path)
    if "output_type_id" in d.columns:
        d = d.rename(columns={"output_type_id": "sample"})
    z = d[(d.location.str.count("-") == 2) & ~d.location.isin(seen)
          & d.target_end_date.isin(FORECAST)]
    per = z.groupby(["sample", "location"]).value.sum()
    n_zones = (per > 0).groupby(level=0).sum()
    cases = per.groupby(level=0).sum()
    rows.append(dict(
        model=model, never_seen=z.location.nunique(),
        zones_seeded_median=n_zones.median(),
        zones_seeded_q95=n_zones.quantile(0.95),
        cases_median=cases.median(), cases_q95=cases.quantile(0.95),
        cases_max=cases.max(),
    ))
out = pd.DataFrame(rows)
out.to_csv(S / "seeding.csv", index=False)
print(out.round(1).to_string(index=False))
