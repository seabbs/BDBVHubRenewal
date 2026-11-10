"""Stack the national, province and zone counts of every release.

Writes `<S>/hub/ts.csv` with the columns the flag scripts of the
2026-11-07 round read (location, location_name, location_type, as_of,
target_end_date, observation), from the repo's tidy `data/` tables.
"""

import sys
from pathlib import Path

import pandas as pd

S, DATA = Path(sys.argv[1]), Path(sys.argv[2])
t = pd.read_csv(DATA / "target.csv")
z = pd.read_csv(DATA / "health_zones.csv.gz")
t["location_type"] = t.location.map(
    lambda x: "country" if x == "CD" else "province"
)
z["location_type"] = "health_zone"
cols = ["location", "location_name", "location_type", "vintage",
        "target_end_date", "total"]
ts = pd.concat([t[cols], z[cols]]).rename(
    columns={"vintage": "as_of", "total": "observation"}
)
(S / "hub").mkdir(parents=True, exist_ok=True)
ts.to_csv(S / "hub" / "ts.csv", index=False)
print(ts.groupby("as_of").size().tail(3))
