"""Plausibility check table (place type and flag) for zone forecasts of arms."""
import sys
import pandas as pd
sys.path.insert(0, sys.argv[1])
import checks
from bdbvhubrenewal.data import load_daily_zones
from bdbvhubrenewal.data.zones import weekly_zone_counts

D, arms = sys.argv[2], sys.argv[3].split(",")
dz = load_daily_zones()
counts = pd.concat([weekly_zone_counts(dz, v).assign(vintage=v) for v in ["2026-10-17", "2026-10-24"]])
out = []
for a in arms:
    z = pd.read_csv(f"{D}/{a}/zone-scores.csv")
    z = z[z.model.str.endswith("(summed to provinces)") & (z.target == "reported")].copy()
    z["model"] = z.model.str.replace(" (summed to provinces)", "", regex=False)
    z["scale"] = "zone"
    q = pd.read_csv(f"{D}/{a}/zq.csv")
    q = q[q.target_end_date == q.reference_date].copy()
    q["origin"] = (pd.to_datetime(q.reference_date) - pd.Timedelta(days=7)).dt.strftime("%Y-%m-%d")
    q = q[q.origin.isin(["2026-10-17", "2026-10-24"])]
    types = checks.place_types(counts, z[["origin", "location"]].drop_duplicates())
    fl = checks.flags(counts, q[["model", "origin", "location", "q50", "q95"]])
    t = checks.check_summary(z, types, fl).assign(arm=a)
    out.append(t)
t = pd.concat(out)
t = t[t.model == "randomwalknowcast"]
t["cell"] = t["WIS"].round(2).astype(str) + " / " + t["90% coverage"].round(2).astype(str) + " (" + t["n"].astype(str) + ")"
pd.set_option("display.width", 250)
print(t.pivot_table(index=["split", "group"], columns="arm", values="cell", aggfunc="first").to_string())
