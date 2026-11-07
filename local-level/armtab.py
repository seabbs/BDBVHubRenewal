"""Score table for arms: overall and small-count places, randomwalknowcast and randomwalk."""
import sys
import pandas as pd
from bdbvhubrenewal.data import load_daily_zones
from bdbvhubrenewal.data.zones import weekly_zone_counts

D = sys.argv[1]
arms = sys.argv[2].split(",")
SMALL = ["CD-BU", "CD-NU", "CD-SA", "CD-SU"]
zc = weekly_zone_counts(load_daily_zones(), "2026-11-01")
recent = zc[zc.target_end_date >= "2026-09-20"].groupby("location").total.max()
small_zones = set(recent[recent <= 5].index)
rows = []
for a in arms:
    for m in ["randomwalknowcast", "randomwalk"]:
        r = pd.read_csv(f"{D}/{a}/recovery-scores.csv"); r = r[r.model == m]
        n = pd.read_csv(f"{D}/{a}/nowcast-scores.csv"); n = n[(n.model == m) & (n.target == "reported")]
        z = pd.read_csv(f"{D}/{a}/zone-scores.csv"); z = z[z.model == m + " (summed to provinces)"]
        ns = n[n.location.isin(SMALL)]
        zs = z[z.location.isin(small_zones)]
        rows.append(dict(arm=a, model=m,
            rec=f"{r.wis.mean():.2f}/{r.log_wis.mean():.3f}/{r.coverage_90.mean():.2f}",
            rel=f"{n.wis.mean():.2f}/{n.log_wis.mean():.3f}/{n.coverage_90.mean():.2f}",
            zone=f"{z.wis.mean():.3f}/{z.log_wis.mean():.3f}/{z.coverage_90.mean():.2f}",
            small_prov=f"{ns.wis.mean():.3f}/{ns.log_wis.mean():.3f}/{ns.coverage_90.mean():.2f}",
            small_zones=f"{zs.wis.mean():.3f}/{zs.log_wis.mean():.3f}/{zs.coverage_90.mean():.2f}"))
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
print("small zones:", len(small_zones))
