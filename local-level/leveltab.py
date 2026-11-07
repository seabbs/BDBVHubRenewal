"""90% coverage and bias by weeks back and level for randomwalknowcast."""
import sys
import pandas as pd
D, arms = sys.argv[1], sys.argv[2].split(",")
SMALL = {"CD-BU", "CD-NU", "CD-SA", "CD-SU"}
out = []
for a in arms:
    n = pd.read_csv(f"{D}/{a}/nowcast-scores.csv"); n = n[(n.model == "randomwalknowcast") & (n.target == "reported")]
    n["level"] = n.location.map(lambda l: "national" if l == "CD" else "large" if l in ("CD-IT", "CD-NK") else "small" if l in SMALL else "other")
    z = pd.read_csv(f"{D}/{a}/zone-scores.csv"); z = z[(z.model == "randomwalknowcast (summed to provinces)") & (z.target == "reported")]
    z = z.assign(level="zones", weeks_back=-z.horizon - 1)
    d = pd.concat([n, z])
    d["wb"] = d.weeks_back.clip(lower=-1, upper=3)
    g = d.groupby(["level", "wb"]).agg(c90=("coverage_90", "mean"), bias=("bias", "mean"))
    out.append(g.round(2).astype(str).agg(" / ".join, axis=1).rename(a))
pd.set_option("display.width", 250)
print("cells: 90% coverage / bias; weeks back -1 means forecast weeks")
print(pd.concat(out, axis=1).to_string())
