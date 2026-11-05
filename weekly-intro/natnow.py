import sys, numpy as np
from bdbvhubrenewal.data import load_daily
from bdbvhubrenewal.data.snapshots import snapshot_arrays
from bdbvhubrenewal.forecasters.chain import build_origin, Settings
daily = load_daily(sys.argv[2])
prov = tuple(sorted(set(daily["location"]) - {"CD"}))
o = build_origin(snapshot_arrays(daily, "2026-11-01", prov), Settings(), errors=False)
cd = o.snap.locations.index("CD"); pv = [i for i, c in enumerate(o.snap.locations) if c != "CD"]
W = ["2026-10-17", "2026-10-24", "2026-10-31"]; idx = [list(o.snap.week_ends).index(x) for x in W]; w = W
print(sys.argv[1], "weeks", w, "CD point", o.point[cd, idx].round(0), "sum of provinces", o.point[pv].sum(0)[idx].round(0), "reported", o.reported[cd, idx], "g0 national %.2f" % o.fit.g[cd, 0])
