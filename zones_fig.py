import pickle, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from bdbvhubrenewal.data.tidy import load_daily_zones
from bdbvhubrenewal.data.zones import weekly_zone_counts
S = sys.argv[1]
names = pd.read_csv("data/health_zones.csv.gz")[["location", "location_name"]].drop_duplicates().set_index("location").location_name
Z = ["CD-HU-11", "CD-HU-07", "CD-HU-12", "CD-HU-02", "CD-HU-10", "CD-TO-02"]
VS = ["2026-10-24", "2026-11-01"]
dz = load_daily_zones()
res = {v: pickle.load(open(f"{S}/zones_{v}.pkl", "rb")) for v in ["before", "A"]}
fig, ax = plt.subplots(len(VS), len(Z), figsize=(14, 5.2), sharex="col")
col = {"before": "#eb6834", "A": "#2a78d6"}
for i, vin in enumerate(VS):
    late = weekly_zone_counts(dz, "2026-11-01").set_index(["location", "target_end_date"])["total"]
    at = weekly_zone_counts(dz, vin).set_index(["location", "target_end_date"])["total"]
    for j, z in enumerate(Z):
        a = ax[i, j]
        for var in ["before", "A"]:
            d = res[var][(vin, "randomwalknowcast")]["rec"]; d = d[d.location == z]
            q = d.groupby("target_end_date").value.quantile([.05, .5, .95]).unstack()
            t = pd.to_datetime(q.index)
            a.fill_between(t, q[.05], q[.95], color=col[var], alpha=.18, lw=0)
            a.plot(t, q[.5], color=col[var], lw=2, label={"before": "current", "A": "start at π ≥ 0.5"}[var])
        wk = pd.date_range(pd.Timestamp(vin) - pd.Timedelta(weeks=8), pd.Timestamp(vin) + pd.Timedelta(weeks=4), freq="7D")
        wk = [w for w in pd.date_range("2026-09-05", "2026-11-28", freq="7D")]
        a.plot(wk, [at.get((z, w.date().isoformat()), 0) for w in wk], "o", color="#444444", ms=3, label=f"reported at {vin}")
        if vin != "2026-11-01":
            a.plot(wk, [late.get((z, w.date().isoformat()), 0) for w in wk], "x", color="#008300", ms=4, label="reported at 2026-11-01")
        a.axvline(pd.Timestamp(vin), color="#bbbbbb", lw=.8)
        a.set_title(f"{names[z]}, origin {vin}", fontsize=8); a.grid(alpha=.25); a.tick_params(labelsize=7)
        a.set_xlim(pd.Timestamp("2026-09-01"), pd.Timestamp("2026-12-01"))
        for lab in a.get_xticklabels(): lab.set_rotation(45)
ax[0, 0].legend(fontsize=6, loc="upper left"); ax[0, 0].set_ylabel("weekly cases"); ax[1, 0].set_ylabel("weekly cases")
fig.suptitle("randomwalknowcast zone forecasts (as submitted, median and 90%), current start week against a start at π ≥ 0.5", fontsize=9)
fig.tight_layout(); fig.savefig(f"{S}/zones-before-after.png", dpi=120)
