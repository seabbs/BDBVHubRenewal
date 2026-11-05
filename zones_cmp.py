import pickle, sys, numpy as np, pandas as pd
pd.set_option("display.width", 300)
S = sys.argv[1]
names = pd.read_csv("data/health_zones.csv.gz")[["location", "location_name"]].drop_duplicates().set_index("location").location_name
from bdbvhubrenewal.data.tidy import load_daily_zones
from bdbvhubrenewal.data.zones import weekly_zone_counts
truth = weekly_zone_counts(load_daily_zones(), "2026-11-01").set_index(["location", "target_end_date"])["total"]
FOCUS = ["CD-HU-11", "CD-HU-07", "CD-HU-12", "CD-TO-02", "CD-HU-10", "CD-HU-02"]
summ, focus = [], []
for var in ["before", "A", "B", "AB"]:
    res = pickle.load(open(f"{S}/zones_{var}.pkl", "rb"))
    for (v, name), r in res.items():
        locs = r["locs"]; rep = r["reported"]; ix = {c: i for i, c in enumerate(locs)}
        ref = pd.Timestamp(r["ref"])
        for kind in ["direct", "rec"]:
            d = r[kind].copy(); d["h"] = (pd.to_datetime(d.target_end_date) - ref).dt.days // 7
            q = d.groupby(["location", "target_end_date", "h"]).value.quantile([.05, .5, .95]).unstack().reset_index()
            q.columns = ["location", "t", "h", "q05", "q50", "q95"]
            zq = q[q.location.str.count("-") == 2]
            zi = zq.location.map(ix)
            # recent reported: last 9 weeks up to cut-week minus 1
            n_w = rep.shape[1]
            recent = {z: rep[ix[z], max(0, n_w - 10):n_w - 1] for z in zq.location.unique()}
            zero = zq.location.map(lambda z: recent[z].sum() == 0)
            rmax = zq.location.map(lambda z: recent[z].max())
            fut = zq[zq.h >= 0]
            tr = [truth.get((l, t), 0) for l, t in zip(zq.location, zq.t)]
            zq = zq.assign(truth=tr)
            old = zq[(zq.h < -1)] if v != "2026-11-01" else zq.iloc[:0]
            summ.append(dict(variant=var, vintage=v, model=name, kind=kind,
                zero_zones=int(zero[zq.h >= 0].groupby(zq.location).first().sum()),
                zero_q95_gt3=round(float((fut[zero[zq.h >= 0]].groupby("location").q95.max() > 3).mean()), 2),
                q95_gt5max=round(float((fut[~zero[zq.h >= 0]].assign(r=rmax).pipe(lambda x: x.q95 > 5 * np.maximum(x.r, 1))).groupby(fut[~zero[zq.h >= 0]].location).any().mean()), 2),
                cover90_old=round(float(((old.truth >= old.q05) & (old.truth <= old.q95)).mean()), 2) if len(old) else np.nan))
            for z in FOCUS:
                for _, row in zq[(zq.location == z) & zq.h.isin([-1, 0, 3])].iterrows():
                    focus.append(dict(variant=var, vintage=v, model=name, kind=kind, zone=names[z], h=row.h,
                        band=f"{row.q50:.0f} [{row.q05:.0f}, {row.q95:.0f}]", truth=row.truth,
                        recent=" ".join(str(int(x)) for x in rep[ix[z], -6:])))
S_ = pd.DataFrame(summ); F = pd.DataFrame(focus)
S_.to_csv(f"{S}/zones-summary.csv", index=False); F.to_csv(f"{S}/zones-focus.csv", index=False)
print(S_.to_string(index=False))
