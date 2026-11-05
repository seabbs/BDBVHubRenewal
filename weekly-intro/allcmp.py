import pickle, sys, numpy as np, pandas as pd
pd.set_option("display.width", 250)
S = sys.argv[1]
FOC = {"CD-HU-11": "Rungu", "CD-HU-07": "Isiro", "CD-HU-12": "Wamba", "CD-TO-02": "Bafwasende", "CD-IT-22": "Logo",
       "CD-NK-26": "Mutwanga", "CD-NK-30": "Pinga", "CD-NK-20": "Lubero", "CD-HU-08": "Makoro", "CD-NK-05": "Binza",
       "CD-HU": "Haut-Uele", "CD-NK": "North Kivu"}
rows, foc = [], []
for var in sys.argv[2].split(","):
    r = pickle.load(open(f"{S}/zonesall_{var}.pkl", "rb"))
    for (v, m), x in r.items():
        locs = x["locs"]; rep = x["reported"]; ix = {c: i for i, c in enumerate(locs)}; ref = pd.Timestamp(x["ref"])
        cw = rep.shape[1] - 1
        silent = {l for l in locs if l.count("-") == 2 and rep[ix[l], cw - 8:cw + 1].sum() == 0}
        for kind in ["direct", "rec"]:
            d = x[kind] if kind == "direct" else pd.concat([x["rec"], x["prov"][x["prov"].location.isin(["CD-HU", "CD-NK"])]])
            d = d.assign(h=(pd.to_datetime(d.target_end_date) - ref).dt.days // 7)
            q = d[d.h >= 0].groupby("location").value.quantile(.95)
            s = q[q.index.isin(silent)]
            rows.append(dict(variant=var, model=m, kind=kind, silent=len(s), share_q95_gt3=round(float((s > 3).mean()), 3)))
            if kind == "rec":
                qq = d[d.h.isin([-1, 0, 3])].groupby(["location", "h"]).value.quantile([.05, .5, .95]).unstack()
                for z, nm in FOC.items():
                    for h in [-1, 0, 3]:
                        if (z, h) in qq.index:
                            a = qq.loc[(z, h)]
                            foc.append(dict(model=m, place=nm, h=h, variant=var, band=f"{a[.5]:.0f} [{a[.05]:.0f}, {a[.95]:.0f}]",
                                            recent=" ".join(str(int(t)) for t in rep[ix[z], -5:-1]) if z in ix else ""))
R_ = pd.DataFrame(rows); F = pd.DataFrame(foc)
print(R_.to_string(index=False))
print(F.pivot_table(index=["model", "place", "recent", "h"], columns="variant", values="band", aggfunc="first").to_string())
R_.to_csv(f"{S}/allzones-share.csv", index=False); F.to_csv(f"{S}/allzones-focus.csv", index=False)
