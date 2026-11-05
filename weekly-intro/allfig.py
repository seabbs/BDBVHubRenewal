import pickle, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = sys.argv[1]
P = {"CD-HU": "Haut-Uele", "CD-NK": "North Kivu", "CD-HU-11": "Rungu", "CD-HU-07": "Isiro", "CD-HU-12": "Wamba",
     "CD-TO-02": "Bafwasende", "CD-IT-22": "Logo", "CD-NK-26": "Mutwanga", "CD-NK-20": "Lubero", "CD-NK-30": "Pinga"}
res = {v: pickle.load(open(f"{S}/zonesall_{v}.pkl", "rb"))[("2026-11-01", "randomwalknowcast")] for v in ["before", "intro"]}
col = {"before": "#eb6834", "intro": "#2a78d6"}
lab = {"before": "main", "intro": "this PR"}
fig, ax = plt.subplots(2, 5, figsize=(15, 5.6))
for a, (z, nm) in zip(ax.ravel(), P.items()):
    x0 = res["before"]; ix = x0["locs"].index(z) if z in x0["locs"] else None
    for v, x in res.items():
        d = pd.concat([x["rec"], x["prov"]]); d = d[d.location == z]
        q = d.groupby("target_end_date").value.quantile([.05, .5, .95]).unstack()
        t = pd.to_datetime(q.index)
        a.fill_between(t, q[.05], q[.95], color=col[v], alpha=.2, lw=0)
        a.plot(t, q[.5], color=col[v], lw=2, label=lab[v])
    rep = x0["reported"][ix]; wk = pd.to_datetime(x0["weeks"])
    keep = wk >= pd.Timestamp("2026-09-01")
    a.plot(wk[keep][:-1], rep[keep][:-1], "o", color="#444444", ms=3, label="reported at 11-01")
    a.axvline(pd.Timestamp("2026-11-01"), color="#bbbbbb", lw=.8)
    a.set_title(nm, fontsize=9); a.grid(alpha=.25); a.tick_params(labelsize=7)
    for l in a.get_xticklabels(): l.set_rotation(45)
ax[0, 0].legend(fontsize=7, loc="upper left"); ax[0, 0].set_ylabel("weekly cases"); ax[1, 0].set_ylabel("weekly cases")
fig.suptitle("randomwalknowcast at origin 2026-11-01, as submitted: median and 90% interval", fontsize=10)
fig.tight_layout(); fig.savefig(f"{S}/rwn-places-11-01.png", dpi=110)
