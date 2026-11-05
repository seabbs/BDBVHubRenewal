"""Write every table the PR #63 body cites, from runs of main and the PR head."""
import pickle, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

E = sys.argv[1]
TAGS = {"main": "main a5ad8b92", "pr": "PR head"}
PLACES = {"CD-HU": "Haut-Uele", "CD-NK": "North Kivu", "CD-HU-11": "Rungu", "CD-IT-22": "Logo",
          "CD-NK-26": "Mutwanga", "CD-HU-07": "Isiro", "CD-NK-20": "Lubero", "CD-TO-02": "Bafwasende",
          "CD-HU-12": "Wamba", "CD-NK-30": "Pinga"}

# Scores by model and check.
rows = []
for t in TAGS:
    for m in ["randomwalk", "randomwalknowcast"]:
        r = pd.read_csv(f"{E}/bt_{t}/recovery-scores.csv"); r = r[r.model == m]
        n = pd.read_csv(f"{E}/bt_{t}/nowcast-scores.csv"); n = n[(n.model == m) & (n.target == "reported")]
        z = pd.read_csv(f"{E}/zb_{t}/zone-scores.csv"); z = z[z.model == m + " (summed to provinces)"]
        for check, d in [("recovery", r), ("release", n), ("zone", z)]:
            rows.append(dict(version=t, model=m, check=check, n=len(d), wis=d.wis.mean(),
                             log_wis=d.log_wis.mean(), coverage_90=d.coverage_90.mean()))
pd.DataFrame(rows).round(3).to_csv(f"{E}/scores.csv", index=False)


def load(name, t):
    return pickle.load(open(f"{E}/{name}_{t}.pkl", "rb"))


def silent_zones(x):
    locs, rep = x["locs"], x["reported"]; cw = rep.shape[1] - 1
    return [l for i, l in enumerate(locs) if l.count("-") == 2 and rep[i, cw - 8:cw + 1].sum() == 0]


# Silent share and named places, 100 samples as submitted.
share, places = [], []
for t in TAGS:
    for (v, m), x in load("zones100", t).items():
        ref = pd.Timestamp(x["ref"]); sil = silent_zones(x)
        d = pd.concat([x["rec"], x["prov"]])
        d = d.assign(h=(pd.to_datetime(d.target_end_date) - ref).dt.days // 7)
        q = d[d.h >= 0].groupby("location").value.quantile(.95)
        share.append(dict(version=t, model=m, silent_zones=len(sil), share_q95_above_3=round(float((q[sil] > 3).mean()), 3)))
        q0 = d[d.h == 0].groupby("location").value.quantile([.05, .5, .95]).unstack()
        for z, nm in PLACES.items():
            a = q0.loc[z]
            places.append(dict(version=t, model=m, place=nm, week="2026-11-07", median=a[.5], q05=a[.05], q95=a[.95]))
pd.DataFrame(share).to_csv(f"{E}/silent-share.csv", index=False)
pd.DataFrame(places).to_csv(f"{E}/places.csv", index=False)

# Never-case zones and upper tails, 1000 samples as submitted, weeks 11-07 to 11-28.
never, tails = [], []
for t in TAGS:
    for name in ["zones1k", "pp1k"]:
        for (v, m), x in load(name, t).items():
            locs, rep = x["locs"], x["reported"]; ix = {c: i for i, c in enumerate(locs)}
            zones = [l for l in locs if l.count("-") == 2]
            nev = [z for z in zones if rep[ix[z]].sum() == 0]
            cased = [z for z in zones if rep[ix[z]].sum() > 0]
            r = x["rec"]; r = r[r.target_end_date >= "2026-11-07"]
            nv = r[r.location.isin(nev)]
            total = nv.groupby("location").value.mean() * 4
            q95 = r[r.location.isin(cased)].groupby(["location", "target_end_date"]).value.quantile(.95)
            rmax = pd.Series({z: max(rep[ix[z], -6:-2].max(), 1) for z in cased})
            over = (q95.groupby("location").max() > 5 * rmax).sum()
            never.append(dict(version=t, model=m, never_case_zones=len(nev), share_weeks_above_0=round(float((nv.value > 0).mean()), 3),
                              largest_draw=nv.value.max(), zones_expecting_3_5_or_more=int((total >= 3.5).sum())))
            row = dict(version=t, model=m, zones_with_cases=len(cased), q95_above_5x_recent_max=int(over))
            for z, nm in [("CD-IT-34", "Rimba"), ("CD-IT-31", "Nyankunde")]:
                qq = r[r.location == z].groupby("target_end_date").value.quantile(.95)
                row[f"{nm}_q95_min"], row[f"{nm}_q95_max"] = qq.min(), qq.max()
            tails.append(row)
pd.DataFrame(never).to_csv(f"{E}/never-case.csv", index=False)
pd.DataFrame(tails).to_csv(f"{E}/tails.csv", index=False)

# Figure of the named places, 100 samples as submitted.
col = {"main": "#eb6834", "pr": "#2a78d6"}
fig, ax = plt.subplots(2, 5, figsize=(15, 5.6))
xs = {t: load("zones100", t)[("2026-11-01", "randomwalknowcast")] for t in TAGS}
for a, (z, nm) in zip(ax.ravel(), PLACES.items()):
    for t, x in xs.items():
        d = pd.concat([x["rec"], x["prov"]]); d = d[d.location == z]
        q = d.groupby("target_end_date").value.quantile([.05, .5, .95]).unstack()
        tt = pd.to_datetime(q.index)
        a.fill_between(tt, q[.05], q[.95], color=col[t], alpha=.2, lw=0)
        a.plot(tt, q[.5], color=col[t], lw=2, label=TAGS[t])
    x = xs["main"]; i = x["locs"].index(z); wk = pd.to_datetime(x["weeks"]); keep = wk >= pd.Timestamp("2026-09-01")
    a.plot(wk[keep][:-1], x["reported"][i][keep][:-1], "o", color="#444444", ms=3, label="reported at 11-01")
    a.axvline(pd.Timestamp("2026-11-01"), color="#bbbbbb", lw=.8)
    a.set_title(nm, fontsize=9); a.grid(alpha=.25); a.tick_params(labelsize=7)
    for l in a.get_xticklabels(): l.set_rotation(45)
ax[0, 0].legend(fontsize=7, loc="upper left"); ax[0, 0].set_ylabel("weekly cases"); ax[1, 0].set_ylabel("weekly cases")
fig.suptitle("randomwalknowcast at origin 2026-11-01, as submitted: median and 90% interval", fontsize=10)
fig.tight_layout(); fig.savefig(f"{E}/rwn-places-11-01.png", dpi=110)
