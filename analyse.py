import pickle, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from bdbvhubrenewal.forecasters.base import add_national
S = sys.argv[1]; V = sys.argv[2]
r100 = pickle.load(open(f"{S}/res100.pkl", "rb")); r4k = pickle.load(open(f"{S}/res4000.pkl", "rb"))
pd.set_option("display.width", 250)
rng = np.random.default_rng(1)
LOCS = {"CD": "National", "CD-IT": "Ituri", "CD-NK": "North Kivu"}
def sd(v): return np.log1p(v).std(ddof=1)
rows = []
for model in ["randomwalk", "randomwalknowcast"]:
    s100, info, ref = r100[(V, model)]; s4k = r4k[(V, model)][0]
    s100 = add_national(s100); s4k = add_national(s4k)
    sig = dict(zip(info["locs"], info["sigma"]))
    provs = [l for l in info["locs"] if l != "CD"]
    weeks = sorted(s4k.target_end_date.unique())
    # province theory: Var = Var(log1p x0) + k sigma^2 + 1/(1+m)
    th = {}
    for p in provs:
        w0 = info["start_nowcast"] or info["start"][info["locs"].index(p)]
        g = s4k[s4k.location == p]
        x0 = g[g.target_end_date == w0].value.values
        v0 = np.log1p(x0).var() if len(x0) else 0.0
        mu0 = np.log1p(x0).mean() if len(x0) else np.log1p(info["reported"][info["locs"].index(p)][info["week_ends"].index(w0)])
        for t in weeks:
            k = (pd.Timestamp(t) - pd.Timestamp(w0)).days // 7
            if k < 1: continue
            var_lam = v0 + k * sig[p] ** 2
            m = np.exp(mu0 + var_lam / 2)  # mean of 1+lambda, lognormal
            th[(p, t)] = dict(k=k, sd=np.sqrt(var_lam + 1 / m), mean=m - 1,
                              var=m**2 * (np.exp(var_lam) - 1) + (m - 1))
    for t in weeks:
        if all((p, t) in th for p in provs):
            mn = sum(th[(p, t)]["mean"] for p in provs); vr = sum(th[(p, t)]["var"] for p in provs)
            th[("CD", t)] = dict(k=th[("CD-IT", t)]["k"], sd=np.sqrt(np.log1p(vr / (1 + mn) ** 2)))
    for loc in LOCS:
        w0 = info["start_nowcast"] or info["start"][info["locs"].index(loc if loc != "CD" else "CD-IT")]
        for t in weeks:
            a = s100[(s100.location == loc) & (s100.target_end_date == t)].value.values
            b = s4k[(s4k.location == loc) & (s4k.target_end_date == t)].value.values
            boot = [sd(rng.choice(a, a.size)) for _ in range(500)]
            q = np.quantile(a, [.05, .25, .5, .75, .95])
            rows.append(dict(model=model, loc=loc, h=(pd.Timestamp(t) - pd.Timestamp(ref)).days // 7,
                             step=(pd.Timestamp(t) - pd.Timestamp(w0)).days // 7,
                             sigma=sig[loc] if loc != "CD" else np.nan,
                             q05=q[0], q25=q[1], q50=q[2], q75=q[3], q95=q[4],
                             sd100=sd(a), lo100=np.quantile(boot, .05), hi100=np.quantile(boot, .95),
                             sd4000=sd(b), theory=th.get((loc, t), {}).get("sd", np.nan)))
df = pd.DataFrame(rows)
print(df.round(3).to_string()); df.to_csv(f"{S}/walk-sd-{V}.csv", index=False)

fig, ax = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
for i, model in enumerate(["randomwalk", "randomwalknowcast"]):
    for j, (loc, lab) in enumerate(LOCS.items()):
        d = df[(df.model == model) & (df.loc_ == loc)] if False else df[(df.model == model) & (df["loc"] == loc)]
        a = ax[i, j]
        a.fill_between(d.h, d.lo100, d.hi100, color="#86b6ef", alpha=.5, lw=0, label="100 samples, 90% bootstrap")
        a.plot(d.h, d.sd100, "o-", color="#2a78d6", ms=4, lw=1.5, label="stored, 100 samples")
        a.plot(d.h, d.sd4000, "-", color="#eb6834", lw=2, label="rerun, 4000 samples")
        a.plot(d.h, d.theory, "--", color="#444444", lw=1.5, label="theory, √(v₀ + kσ² + 1/λ)")
        a.axvline(-0.5, color="#bbbbbb", lw=.8)
        a.set_title(f"{model}: {lab}", fontsize=9); a.grid(alpha=.25)
        if j == 0: a.set_ylabel("SD of log(1 + y)")
        if i == 1: a.set_xlabel("horizon (weeks)")
ax[0, 0].legend(fontsize=7, loc="upper left")
fig.suptitle(f"Forecast spread by horizon, vintage {V}", fontsize=10)
fig.tight_layout(); fig.savefig(f"{S}/walk-sd-{V}.png", dpi=130)

fig, ax = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True)
for i, model in enumerate(["randomwalk", "randomwalknowcast"]):
    for j, (loc, lab) in enumerate(LOCS.items()):
        d = df[(df.model == model) & (df["loc"] == loc)]; a = ax[i, j]
        a.fill_between(d.h, d.q05, d.q95, color="#b7d3f6", lw=0, label="5-95%")
        a.fill_between(d.h, d.q25, d.q75, color="#6da7ec", lw=0, label="25-75%")
        a.plot(d.h, d.q50, color="#184f95", lw=2, label="median")
        a.axvline(-0.5, color="#bbbbbb", lw=.8); a.set_title(f"{model}: {lab}", fontsize=9); a.grid(alpha=.25)
        if i == 1: a.set_xlabel("horizon (weeks)")
        if j == 0: a.set_ylabel("weekly cases")
ax[0, 0].legend(fontsize=7, loc="upper left")
fig.suptitle(f"Stored quantiles (100 samples), vintage {V}", fontsize=10)
fig.tight_layout(); fig.savefig(f"{S}/walk-q-{V}.png", dpi=130)
