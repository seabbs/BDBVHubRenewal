import sys, pickle, numpy as np, pandas as pd
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
res = pickle.load(open(sys.argv[1], "rb")); V = sys.argv[2]; provs = sys.argv[3].split(",")
names = pd.read_csv("data/health_zones.csv.gz")[["location", "location_name"]].drop_duplicates().set_index("location").location_name
for name in ["randomwalk", "randomwalknowcast"]:
    r = res[(V, name)]; locs = r["locs"]; L = len(locs); ix = {c: i for i, c in enumerate(locs)}
    ref = r["ref"]; diag = r["diag"].set_index("location")
    rows = []
    for z in [l for l in locs if l[:5] in provs]:
        i = ix[z]
        rep = r["reported"][i]
        w0 = r["start_nowcast"] if r["start_nowcast"] is not None else None
        def q(df, wk): 
            v = df[(df.location == z) & (df.target_end_date == wk)].value
            return (v.quantile(.5), v.quantile(.95)) if len(v) else (np.nan, np.nan)
        wk_h0 = ref; wk_h3 = (pd.Timestamp(ref) + pd.Timedelta(days=21)).date().isoformat()
        src = int(r["sigma_source"][i])
        rows.append(dict(zone=z, name=names.get(z, ""), level=round(float(r["level"][i]), 1),
            last6=" ".join(str(int(x)) for x in rep[-6:]),
            pt_last3=" ".join(f"{x:.1f}" for x in r["point"][i][-3:]),
            sigma=round(float(r["sigma"][i]), 2), sig_src=("own" if src < L else "pool:" + locs[src - L]),
            delay=diag.loc[z, "delay_source"] if z in diag.index else "",
            obs=diag.loc[z, "obs_source"] if z in diag.index else "",
            dir_h0=q(r["direct"], wk_h0), dir_h3=q(r["direct"], wk_h3),
            rec_h0=q(r["rec"], wk_h0), rec_h3=q(r["rec"], wk_h3)))
    print(V, name, "ref", ref, "w0", r["start_nowcast"] and r["weeks"][r["start_nowcast"]], "weeks", r["weeks"][-6:])
    print(pd.DataFrame(rows).to_string(index=False))
    pv = r["prov"]
    for p in provs:
        x = pv[pv.location == p].groupby("target_end_date").value.quantile([.5, .95]).unstack()
        print(p, "province", x.tail(5).round(0).to_dict("index"))
