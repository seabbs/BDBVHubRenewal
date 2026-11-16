import pandas as pd, sys
D="/home/seabbs/bdbv-perf/"
t=pd.read_csv(D+"zone_vs_others.csv")
w=t[(t.n>=6)&(t.mean_truth*t.n>=10)]
pd.set_option("display.width",250)
for m in ["seabbs-renewal","seabbs-rwnowcast"]:
    x=w[w.model==m].sort_values("rel_log",ascending=False).head(16)
    print(m);print(x[["location","name","size","n","n_teams","mean_truth","crps","crps_others","rel_crps","rel_log"]].round(3).to_string())
sc=pd.read_csv(D+"scores_cells.csv")
OURS=["seabbs-renewal","seabbs-rwnowcast"]
KEY=["reference_date","location","target_end_date"]
p=sc[sc.scale.isin(["province","national"])]
o=p[~p.model.isin(OURS)]
rows=[]
for m in OURS:
  for loc,g in p[p.model==m].groupby("location"):
    oo=o[o.location==loc]; cnt=oo.groupby(KEY).model.nunique(); keep=cnt[cnt>=2].index
    gi=g.set_index(KEY); idx=gi.index.intersection(keep)
    if len(idx)<4: continue
    oc=oo.set_index(KEY).loc[idx].groupby("model")[["crps","log_crps"]].mean()
    mc=gi.loc[idx][["crps","log_crps"]].mean()
    rows.append(dict(model=m,loc=loc,n=len(idx),nt=len(oc),truth=gi.loc[idx].truth.mean(),crps=mc.crps,oth=oc.crps.mean(),rel_crps=mc.crps/oc.crps.mean(),rel_log=mc.log_crps/oc.log_crps.mean()))
r=pd.DataFrame(rows).sort_values(["model","rel_crps"],ascending=[True,False]);print(r.round(3).to_string())
r.to_csv(sys.argv[1]+"/prov_vs_others.csv",index=False)
