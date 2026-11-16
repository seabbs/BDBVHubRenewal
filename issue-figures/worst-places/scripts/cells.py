import json, sys, pandas as pd
D="/home/seabbs/bdbv-perf/"
d=json.load(open(D+"report_data.json"))
sc=pd.read_csv(D+"scores_cells.csv")
OURS=["seabbs-renewal","seabbs-rwnowcast"]
asof=d["rounds"]
for loc in sys.argv[1].split(","):
  tr=d["truth"].get(loc,{})
  print(f"=== {loc}  latest:", {k:v for k,v in tr.get("2026-11-15",{}).items() if k>="09-19"})
  for r,a in asof.items():
    print(f" round {r} asof {a}: ", {k:v for k,v in tr.get(a,{}).items() if k>="09-19"})
    c=sc[(sc.location==loc)&(sc.reference_date==r)]
    oth=c[~c.model.isin(OURS)].groupby("target_end_date").agg(oc=("crps","mean"),omed=("median","median"),nt=("model","nunique"))
    for m in OURS:
      f=d["forecasts"].get(r,{}).get(m,{}).get(loc)
      if not f: continue
      q={w:x for w,x in zip(f["w"],f["q"])}
      mc=c[c.model==m].set_index("target_end_date")
      line=[]
      for w in sorted(mc.index):
        k=w[5:];qq=q.get(k,[None]*5)
        o=oth.loc[w] if w in oth.index else None
        line.append(f"{k} h{int(mc.loc[w,'horizon'])} y{mc.loc[w,'truth']:.0f} [{qq[0]},{qq[2]},{qq[4]}] c{mc.loc[w,'crps']:.1f}" + (f"/o{o.oc:.1f} om{o.omed:.0f} n{o.nt}" if o is not None else ""))
      print(f"  {m[7:]}: "+" | ".join(line))
