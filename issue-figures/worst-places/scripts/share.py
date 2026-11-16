import pandas as pd
H="/home/seabbs/bdbv-perf/hub/"
t=pd.read_csv(H+"target-data/time-series.csv")
T=lambda a,l,w: t[(t.as_of==a)&(t.location==l)&(t.target_end_date==w)].observation.sum()
zs={"CD-NK-14":"CD-NK","CD-NK-07":"CD-NK","CD-NK-25":"CD-NK","CD-IT-29":"CD-IT","CD-IT-30":"CD-IT","CD-IT-10":"CD-IT","CD-IT-35":"CD-IT","CD-TO-11":"CD-TO","CD-TO-02":"CD-TO"}
for r,a in [("2026-11-07","2026-11-01"),("2026-11-14","2026-11-08")]:
  d=pd.read_csv(H+f"model-output/seabbs-renewal/{r}-seabbs-renewal.csv")
  med=d.groupby(["location","target_end_date"]).value.median()
  print("round",r)
  for z,p in zs.items():
    out=[]
    for w in ["2026-10-31","2026-11-07"] if r=="2026-11-07" else ["2026-10-31","2026-11-07"]:
      zr,pr=T(a,z,w),T(a,p,w); zl,pl=T("2026-11-15",z,w),T("2026-11-15",p,w)
      zm,pm=med[(z,w)],med[(p,w)]
      out.append(f"{w[5:]}: prov med {pm:.0f} vs {pl:.0f} | zone share of unreported ours {(zm-zr)/max(pm-pr,1):.2f} obs {(zl-zr)/max(pl-pr,1):.2f} | zone med {zm:.0f} vs {zl:.0f}")
    print(" ",z," || ".join(out))
