# relative CRPS vs mean of other teams on matched cells, split by truth age
import pandas as pd
D="/home/seabbs/bdbv-perf/"
sc=pd.read_csv(D+"scores_cells.csv")
OURS=["seabbs-renewal","seabbs-rwnowcast"]
KEY=["reference_date","location","target_end_date"]
places={"seabbs-renewal":"CD-NK-14 CD-NK-25 CD-IT-29 CD-TO-11 CD-TO-02 CD-NK-07 CD-IT-30 CD-IT-28 CD-IT-10 CD-IT-13 CD-IT-35 CD-HU-07 CD-BU CD-TO CD CD-HU CD-IT".split(),
 "seabbs-rwnowcast":"CD-HU-02 CD-IT-28 CD-IT-27 CD-HU-10 CD-NK-25 CD-NK-26 CD-HU-11 CD-HU-07 CD-IT-17 CD-IT-07 CD-IT-21 CD-IT-36 CD-HU CD-NU CD".split()}
rows=[]
for m,locs in places.items():
  for loc in locs:
    c=sc[sc.location==loc]; o=c[~c.model.isin(OURS)]
    cnt=o.groupby(KEY).model.nunique(); keep=cnt[cnt>=2].index
    mine=c[c.model==m].set_index(KEY); idx=mine.index.intersection(keep)
    mi=mine.loc[idx]; oo=o.set_index(KEY).loc[idx].reset_index()
    def rel(sel):
      ii=mi.index[sel]
      if len(ii)==0: return float("nan"),0
      ot=oo.set_index(KEY).loc[ii].groupby("model").crps.mean().mean()
      return mi.loc[ii].crps.mean()/ot, len(ii)
    a,n=rel(mi.age>=0); s,ns=rel(mi.age>=2); y,ny=rel(mi.age<=1)
    # excess share from age<=1
    ot_cell=oo.groupby(KEY).crps.mean()
    ex=(mi.crps-ot_cell.loc[idx])
    share=ex[mi.age<=1].sum()/ex.sum() if ex.sum()>0 else float("nan")
    # bias of median vs truth at age>=2
    b=(mi[mi.age>=2]["median"]-mi[mi.age>=2].truth).mean()
    rows.append(dict(model=m[7:],loc=loc,n=n,rel_all=a,rel_age2plus=s,n2=ns,rel_age01=y,excess_from_age01=share,med_bias_age2=b,cov90_age2=mi[mi.age>=2].cov90.mean(),cov90_all=mi.cov90.mean()))
r=pd.DataFrame(rows);pd.set_option("display.width",250);print(r.round(2).to_string())
r.to_csv("agesplit.csv",index=False)
