"""Apply the delay fitted at 11-08 to earlier releases and compare the
growth it predicts up to 11-08 with what was reported at 11-08."""
import argparse, numpy as np, pandas as pd
from bdbvhubrenewal import cli
from bdbvhubrenewal.forecasters import randomwalk as rw
from bdbvhubrenewal.forecasters.chain import week_nowcast, fit_delay
from bdbvhubrenewal.data.snapshots import weekly_sums
pd.set_option('display.width',250)
ns=argparse.Namespace(vintage="2026-11-08",reference_date="2026-11-14",data="data",backtest_config="config/backtest.toml")
task,_=cli._task(ns)
o=rw.RandomWalkNowcast().origin_at(task, errors=False)
s=o.snap; locs=list(s.locations); oo=o.o
rep_o=weekly_sums(s,oo)
rows=[]
for r in range(1,oo):
    age=s.cut[r]-np.arange(s.week.size); gap=int(s.cut[oo]-s.cut[r])
    pred_now=week_nowcast(s,r,o.fit.g,age,gap)   # 11-08 delay, growth to 11-08
    fr=fit_delay(s,r,o.settings)
    pred_r=week_nowcast(s,r,fr.g,age,gap) if fr is not None else np.full_like(pred_now,np.nan)
    rep_r=weekly_sums(s,r); wa=s.cut_week(r)
    for c in ['CD','CD-IT','CD-NK','CD-HU','CD-TO']:
        i=locs.index(c)
        for k in range(0,4):
            w=wa-k
            rows.append(dict(release=s.releases[r],loc=c,k=k,week=s.week_ends[w],rep_r=rep_r[i,w],
              rep_1108=rep_o[i,w],pred_g1108=round(pred_now[i,w],1),pred_g_r=round(pred_r[i,w],1),
              obs_ratio=round(rep_o[i,w]/max(rep_r[i,w],1),2),ratio_g1108=round(pred_now[i,w]/max(rep_r[i,w],1),2)))
d=pd.DataFrame(rows)
print(d.to_string(index=False))
d.to_csv('/tmp/claude-1000/-home-seabbs-code-seabbs-BDBVHubRenewal/54f1e2de-e074-4f63-90d5-9393f9cb5b1b/scratchpad/ex/backcheck.csv',index=False)
g=d[d.k.isin([1,2,3])].groupby(['loc','k'])[['rep_1108','pred_g1108','pred_g_r']].sum()
g['bias_g1108']=(g.pred_g1108/g.rep_1108).round(3); g['bias_g_r']=(g.pred_g_r/g.rep_1108).round(3); print(g)
print('releases', s.releases, 'cuts', s.cut)
