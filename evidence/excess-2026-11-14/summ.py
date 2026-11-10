import sys, numpy as np, pandas as pd
from pathlib import Path
out=Path(sys.argv[1]); tags=sys.argv[2].split(',')
locs=(out/'locs.txt').read_text().split('\n'); weeks=(out/'weeks.txt').read_text().split('\n')
pl=pd.read_csv(out/'places.csv')
nat=locs.index('CD'); provs=[i for i,p in enumerate(pl.parent) if p==nat]
show=['2026-10-17','2026-10-24','2026-10-31','2026-11-07','2026-11-14','2026-12-05']
cols=[weeks.index(w) for w in show]
def q(x): return f"{np.median(x):.0f} [{np.quantile(x,.05):.0f},{np.quantile(x,.95):.0f}]"
for t in tags:
    v=np.load(out/f'values_{t}.npy'); v=np.maximum(np.round(v),0)
    natv=v[:,provs,:].sum(1)
    print(t,'| national', ' | '.join(f"{w[5:]} {q(natv[:,c])}" for w,c in zip(show,cols)))
    if len(sys.argv)>3:
      for i in provs:
        if v[:,i,cols[-2]].max()>0 and np.median(v[:,i,cols[-2]])>=3:
          print('   ',locs[i], ' | '.join(f"{w[5:]} {q(v[:,i,c])}" for w,c in zip(show,cols)))
