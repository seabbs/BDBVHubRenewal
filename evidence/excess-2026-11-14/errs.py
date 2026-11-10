import argparse, dataclasses, numpy as np, pandas as pd
from bdbvhubrenewal import cli
from bdbvhubrenewal.forecasters import randomwalk as rw
ns=argparse.Namespace(vintage="2026-11-08",reference_date="2026-11-14",data="data",backtest_config="config/backtest.toml")
task,_=cli._task(ns)
o=rw.RandomWalkNowcast().origin_at(task, errors=True)
e=o.errors; locs=list(o.snap.locations)
print("releases", e.releases)
for c in ['CD','CD-NK','CD-IT','CD-HU']:
    i=locs.index(c)
    print(c, 'obs_n', o.obs_n[i])
    for r,rel in enumerate(e.releases):
        print('  ',rel,'reported',e.reported[r,i].round(0),'\n      remaining',e.remaining[r,i].round(0),'\n      added',e.added[r,i].round(0), 'removed', e.removed[r,i].round(0))
