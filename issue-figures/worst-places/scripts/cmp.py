import pandas as pd, sys
S=sys.argv[1]
sub=pd.read_csv("/home/seabbs/bdbv-perf/hub/model-output/seabbs-rwnowcast/2026-11-07-seabbs-rwnowcast.csv")
new=pd.read_csv(S+"/rwnc-main-1101/model-output/seabbs-rwnowcast/2026-11-07-seabbs-rwnowcast.csv")
locs="CD-HU CD-HU-02 CD-HU-07 CD-HU-10 CD-HU-11 CD-IT-28 CD-IT-29 CD-IT-07 CD-IT-13 CD-IT-17 CD-NK-25 CD-NK-26 CD-IT-27 CD-IT CD-NK CD".split()
def s(d,l,w):
  v=d[(d.location==l)&(d.target_end_date==w)].value
  return f"{v.median():.0f} [{v.quantile(.05):.0f},{v.quantile(.95):.0f}] P0={(v==0).mean():.2f}"
for w in ["2026-10-31","2026-11-07"]:
  print(w)
  for l in locs: print(f"  {l:9s} submitted {s(sub,l,w):28s} main {s(new,l,w)}")
