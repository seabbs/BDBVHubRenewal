import sys, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as md
S, OUT = sys.argv[1], sys.argv[2]
H="/home/seabbs/bdbv-perf/hub/"
t=pd.read_csv(H+"target-data/time-series.csv",parse_dates=["target_end_date"])
sc=pd.read_csv("/home/seabbs/bdbv-perf/scores_cells.csv",parse_dates=["target_end_date"])
OURS=["seabbs-renewal","seabbs-rwnowcast"]
ASOF={"2026-11-07":"2026-11-01","2026-11-14":"2026-11-08"}
COL={"seabbs-renewal":"#2a78d6","seabbs-rwnowcast":"#eb6834","main":"#1baf7a"}
_cache={}
def samples(model,r):
  k=(model,r)
  if k not in _cache:
    f=S+"/rwnc-main-1101/model-output/seabbs-rwnowcast/2026-11-07-seabbs-rwnowcast.csv" if model=="main" else H+f"model-output/{model}/{r}-{model}.csv"
    _cache[k]=pd.read_csv(f,parse_dates=["target_end_date"])
  return _cache[k]
LAB={"seabbs-renewal":"seabbs-renewal (submitted)","seabbs-rwnowcast":"seabbs-rwnowcast (submitted)","main":"randomwalknowcast on main, rerun"}
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#52514e","axes.labelcolor":"#0b0b0b"})
def fig(panels,fname,title):
  fig,axs=plt.subplots(1,len(panels),figsize=(4.6*len(panels),3.4),squeeze=False)
  for ax,(loc,name,r,models) in zip(axs[0],panels):
    a=ASOF[r]
    for v in sorted(t.as_of.unique()):
      if v<"2026-10-17": continue
      d=t[(t.location==loc)&(t.as_of==v)&(t.target_end_date>="2026-09-20")]
      if v=="2026-11-15": ax.plot(d.target_end_date,d.observation,color="#0b0b0b",lw=2,marker="o",ms=4,label="data, 11-15 release",zorder=5)
      elif v==a: ax.plot(d.target_end_date,d.observation,color="#0b0b0b",lw=1.2,ls="--",marker="o",ms=3,mfc="white",label=f"data at forecast ({a[5:]} release)",zorder=4)
      else: ax.plot(d.target_end_date,d.observation,color="#b5b4ae",lw=1,zorder=1,label="older releases" if v=="2026-10-17" else None)
    o=sc[(sc.location==loc)&(sc.reference_date==r)&~sc.model.isin(OURS)].groupby("target_end_date")["median"].median()
    ax.plot(o.index,o.values,color="#6250d6",lw=1.5,ls=":",label="other hub teams, median of medians",zorder=3)
    for m in models:
      d=samples(m,r); d=d[(d.location==loc)&(d.target_end_date>="2026-10-17")]
      q=d.groupby("target_end_date").value.quantile([.05,.5,.95]).unstack()
      ax.fill_between(q.index,q[.05],q[.95],color=COL[m],alpha=.18,lw=0)
      ax.plot(q.index,q[.5],color=COL[m],lw=2,label=LAB[m]+", median, 90%",zorder=4)
    ax.axvline(pd.Timestamp(a),color="#52514e",lw=.8)
    ax.set_title(f"{name} ({loc}), round {r}",fontsize=9,loc="left")
    ax.xaxis.set_major_formatter(md.DateFormatter("%m-%d")); ax.set_ylim(bottom=0)
    ax.set_ylabel("weekly confirmed cases"); ax.grid(axis="y",color="#e8e7e3",lw=.6)
  h,l=[],[]
  for ax in axs[0]:
    for hh,ll in zip(*ax.get_legend_handles_labels()):
      if ll not in l: h.append(hh); l.append(ll)
  fig.legend(h,l,loc="lower center",ncol=3,frameon=False,fontsize=8)
  fig.suptitle(title,x=.01,ha="left",fontsize=10)
  fig.tight_layout(rect=(0,.14,1,.95)); fig.savefig(f"{OUT}/{fname}",dpi=130,facecolor="white"); plt.close(fig)
R="seabbs-renewal"; W="seabbs-rwnowcast"
fig([("CD","National","2026-11-07",[R,W]),("CD","National","2026-11-14",[R,W]),("CD-IT","Ituri","2026-11-14",[R,W])],"01-national-ituri-overshoot.png","Both models nowcast weeks 1-3 weeks old above where they settled")
fig([("CD-NK-14","Katwa","2026-11-07",[R]),("CD-NK-14","Katwa","2026-11-14",[R]),("CD-TO-02","Bafwasende","2026-11-14",[R])],"02-renewal-zone-share.png","seabbs-renewal zones: split share from the 16-week window")
fig([("CD-HU","Haut-Uele","2026-11-07",[W,"main"]),("CD-HU-02","Boma Mangbetu","2026-11-07",[W,"main"]),("CD-HU-10","Pawa","2026-11-07",[W,"main"])],"03-rwnowcast-hu-1107.png","seabbs-rwnowcast 11-07: Haut-Uele jump split onto Boma Mangbetu; Pawa all 0")
fig([("CD-IT-28","Mongbwalu","2026-11-07",[W,"main"]),("CD-HU-07","Isiro","2026-11-07",[W,"main"]),("CD-IT","Ituri","2026-11-07",[W,"main"])],"04-rwnowcast-zero-zones-main.png","randomwalknowcast at 11-01 data: submitted vs rerun on main")
