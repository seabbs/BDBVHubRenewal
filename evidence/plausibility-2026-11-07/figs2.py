"""Sam's places for each random walk model, and Mutwanga against the hub."""

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
PLACES = ["CD-HU-11", "CD-IT-22", "CD-HU-07", "CD-HU-12", "CD-TO-02",
          "CD-NK-20", "CD-NK-26", "CD-NK-30"]
ts = pd.read_csv(S / "hub/ts.csv")
ts = ts[ts.target_end_date >= "2026-08-29"]
ts["date"] = pd.to_datetime(ts.target_end_date)
names = ts.drop_duplicates("location").set_index("location").location_name
q = pd.read_csv(S / "quantiles.csv")
q["date"] = pd.to_datetime(q.target_end_date)
flags = pd.read_csv(S / "flags2.csv")
blue = "#2a6fdb"


def observed(ax, loc):
    for as_of, style in [("2026-10-24", dict(mfc="white", mec="grey")),
                         ("2026-11-01", dict(mfc="black", mec="black"))]:
        r = ts[(ts.location == loc) & (ts.as_of == as_of)]
        ax.plot(r.date, r.observation, "o", ms=4, ls="none",
                label=f"reported {as_of}", **style)


def tidy(ax):
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


for model in ["randomwalknowcast", "pprandomwalknowcast", "randomwalk"]:
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    for ax, loc in zip(axes.flat, PLACES, strict=True):
        g = q[(q.model == model) & (q.location == loc)]
        ax.fill_between(g.date, g.q05, g.q95, color=blue, alpha=0.2, lw=0,
                        label="90% interval")
        ax.plot(g.date, g.q50, color=blue, lw=2, label="median")
        observed(ax, loc)
        why = ", ".join(flags.loc[(flags.model == model)
                                  & (flags.location == loc), "flag"])
        ax.set_title(f"{names[loc]} ({loc})\n{why or 'not flagged'}",
                     fontsize=9)
        tidy(ax)
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{model} at the 2026-11-07 round (data as of 2026-11-01)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{model}-places.png", dpi=110)
    plt.close(fig)

# Mutwanga: each hub team's median and 90% interval
fig, ax = plt.subplots(figsize=(8, 4))
files = sorted(glob.glob(str(S / "hubsubs/*.csv")))
teams = [f.split("2026-11-07-")[1][:-4] for f in files]
off = np.linspace(-2.5, 2.5, len(files))
for k, (f, team) in enumerate(zip(files, teams, strict=True)):
    d = pd.read_csv(f, low_memory=False)
    d = d[(d.location == "CD-NK-26") & (d.target_end_date >= "2026-10-24")]
    if d.empty:
        continue
    rows = []
    for ted, g in d.groupby("target_end_date"):
        if (g.output_type == "sample").any():
            v = g.loc[g.output_type == "sample", "value"]
            rows.append((ted, *v.quantile([0.05, 0.5, 0.95])))
        else:
            g = g[g.output_type == "quantile"]
            s = g.set_index(g.output_type_id.astype(float).round(3)).value
            rows.append((ted, s.get(0.05), s.get(0.5), s.get(0.95)))
    r = pd.DataFrame(rows, columns=["ted", "lo", "med", "hi"])
    x = pd.to_datetime(r.ted) + pd.to_timedelta(off[k], unit="D")
    ours = team == "seabbs-rwnowcast"
    ax.errorbar(x, r.med, yerr=[r.med - r.lo, r.hi - r.med], fmt="o", ms=4,
                lw=1.5 if ours else 0.8, color=blue if ours else "grey",
                alpha=1 if ours else 0.6, label=team if ours else None)
ax.plot([], [], "o", color="grey", alpha=0.6, label="other hub teams")
observed(ax, "CD-NK-26")
ax.set_title("Mutwanga (CD-NK-26): submitted 2026-11-07 forecasts, "
             "median and 90% interval", fontsize=10)
tidy(ax)
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "mutwanga-hub.png", dpi=110)
