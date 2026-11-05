"""Figure grid of the flagged places, one per model."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

S = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SAM = ["CD-HU-11", "CD-HU-07", "CD-HU-12", "CD-TO-02"]
ORDER = ["median high", "median low", "trend unsupported",
         "interval narrow", "upper tail"]

ts = pd.read_csv(S / "hub/ts.csv")
ts = ts[ts.target_end_date >= "2026-08-29"]
ts["date"] = pd.to_datetime(ts.target_end_date)
q = pd.read_csv(S / "quantiles.csv")
q["date"] = pd.to_datetime(q.target_end_date)
flags = pd.read_csv(S / "flags.csv")
blue = "#2a6fdb"

for model, f in flags.groupby("model"):
    f = f.assign(rank=f.flag.map(ORDER.index))
    first = f.groupby("location")["rank"].min().sort_values()
    locs = [x for x in SAM if x in first.index]
    locs += [x for x in first.index if x not in locs]
    locs = locs[:16]
    ncol = 4
    nrow = -(-len(locs) // ncol)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(13, 2.7 * nrow), squeeze=False
    )
    for ax, loc in zip(axes.flat, locs, strict=False):
        g = q[(q.model == model) & (q.location == loc)]
        ax.fill_between(g.date, g.q05, g.q95, color=blue, alpha=0.2,
                        lw=0, label="90% interval")
        ax.plot(g.date, g.q50, color=blue, lw=2, label="median")
        for as_of, style in [("2026-10-24", dict(mfc="white", mec="grey")),
                             ("2026-11-01", dict(mfc="black", mec="black"))]:
            r = ts[(ts.location == loc) & (ts.as_of == as_of)]
            ax.plot(r.date, r.observation, "o", ms=4, ls="none",
                    label=f"reported {as_of}", **style)
        name = ts.loc[ts.location == loc, "location_name"].iloc[0]
        why = ", ".join(f.loc[f.location == loc, "flag"])
        ax.set_title(f"{name} ({loc})\n{why}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(locs):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        f"{model}: flagged places at the 2026-11-07 round "
        f"(vintage 2026-11-01), {len(locs)} of {first.size} shown",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / f"{model}.png", dpi=110)
    plt.close(fig)
    print(model, len(locs), first.size)
