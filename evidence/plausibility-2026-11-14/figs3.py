"""Figure grid per model: national, the four provinces, then flagged zones.

Usage: figs3.py <S> <out dir> [model ...]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

S, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
FIXED = ["CD", "CD-IT", "CD-NK", "CD-HU", "CD-TO"]
ORDER = ["blow-up", "collapse", "no cases ever", "uplift", "step-up",
         "upper tail", "interval wide", "median high", "median low",
         "trend unsupported", "interval narrow", "interval narrow at H3"]

ts = pd.read_csv(S / "hub/ts.csv")
ts = ts[ts.target_end_date >= "2026-08-29"]
ts["date"] = pd.to_datetime(ts.target_end_date)
q = pd.read_csv(S / "quantiles.csv")
q = q[q.target_end_date >= "2026-09-26"]
q["date"] = pd.to_datetime(q.target_end_date)
flags = pd.read_csv(S / "flags3.csv")
names = flags.drop_duplicates("location").set_index("location")["name"]
blue = "#2a6fdb"
models = sys.argv[3:] or sorted(q.model.unique())

for model in models:
    f = flags[flags.model == model]
    f = f.assign(rank=f.flag.map(ORDER.index))
    first = f[f.scale == "zone"].groupby("location")["rank"].min()
    zones = list(first.sort_values().index)
    locs = FIXED + zones[: 20 - len(FIXED)]
    ncol = 5
    nrow = -(-len(locs) // ncol)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(16, 2.8 * nrow), squeeze=False
    )
    for ax, loc in zip(axes.flat, locs, strict=False):
        g = q[(q.model == model) & (q.location == loc)]
        ax.fill_between(g.date, g.q05, g.q95, color=blue, alpha=0.15,
                        lw=0, label="90% interval")
        ax.fill_between(g.date, g.q25, g.q75, color=blue, alpha=0.3,
                        lw=0, label="50% interval")
        ax.plot(g.date, g.q50, color=blue, lw=2, label="median")
        for as_of, style in [("2026-11-01", dict(mfc="white", mec="grey")),
                             ("2026-11-08", dict(mfc="black", mec="black"))]:
            r = ts[(ts.location == loc) & (ts.as_of == as_of)]
            # zones have no row in weeks with no case
            weeks = sorted(ts.loc[ts.as_of == as_of, "date"].unique())
            r = r.set_index("date").reindex(weeks).fillna({"observation": 0})
            r = r.rename_axis("date").reset_index()
            ax.plot(r.date, r.observation, "o", ms=3.5, ls="none",
                    label=f"reported {as_of}", **style)
        name = names.get(loc)
        if not isinstance(name, str):
            hit = ts.loc[ts.location == loc, "location_name"]
            name = hit.iloc[0] if len(hit) else loc
        why = ", ".join(f.loc[f.location == loc, "flag"]) or "not flagged"
        ax.set_title(f"{name} ({loc})\n{why}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(locs):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=6, loc="lower left")
    fig.suptitle(
        f"{model} at the 2026-11-14 round (data as of 2026-11-08): national,"
        f" provinces and {len(locs) - len(FIXED)} of {len(zones)} flagged"
        " zones",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / f"{model}.png", dpi=100)
    plt.close(fig)
    print(model, len(zones))
