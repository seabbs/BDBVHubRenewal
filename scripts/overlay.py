"""Overlay of the Chronos-2 models and randomwalknowcast for one source,
for the country and Ituri at 2026-10-24 and 2026-11-01.

The ragged models, zero-shot and fine-tuned, are at D = 4 w / 28 d.

Usage: overlay.py SOURCE FT_STORE BASE_STORE RW_FORECASTS OUT, from the
fm-finetune worktree's root. RW_FORECASTS is the forecasts.csv.gz of a
release backtest with randomwalknowcast (as score.sh writes).
"""
import glob
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bdbvhubrenewal.evaluation.store import read_fit

source, ft_store, base_store, rw_path, out = sys.argv[1:6]
latest = "2026-11-08"
m = f"fm_chronos_{source}"
# Okabe-Ito colours
models = [
    ("Chronos-2 ragged, fine-tuned", ft_store, f"{m}_ragged_ft", "#0072B2"),
    ("Chronos-2 ragged, zero-shot", ft_store, f"{m}_ragged", "#009E73"),
    ("Chronos-2 complete", base_store, f"{m}_complete", "#E69F00"),
    ("randomwalknowcast", None, "randomwalknowcast", "#CC79A7"),
]
t = pd.read_csv("data/target.csv")
rw = pd.read_csv(rw_path)
rw = rw[(rw.model == "randomwalknowcast") & rw.as_of.isna()]


def forecasts(store, name, origin):
    if store is None:
        return rw[rw.vintage == origin]
    f = read_fit(glob.glob(f"{store}/{name}/{origin}/*.json.gz")[0])
    return f.forecasts[f.forecasts.as_of.isna()]


fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
for row, origin in enumerate(["2026-10-24", "2026-11-01"]):
    for col, loc in enumerate(["CD", "CD-IT"]):
        ax = axes[row, col]
        start = None
        top = 0
        for label, store, name, c in models:
            f = forecasts(store, name, origin)
            f = f[f.location == loc]
            q = (f.groupby("target_end_date")["value"]
                 .quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack())
            x = pd.to_datetime(q.index)
            start = min(start or q.index.min(), q.index.min())
            top = max(top, q[0.5].max())
            ax.fill_between(x, q[0.05], q[0.95], color=c, alpha=0.06, lw=0)
            ax.fill_between(x, q[0.25], q[0.75], color=c, alpha=0.25, lw=0)
            ax.plot(x, q[0.5], color=c, lw=1.8, label=label)
        for v, c, ls, lab in (
            (origin, "black", "-", "reported at origin"),
            (latest, "grey", "--", "latest release (incomplete)"),
        ):
            r = t[(t.vintage == v) & (t.location == loc)
                  & (t.target_end_date >= start)]
            ax.plot(pd.to_datetime(r.target_end_date), r.total, marker="o",
                    ls=ls, color=c, ms=3.5, lw=1.3, label=lab, zorder=5)
            top = max(top, r.total.max())
        ax.axvline(pd.Timestamp(origin), color="black", lw=0.8, ls=":",
                   label="origin")
        place = "national" if loc == "CD" else "Ituri"
        ax.set_title(f"{place} ({loc}), origin {origin}", fontsize=10)
        ax.set_ylim(0, 2 * top)
        ax.tick_params(labelsize=8)
        ax.tick_params(axis="x", labelrotation=30)
        if col == 0:
            ax.set_ylabel("weekly cases", fontsize=9)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, 0.995))
fig.suptitle(f"{source} source: median, 50% and 90% intervals; "
             "ragged models at D = 4 w / 28 d; y-axis cut at twice the "
             "highest median or count",
             fontsize=10, y=0.925)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(out, dpi=100)
