"""Fans of one ragged model, data-driven D against D = 4 weeks / 28 days,
for the country and Ituri at 2026-10-24 and 2026-11-01."""
import glob
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bdbvhubrenewal.evaluation.store import read_fit

model, base, dmax, out = sys.argv[1:5]
t = pd.read_csv("data/target.csv")
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for col, (origin, loc) in enumerate(
    [("2026-10-24", "CD"), ("2026-10-24", "CD-IT"), ("2026-11-01", "CD"), ("2026-11-01", "CD-IT")]
):
    for row, (label, root) in enumerate([("D = 4 w / 28 d", base), ("D from the data", dmax)]):
        ax = axes[row, col]
        f = read_fit(glob.glob(f"{root}/{model}/{origin}/*.json.gz")[0]).forecasts
        f = f[(f.location == loc) & f.as_of.isna()]
        q = f.groupby("target_end_date")["value"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
        x = pd.to_datetime(q.index)
        ax.fill_between(x, q[0.05], q[0.95], alpha=0.2, color="C0")
        ax.fill_between(x, q[0.25], q[0.75], alpha=0.4, color="C0")
        ax.plot(x, q[0.5], color="C0", label="median")
        for v, c in ((origin, "k"), ("2026-11-08", "C3")):
            r = t[(t.vintage == v) & (t.location == loc) & (t.target_end_date >= q.index.min())]
            ax.plot(pd.to_datetime(r.target_end_date), r.total, "o-", color=c, ms=3, label=f"reported {v}")
        ax.axvline(pd.Timestamp(origin), color="grey", lw=0.5)
        ax.set_title(f"{label}: {loc} at {origin}", fontsize=9)
        ax.tick_params(labelsize=7)
axes[0, 0].legend(fontsize=7)
fig.suptitle(model, fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=80)
