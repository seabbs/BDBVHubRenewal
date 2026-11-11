"""Fan charts of the national forecasts against the reported counts."""
import glob
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bdbvhubrenewal.evaluation.store import read_fit

store, out = sys.argv[1], sys.argv[2]
t = pd.read_csv("data/target.csv")
models = ["fm_chronos_weekly_complete", "fm_chronos_weekly_ragged",
          "fm_chronos_daily_complete", "fm_chronos_daily_ragged"]
fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey=True)
for row, origin in enumerate(["2026-10-24", "2026-11-01"]):
    for col, m in enumerate(models):
        ax = axes[row, col]
        f = read_fit(glob.glob(f"{store}/{m}/{origin}/*.json.gz")[0]).forecasts
        f = f[(f.location == "CD") & f.as_of.isna()]
        q = f.groupby("target_end_date")["value"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
        x = pd.to_datetime(q.index)
        ax.fill_between(x, q[0.05], q[0.95], alpha=0.2, color="C0")
        ax.fill_between(x, q[0.25], q[0.75], alpha=0.4, color="C0")
        ax.plot(x, q[0.5], color="C0", label="median")
        for v, c in ((origin, "k"), ("2026-11-08", "C3")):
            r = t[(t.vintage == v) & (t.location == "CD")]
            r = r[r.target_end_date >= q.index.min()]
            ax.plot(pd.to_datetime(r.target_end_date), r.total, "o-", color=c, ms=3, label=f"reported {v}")
        ax.axvline(pd.Timestamp(origin), color="grey", lw=0.5)
        ax.set_title(f"{m.replace('fm_chronos_', '')} at {origin}", fontsize=9)
        ax.tick_params(labelsize=7)
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(out, dpi=80)
