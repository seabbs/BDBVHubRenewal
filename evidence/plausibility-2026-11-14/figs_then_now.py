"""Last week's against this week's forecasts for CD and four provinces.

Usage: figs_then_now.py <S> <subs dir> <out png>
Reads `<S>/q1107.csv` (11-07 candidates), the two submitted files and
`<S>/quantiles.csv` (this round, from `flags3.py`).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

S, SUBS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
LOCS = ["CD", "CD-IT", "CD-NK", "CD-HU", "CD-TO"]
MODELS = ["randomwalknowcast", "pprandomwalknowcast", "randomwalk",
          "renewal", "renewalsp", "renewalbvd"]
SUBMITTED = {
    "randomwalknowcast": "seabbs-rwnowcast/2026-11-07-seabbs-rwnowcast.csv.gz",
    "renewalbvd": "seabbs-renewal/2026-11-07-seabbs-renewal.csv.gz",
}
then = pd.read_csv(S / "q1107.csv")
for model, f in SUBMITTED.items():
    d = pd.read_csv(SUBS / f)
    q = d.groupby(["location", "target_end_date"]).value.quantile(
        [0.05, 0.5, 0.95]
    ).unstack()
    q.columns = ["q05", "q50", "q95"]
    then = pd.concat([then[then.model != model],
                      q.reset_index().assign(model=model)])
now = pd.read_csv(S / "quantiles.csv")
ts = pd.read_csv(S / "hub/ts.csv")
for d in (then, now, ts):
    d["date"] = pd.to_datetime(d.target_end_date)
then, now = then[then.date >= "2026-09-26"], now[now.date >= "2026-09-26"]
ts = ts[ts.date >= "2026-09-12"]
fig, axes = plt.subplots(len(MODELS), len(LOCS), figsize=(17, 2.6 * 6))
for i, model in enumerate(MODELS):
    for j, loc in enumerate(LOCS):
        ax = axes[i, j]
        for q, col, lab in [(then, "#999999", "11-07 round"),
                            (now, "#2a6fdb", "11-14 round")]:
            g = q[(q.model == model) & (q.location == loc)].sort_values("date")
            ax.fill_between(g.date, g.q05, g.q95, color=col, alpha=0.2, lw=0)
            ax.plot(g.date, g.q50, color=col, lw=2, label=lab)
        for as_of, st in [("2026-11-01", dict(mfc="white", mec="grey")),
                          ("2026-11-08", dict(mfc="black", mec="black"))]:
            r = ts[(ts.location == loc) & (ts.as_of == as_of)]
            ax.plot(r.date, r.observation, "o", ms=3, ls="none",
                    label=f"reported {as_of}", **st)
        ax.set_title(f"{model}: {loc}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
axes[0, 0].legend(fontsize=6, loc="lower left")
fig.suptitle("Median and 90% interval: 11-07 round (grey; submitted files for"
             " rwnowcast and renewalbvd) against the 11-14 round (blue)",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=90)
