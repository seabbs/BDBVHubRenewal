"""Draw our forecasts against the hub's, one panel per place.

Usage: plot.py <evidence dir>

Reads `quantiles.csv` from `compare.py` and the 2026-11-08 release.
Hub models (merged and PR #114, baseline left out) are grey: median line
and a light 90% band; the dashed line is the hub ensemble median
(two or more models). Ours are coloured: median line and a 90% band.
Black dots are the counts in the 2026-11-08 release (the last two or
three weeks are still filling). Each panel's y axis stops at 2.5 times
the largest of the reported counts and the hub q95; a model whose q95
goes past it is named in the panel's corner with its largest q95.

Writes `provinces.png` (nation and provinces) and `zones.png` (the zones
the hub models give, most cases first).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

E = Path(sys.argv[1])
OURS = {
    "randomwalknowcast": "#2a78d6",
    "pprandomwalknowcast": "#eb6834",
    "randomwalk": "#1baf7a",
    "renewal": "#eda100",
    "renewalsp": "#e87ba4",
    "renewalbvd": "#008300",
}
GREY = "#8a8985"
q = pd.read_csv(E / "quantiles.csv", parse_dates=["target_end_date"])
t = pd.read_csv("data/target.csv")
z = pd.read_csv("data/health_zones.csv.gz")
data = pd.concat([t[t.vintage == "2026-11-08"], z[z.vintage == "2026-11-08"]])
data["target_end_date"] = pd.to_datetime(data.target_end_date)
data = data[data.target_end_date >= "2026-08-15"]
names = dict(zip(data.location, data.location_name, strict=False))
hub = q[q.group == "hub"]


def panel(ax, loc):
    d = data[data.location == loc]
    h = hub[hub.location == loc]
    o = q[(q.group == "ours") & (q.location == loc)]
    top = max(d.total.max() if len(d) else 0, h.q95.max() if len(h) else 0)
    cap = 2.5 * max(top, 5)
    for _, g in h.groupby("model"):
        g = g.sort_values("target_end_date")
        ax.fill_between(g.target_end_date, g.q05, g.q95, color=GREY,
                        alpha=0.12, lw=0)
        ax.plot(g.target_end_date, g.q50, color=GREY, lw=1.2)
    e = q[(q.group == "ensemble") & (q.location == loc)]
    e = e.sort_values("target_end_date")
    ax.plot(e.target_end_date, e.q50, color="#3a3936", lw=1.4, ls="--")
    over = []
    for m, col in OURS.items():
        g = o[o.model == m].sort_values("target_end_date")
        if g.empty:
            continue
        ax.fill_between(g.target_end_date, g.q05, g.q95, color=col,
                        alpha=0.15, lw=0)
        ax.plot(g.target_end_date, g.q50, color=col, lw=2)
        if g.q95.max() > cap:
            over.append(f"{m} q95 to {g.q95.max():.0f}")
    ax.plot(d.target_end_date, d.total, "o", color="#0b0b0b", ms=3)
    ax.axvline(pd.Timestamp("2026-11-08"), color="#52514e", lw=0.8, ls=":")
    ax.set_ylim(0, cap)
    ax.set_title(f"{names.get(loc, loc)} ({loc})"
                 + ("" if h.model.nunique() else " - no hub model"),
                 fontsize=9, loc="left")
    if over:
        ax.text(0.02, 0.97, "\n".join(over), transform=ax.transAxes,
                va="top", fontsize=6, color="#52514e")
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", color="#e5e4e0", lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))


def grid(locs, path, ncol, title):
    nrow = -(-len(locs) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.6 * nrow),
                             squeeze=False)
    for ax, loc in zip(axes.flat, locs, strict=False):
        panel(ax, loc)
    for ax in axes.flat[len(locs):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=GREY, lw=1.2, label="hub models")]
    handles += [plt.Line2D([], [], color="#3a3936", lw=1.4, ls="--",
                           label="hub ensemble median")]
    handles += [plt.Line2D([], [], color=c, lw=2, label=m)
                for m, c in OURS.items()]
    handles += [plt.Line2D([], [], color="#0b0b0b", marker="o", ls="",
                           ms=3, label="reported, 2026-11-08 release")]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, y=1.03, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


prov = sorted(
    {loc for loc in q.location.unique() if loc.count("-") == 1},
    key=lambda x: -data[data.location == x].total.sum(),
)
grid(["CD", *prov], E / "provinces.png", 3,
     "2026-11-14 round: median and 90% interval, hub models (grey) and ours")
zones = [loc for loc in hub.location.unique() if loc.count("-") == 2]
zones = sorted(zones, key=lambda x: -data[data.location == x].total.sum())
grid(zones, E / "zones.png", 6,
     "2026-11-14 round, zones the hub models give (most cases first)")
print(len(prov), "provinces", len(zones), "zones")
