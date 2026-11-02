"""Figures for rwnc-eval2."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

out = Path(sys.argv[1]); fig_dir = out.parent; hubfile = sys.argv[2]
COL = {"randomwalknowcast": "#2a78d6", "randomwalk": "#eb6834",
       "randomwalknowcast_skellam_rem": "#1baf7a", "randomwalknowcast_nb": "#6250d6",
       "as reported": "#7a7a75"}
LAB = {"randomwalknowcast": "randomwalknowcast (Skellam)", "randomwalk": "randomwalk",
       "randomwalknowcast_skellam_rem": "rwnc Skellam on remainder",
       "randomwalknowcast_nb": "rwnc NB on remainder", "as reported": "as reported"}
NAME = {"CD": "National", "CD-IT": "Ituri", "CD-NK": "North Kivu", "CD-HU": "Haut-Uele",
        "CD-TO": "Tshopo", "CD-BU": "Bas-Uele"}
PLACES = list(NAME)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#999", "figure.facecolor": "white"})
target = pd.read_csv("data/target.csv"); target = target[target.kind == "real"]
REL = {"2026-10-17": "o", "2026-10-24": "s", "2026-11-01": "D"}
RELC = {"2026-10-17": "#9a9a95", "2026-10-24": "#555550", "2026-11-01": "#111110"}
CUTW = {"2026-10-17": "2026-10-17", "2026-10-24": "2026-10-24", "2026-11-01": "2026-10-31"}


def band(ax, q, model, off=0.0):
    q = q.sort_values("target_end_date")
    x = pd.to_datetime(q["target_end_date"]) + pd.Timedelta(days=off)
    c = COL[model]
    ax.fill_between(x, q["q5"], q["q95"], color=c, alpha=0.18, lw=0)
    ax.fill_between(x, q["q25"], q["q75"], color=c, alpha=0.35, lw=0)
    ax.plot(x, q["q50"], color=c, lw=2, label=LAB[model])


def releases(ax, loc, origin, lo, hi):
    for v in [r for r in REL if r >= origin]:
        t = target[(target.vintage == v) & (target.location == loc)]
        t = t[(t.target_end_date >= lo) & (t.target_end_date <= min(hi, CUTW[v]))]
        ax.plot(pd.to_datetime(t.target_end_date), t.total, ls="none", marker=REL[v],
                ms=5, mfc="white" if v == origin else RELC[v], mec=RELC[v], mew=1.2,
                label=f"release {v[5:]}" + (" (origin)" if v == origin else ""), zorder=5)


def fan_grid(fans, origins, models, places, name, title, hub=None, weeks=(-4, 7)):
    fig, axes = plt.subplots(len(places), len(origins), figsize=(max(4.6 * len(origins), 7.5), 2.3 * len(places)),
                             squeeze=False)
    for j, o in enumerate(origins):
        for i, loc in enumerate(places):
            ax = axes[i, j]
            f = fans[(fans.origin == o) & (fans.location == loc) & fans.weeks_back.between(*weeks)]
            for m in models:
                band(ax, f[f.model == m], m)
            lo, hi = f.target_end_date.min(), f.target_end_date.max()
            releases(ax, loc, o, lo, hi)
            if hub is not None:
                h = hub[(hub.location == loc) & (hub.target_end_date >= lo)]
                x = pd.to_datetime(h.target_end_date) + pd.Timedelta(days=1.5)
                ax.errorbar(x, h.q50, yerr=[h.q50 - h.q5, h.q95 - h.q50], fmt="_", color="#111110",
                            ms=8, lw=1.2, capsize=0, label="hub file median, 90%", zorder=6)
            ax.axvline(pd.Timestamp(CUTW[o]) + pd.Timedelta(days=0.5), color="#999", lw=0.8, ls=":")
            ax.set_title(f"{NAME.get(loc, loc)}, origin {o[5:]}", fontsize=9, loc="left")
            ax.set_ylim(bottom=0)
            ax.tick_params(axis="x", labelrotation=0, labelsize=7)
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d"))
    h, l = axes[0, -1].get_legend_handles_labels()
    seen = dict(zip(l, h))
    for ax in axes.flat:
        hh, ll = ax.get_legend_handles_labels()
        seen.update(dict(zip(ll, hh)))
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=4 if len(origins) > 1 else 3, frameon=False, fontsize=8)
    import textwrap
    fig.suptitle("\n".join(textwrap.wrap(title, 95 if len(origins) > 1 else 80)), x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(fig_dir / name, dpi=130)
    plt.close(fig)


fans = pd.read_csv(out / "fans.csv")
fan_grid(fans, ["2026-10-17", "2026-10-24"], ["randomwalk", "randomwalknowcast"], PLACES,
         "fans-origins.png",
         "Final-count forecasts with 50% and 90% intervals, by origin. Open marker: data at origin; filled: later releases. Dotted: cut-off week.")
fan_grid(fans, ["2026-10-24"], ["randomwalknowcast", "randomwalknowcast_skellam_rem", "randomwalknowcast_nb"],
         PLACES, "fans-variants-1024.png", "Observation-model variants at origin 10-24 (50%, 90%)")
hubf = pd.read_csv(hubfile)
hubq = (hubf.groupby(["location", "target_end_date"])["value"].quantile([0.05, 0.5, 0.95]).unstack())
hubq.columns = ["q5", "q50", "q95"]; hubq = hubq.reset_index()
fan_grid(fans, ["2026-11-01"], ["randomwalk", "randomwalknowcast"], PLACES, "fans-1101-submission.png",
         "11-01 release (submission): randomwalknowcast vs randomwalk, with the hub file's median and 90%",
         hub=hubq, weeks=(-4, 7))
# zones
zf = pd.read_csv(out / "zone-fans.csv")
zt = pd.read_csv(out / "zone-target.csv")
top = (zt[zt.vintage == "2026-11-01"].groupby("location").total.apply(lambda s: s.tail(6).sum())
       .sort_values(ascending=False))
zplaces = list(top.index[:4]) + [top.index[10], top.index[20]]
target = pd.concat([target, zt.assign(location_name="")], ignore_index=True)
fan_grid(zf, ["2026-10-24", "2026-11-01"], ["randomwalk", "randomwalknowcast"], zplaces,
         "fans-zones.png", "Health zones (4 largest, 11th and 21st by recent cases), 50% and 90%")

# score summary
sys.path.insert(0, str(Path(__file__).parent))
ORDER_H = ["k4+", "k3", "k2", "k1", "k0", "f1", "f2", "f3", "f4"]
cells = pd.read_csv(fig_dir / "tables" / "cells-origins-pooled.csv")
var = pd.read_csv(fig_dir / "tables" / "variants-by-h.csv")
zones = pd.read_csv(fig_dir / "tables" / "zones-by-h.csv")
levels = ["national", "provinces w/o near-zero", "provinces pooled"]
fig, axes = plt.subplots(3, 3, figsize=(13, 9), squeeze=False)
for j, tgt in enumerate(["next", "complete", "latest"]):
    for i, (metric, ylab) in enumerate([("wis", "mean WIS"), ("cov50", "50% coverage"), ("cov90", "90% coverage")]):
        ax = axes[i, j]
        for lvl, ls, mk in zip(levels, ["-", "--", ":"], ["o", "s", "^"]):
            c = cells[(cells.target == tgt) & (cells.level == lvl)]
            for m in ["as reported", "randomwalk", "randomwalknowcast"]:
                r = c[c.model == m].set_index("h").reindex(ORDER_H).dropna(subset=["n"])
                if r.empty:
                    continue
                x = [ORDER_H.index(h) for h in r.index]
                ax.plot(x, r[metric], ls=ls, marker=mk, color=COL[m], lw=1.5, ms=5,
                        label=f"{LAB[m]}, {lvl}")
        if metric == "wis":
            ax.set_yscale("log")
        else:
            ax.axhline(0.5 if metric == "cov50" else 0.9, color="#333", lw=0.8)
            ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(range(len(ORDER_H)), ORDER_H)
        ax.set_ylabel(ylab)
        if i == 0:
            ax.set_title(f"target: {tgt}", loc="left")
h, l = axes[0, 2].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
fig.suptitle("Scores by horizon (k: weeks back from cut-off week, f: forecast weeks), origins 10-17 and 10-24 pooled. "
             "Line style: level.", x=0.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0.08, 1, 0.97))
fig.savefig(fig_dir / "scores-summary.png", dpi=130)
plt.close(fig)

# relative WIS heat table: model x (level, target), per origin
s = pd.read_csv(fig_dir / "tables" / "cells-by-origin.csv")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"wspace": 0.35})
for ax, o in zip(axes, ["2026-10-17", "2026-10-24"]):
    t = s[(s.origin == o) & (s.model == "randomwalknowcast")]
    lv = ["national", "CD-IT", "CD-NK", "CD-HU", "CD-TO", "provinces w/o near-zero", "provinces pooled"]
    cols = [(tg, h) for tg in ["next", "complete", "latest"] for h in ORDER_H
            if ((t.target == tg) & (t.h == h)).any()]
    M = np.full((len(lv), len(cols)), np.nan)
    N = np.zeros_like(M)
    for a, l in enumerate(lv):
        for b, (tg, h) in enumerate(cols):
            r = t[(t.level == l) & (t.target == tg) & (t.h == h)]
            if len(r):
                M[a, b] = r.rel_wis.iloc[0]; N[a, b] = r.n.iloc[0]
    im = ax.imshow(np.log2(M), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    for a in range(len(lv)):
        for b in range(len(cols)):
            if not np.isnan(M[a, b]):
                ax.text(b, a, f"{M[a, b]:.2f}\nn={int(N[a, b])}", ha="center", va="center", fontsize=6.5,
                        color="#111" if abs(np.log2(M[a, b])) < 1.2 else "white")
    short = {"provinces w/o near-zero": "prov. w/o near-zero", "provinces pooled": "prov. pooled"}
    ax.set_yticks(range(len(lv)), [short.get(x, NAME.get(x, x)) for x in lv])
    ax.set_xticks(range(len(cols)), [f"{tg[0].upper()}\n{h}" for tg, h in cols], fontsize=7)
    ax.grid(False)
    ax.set_title(f"origin {o[5:]}", loc="left", fontsize=9)
fig.colorbar(im, ax=axes, label="log2 relative WIS", shrink=0.7)
fig.suptitle("randomwalknowcast relative WIS. N: next release (vs as reported); C: complete weeks, L: latest release (vs randomwalk). Below 1 is better.",
             x=0.1, ha="left", fontsize=9)
fig.savefig(fig_dir / "relwis-heat.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("figs ok")
