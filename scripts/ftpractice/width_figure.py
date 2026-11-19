"""One figure for the width and conditioning check.

    python scripts/ftpractice/width_figure.py GERMAN_WIDTHS.csv COND_DIR OUT.png

(A) German development half: median 95% interval width over the final
count by horizon, daily ragged model. (B) Held-out sims: median nowcast
90% width over the final count by sqrt_phi quintile, with the sims' own
realised spread. (C) The same by delay_change level, with Germany's
realised spread.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOURS = {
    "zero-shot": "#2a78d6",
    "r5-n4000": "#eb6834",
    "r6-n250": "#1baf7a",
    "r6-n500": "#eda100",
    "r6-n750": "#e87ba4",
}
GERMAN = {
    ("zero-shot", 0): "zero-shot",
    ("r5", 4000): "r5-n4000",
    ("r6", 250): "r6-n250",
    ("r6", 500): "r6-n500",
    ("r6", 750): "r6-n750",
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply recessive axes and grid."""
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def label_end(ax, x, y, text: str, colour: str) -> None:
    """Direct-label a line at its last point."""
    ax.plot(x[-1], y[-1], "o", ms=6, color=colour, mec="white", mew=1.5)
    ax.annotate(
        text,
        (x[-1], y[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=INK,
    )


def main() -> None:
    """Draw the three panels."""
    german, cond, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    g = pd.read_csv(german)
    g = g[(g["model"] == "fm_chronos_daily_ragged") & (g["horizon"] > -28)]
    per = pd.read_csv(cond / "per-sim.csv")
    per = per[per["part"] == "nowcast"]
    ratios = pd.read_csv(cond / "sim-ratios.csv")
    summary = (cond / "summary.md").read_text().splitlines()
    german_spread = float(summary[1].split("spread ")[1].split(",")[0])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    ax = axes[0]
    for (run, n), name in GERMAN.items():
        s = g[(g["run"] == run) & (g["n"] == n)].sort_values("horizon")
        if s.empty:
            continue
        x, y = s["horizon"].to_numpy(), s["rel_width95_median"].to_numpy()
        ax.plot(x, y, color=COLOURS[name], lw=2)
        label_end(ax, x, y, name, COLOURS[name])
    style(
        ax,
        "A  Germany, dev half (daily model)",
        "horizon (days)",
        "95% width / final count",
    )
    ax.set_xticks([-21, -14, -7, 0])
    ax.set_xlim(-22, 4)

    ax = axes[1]
    per["phi_bin"] = pd.qcut(per["sqrt_phi"], 5, labels=False)
    mids = per.groupby("phi_bin")["sqrt_phi"].median()
    for name, colour in COLOURS.items():
        s = per[per["checkpoint"] == name]
        if s.empty:
            continue
        y = s.groupby("phi_bin")["width90"].median().reindex(mids.index)
        ax.plot(mids.to_numpy(), y.to_numpy(), color=colour, lw=2)
        label_end(ax, mids.to_numpy(), y.to_numpy(), name, colour)
    sims = per.drop_duplicates("simulation")
    own = sims.groupby("phi_bin")["own_spread"].median().reindex(mids.index)
    ax.plot(mids.to_numpy(), own.to_numpy(), color=MUTED, lw=2, ls="--")
    label_end(ax, mids.to_numpy(), own.to_numpy(), "sims' own spread", MUTED)
    style(
        ax,
        "B  Held-out sims: nowcast width vs sqrt_phi",
        "sqrt_phi (quintile medians)",
        "90% width / final count",
    )

    ax = axes[2]
    levels = sorted(per["delay_change"].unique())
    x = np.arange(len(levels))
    names = [n for n in COLOURS if (per["checkpoint"] == n).any()]
    for j, name in enumerate(names):
        s = per[per["checkpoint"] == name]
        y = s.groupby("delay_change")["width90"].median().reindex(levels)
        off = (j - (len(names) - 1) / 2) * 0.12
        ax.plot(
            x + off,
            y.to_numpy(),
            "o",
            ms=7,
            color=COLOURS[name],
            mec="white",
            mew=1.5,
            label=name,
        )
    sims_by = sims.set_index("simulation")["delay_change"]
    ratios["delay_change"] = ratios["simulation"].map(sims_by)
    real = (
        ratios.groupby("delay_change")["ratio"]
        .apply(
            lambda r: (
                (np.quantile(r, 0.95) - np.quantile(r, 0.05)) / np.median(r)
            )
        )
        .reindex(levels)
    )
    ax.bar(
        x,
        real.to_numpy(),
        width=0.7,
        color=GRID,
        zorder=0,
        label="sims' realised spread",
    )
    ax.axhline(german_spread, color=INK, lw=1, ls=":")
    ax.annotate(
        "Germany realised spread",
        (len(levels) - 0.5, german_spread),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK,
    )
    ax.set_xticks(x, levels, rotation=30)
    style(ax, "C  Held-out sims by delay_change", "", "90% width / final count")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.savefig(out, dpi=160, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
