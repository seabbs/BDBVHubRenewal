"""One figure for the width and conditioning check.

    python scripts/ftpractice/width_figure.py GERMAN_WIDTHS.csv COND_DIR \
        QUANTILES_DIR OUT.png

(A) German development half: median 80% interval width over the final
count by horizon, daily ragged model. (B) Held-out sims: median nowcast
90% width over the final count by sqrt_phi quintile. (C) Like for like,
the 7-day count at the cut-off, from QUANTILES_DIR/../like-for-like.csv:
80% width on held-out sims (weekly blocks at the last step, and the
weekly total rows of daily+weekly blocks at the last period end) against
Germany (horizon 0), with Germany's 80%
coverage.
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
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


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


def end_labels(ax, ends: dict[str, tuple[float, float]], gap: float) -> None:
    """Direct-label line ends, nudged apart by at least `gap`."""
    placed: list[float] = []
    for name, (x, y) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        ty = max(y, placed[-1] + gap) if placed else y
        placed.append(ty)
        ax.plot(x, y, "o", ms=6, color=COLOURS[name], mec="white", mew=1.5)
        ax.annotate(
            name,
            (x, ty),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )


def main() -> None:
    """Draw the three panels."""
    german, cond = Path(sys.argv[1]), Path(sys.argv[2])
    quantiles, out = Path(sys.argv[3]), Path(sys.argv[4])
    g = pd.read_csv(german)
    per = pd.read_csv(cond / "per-sim.csv")
    per = per[per["part"] == "nowcast"].copy()
    names = [n for n in COLOURS if (quantiles / f"{n}.csv.gz").exists()]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    ax = axes[0]
    daily = g[(g["model"] == "fm_chronos_daily_ragged") & (g["horizon"] > -28)]
    ends = {}
    for (run, n), name in GERMAN.items():
        s = daily[(daily["run"] == run) & (daily["n"] == n)]
        s = s.sort_values("horizon")
        if s.empty:
            continue
        x, y = s["horizon"].to_numpy(), s["rel_width80_median"].to_numpy()
        ax.plot(x, y, color=COLOURS[name], lw=2)
        ends[name] = (x[-1], y[-1])
    end_labels(ax, ends, 0.06)
    style(
        ax,
        "A  Germany dev half: 80% width (daily model)",
        "horizon (days)",
        "80% width / final count",
    )
    ax.set_xticks([-21, -14, -7, 0])
    ax.set_xlim(-22, 5)

    ax = axes[1]
    per["phi_bin"] = pd.qcut(per["sqrt_phi"], 5, labels=False)
    mids = per.groupby("phi_bin")["sqrt_phi"].median()
    ends = {}
    for name in names:
        s = per[per["checkpoint"] == name]
        y = s.groupby("phi_bin")["width90"].median().reindex(mids.index)
        ax.plot(mids.to_numpy(), y.to_numpy(), color=COLOURS[name], lw=2)
        ends[name] = (mids.to_numpy()[-1], y.to_numpy()[-1])
    end_labels(ax, ends, 0.05)
    style(
        ax,
        "B  Held-out sims: nowcast 90% width vs sqrt_phi",
        "sqrt_phi (quintile medians)",
        "90% width / final count",
    )
    ax.set_xlim(right=mids.max() * 1.25)
    ax.set_ylim(bottom=0)

    ax = axes[2]
    table = pd.read_csv(quantiles.parent / "like-for-like.csv")
    table = table[table["checkpoint"].isin(names)].reset_index(drop=True)
    for i, row in table.iterrows():
        colour = COLOURS[row["checkpoint"]]
        for off, kind in ((-0.17, "weekly"), (0.17, "dw")):
            de = "de_weekly" if kind == "weekly" else "de_daily"
            sw = row[f"sims_{kind}_w80"]
            gw, gc = row[f"{de}_w80"], row[f"{de}_cov80"]
            x = i + off
            ax.plot([x, x], [sw, gw], color=GRID, lw=2, zorder=1)
            marker = "o" if kind == "weekly" else "s"  # dw: daily+weekly
            ax.plot(x, sw, marker, ms=8, mfc=SURFACE, mec=colour, mew=2)
            ax.plot(x, gw, marker, ms=8, color=colour, mec="white", mew=1.5)
            ax.annotate(
                f"{gc:.0%}",
                (x, gw),
                xytext=(0, -12 if gw < sw else 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=MUTED,
            )
    for marker, text in (
        ("o", "weekly model"),
        ("s", "daily model, weekly total rows"),
    ):
        ax.plot([], [], marker, color=MUTED, label=text)
    ax.plot(
        [], [], "o", mfc=SURFACE, mec=MUTED, mew=2, label="open: held-out sims"
    )
    ax.plot(
        [], [], "o", color=MUTED, label="filled: Germany (label = 80% cover)"
    )
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    ax.set_xticks(np.arange(len(table)), table["checkpoint"])
    ax.set_ylim(0, 2.0)
    style(
        ax,
        "C  7-day count at the cut-off: sims vs Germany",
        "",
        "80% width / final count",
    )
    fig.savefig(out, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    main()
