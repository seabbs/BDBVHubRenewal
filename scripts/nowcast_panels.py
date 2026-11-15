"""Small-multiple nowcast panels: one column per model, one row per panel.

Every panel shows the same truth (black) and the data as of the origin
(grey, dashed), and one model's median with its 50% and 95% intervals,
so no two models' lines overlap. Each column is titled with the model
and its relative WIS against randomwalknowcast. The truth and data lines
are labelled directly in the first column.

`draw_panels` takes the forecasts as quantiles, so the paper's nowcast
figures can call it with any study's models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# The dataviz reference palette, light mode (validated: one series).
SERIES = "#2a78d6"
TRUTH = "#0b0b0b"
DATA = "#8a8984"
TEXT = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"


def _quantiles(fc: pd.DataFrame, origin: str, place: str) -> pd.DataFrame:
    rows = fc.loc[(fc["origin"] == origin) & (fc["location"] == place)]
    q = rows.pivot_table(
        index="target_end_date", columns="quantile", values="value"
    )
    q.index = pd.to_datetime(q.index)
    return q.sort_index()


def draw_panels(
    models: Mapping[str, pd.DataFrame],
    rel_wis: Mapping[str, float],
    truth: pd.DataFrame,
    asof: pd.DataFrame,
    rows: Sequence[tuple[str, str, str]],
    title: str,
    out: Path,
    truth_label: str = "truth",
    truth_short: str = "truth",
    scored_on: str = "all development dates",
) -> None:
    """Draw one column per model and one row per (origin, place) into `out`.

    Parameters
    ----------
    models : mapping
        Model label to quantile forecasts: `origin`, `location`,
        `target_end_date`, `quantile` (with 0.025, 0.25, 0.5, 0.75 and
        0.975) and `value`.
    rel_wis : mapping
        Model label to its relative WIS against randomwalknowcast.
    truth : pandas.DataFrame
        `location`, `target_end_date`, `truth`: the scored truth.
    asof : pandas.DataFrame
        `origin`, `location`, `target_end_date`, `total`: the data as each
        origin saw them.
    rows : sequence of (str, str, str)
        Each row's origin, location and label.
    title, out
        Figure title and file.
    truth_label : str
        What the black line is, for the caption.
    truth_short : str
        The black line's label in the first column.
    scored_on : str
        The cells the relative WIS is on, for the caption.
    """
    labels = list(models)
    fig, axes = plt.subplots(
        len(rows),
        len(labels),
        figsize=(2.6 * len(labels), 2.3 * len(rows) + 0.8),
        sharey="row",
        squeeze=False,
        facecolor=SURFACE,
    )
    for i, (origin, place, row_label) in enumerate(rows):
        seen = asof.loc[
            (asof["origin"] == origin) & (asof["location"] == place)
        ].sort_values("target_end_date")
        x_seen = pd.to_datetime(seen["target_end_date"])
        tr = truth.loc[
            (truth["location"] == place)
            & (truth["target_end_date"] >= seen["target_end_date"].min())
        ].sort_values("target_end_date")
        x_tr = pd.to_datetime(tr["target_end_date"])
        for j, label in enumerate(labels):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            ax.plot(x_tr, tr["truth"], color=TRUTH, lw=1.5, zorder=3)
            ax.plot(
                x_seen, seen["total"], color=DATA, lw=1.2, ls="--", zorder=2
            )
            q = _quantiles(models[label], origin, place)
            if not q.empty:
                ax.fill_between(
                    q.index,
                    q[0.025],
                    q[0.975],
                    color=SERIES,
                    alpha=0.15,
                    lw=0,
                    zorder=1,
                )
                ax.fill_between(
                    q.index,
                    q[0.25],
                    q[0.75],
                    color=SERIES,
                    alpha=0.3,
                    lw=0,
                    zorder=1,
                )
                ax.plot(
                    q.index,
                    q[0.5],
                    color=SERIES,
                    lw=2,
                    marker="o",
                    ms=4,
                    zorder=4,
                )
            ax.grid(axis="y", color=GRID, lw=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=TEXT, labelsize=7)
            ax.margins(x=0.12)
            ax.xaxis.set_major_locator(
                mdates.AutoDateLocator(minticks=3, maxticks=4)
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            if i == 0:
                score = rel_wis.get(label)
                sub = "" if score is None else f"\nrel. WIS {score:.2f}"
                ax.set_title(label + sub, fontsize=9, color=TRUTH)
            if j == 0:
                ax.set_ylabel(row_label, fontsize=8, color=TEXT)
                if len(x_tr):
                    ax.annotate(
                        truth_short,
                        (x_tr.iloc[-1], tr["truth"].iloc[-1]),
                        xytext=(4, 0),
                        va="center",
                        textcoords="offset points",
                        fontsize=7,
                        color=TRUTH,
                    )
                if len(x_seen):
                    ax.annotate(
                        "data",
                        (x_seen.iloc[-1], seen["total"].iloc[-1]),
                        xytext=(4, 0),
                        va="center",
                        textcoords="offset points",
                        fontsize=7,
                        color=TEXT,
                    )
    fig.suptitle(title, fontsize=11, color=TRUTH, x=0.01, ha="left")
    fig.text(
        0.01,
        0.005,
        f"Black: {truth_label}. Grey dashed: data as of the origin. "
        "Blue: median, 50% and 95% intervals. Relative WIS against "
        "randomwalknowcast, "
        f"{scored_on}.",
        fontsize=8,
        color=TEXT,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
