# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Figures of the random walk models' nowcasts against the releases."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bdbvhubrenewal.reporting.plots import (
    MODEL_COLOURS,
    THEMES,
    _date_axis,
    _grid,
)

__all__ = ["plot_release_nowcasts"]


def _draw_bars(ax: Axes, rows: pd.DataFrame, colour: str, shift: float) -> None:
    x = mdates.date2num(pd.to_datetime(rows["target_end_date"])) + shift
    ax.vlines(x, rows["q5"], rows["q95"], color=colour, lw=1.2)
    ax.vlines(x, rows["q25"], rows["q75"], color=colour, lw=4)
    ax.plot(x, rows["q50"], "o", color=colour, ms=3)


def plot_release_nowcasts(
    quantiles: pd.DataFrame,
    releases: pd.DataFrame,
    locations: Sequence[str],
    models: Sequence[str],
    mode: str = "light",
) -> Figure:
    """Plot each model's final forecast of the recent weeks by origin.

    Parameters
    ----------
    quantiles : pandas.DataFrame
        `evaluation.nowcast.ReleaseBacktest.quantiles`.
    releases : pandas.DataFrame
        Real target rows with `vintage`, `location`, `target_end_date`
        and `total`.
    locations : sequence of str
        Places to show, one column each.
    models : sequence of str
        Models to show.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per origin and place. Lines are the counts each release
        reports, bars the 50% and 90% intervals.
    """
    final = quantiles.loc[quantiles["target"] == "complete"]
    final = final.loc[final["model"].isin(models)]
    origins = sorted(final["origin"].unique())
    fig, axes = _grid(
        len(origins) * len(locations), len(locations), width=3.4, height=2.6
    )
    ink = THEMES[mode]
    palette = MODEL_COLOURS[mode]
    shades = np.linspace(0.4, 1.0, len(releases["vintage"].unique()))
    # Each model's bars sit side by side within the week.
    width, mid = 5.6 / len(models), (len(models) - 1) / 2
    for i, origin in enumerate(origins):
        for j, loc in enumerate(locations):
            ax = axes[i * len(locations) + j]
            weeks = final.loc[
                (final["origin"] == origin) & (final["location"] == loc)
            ]["target_end_date"]
            lo = pd.Timestamp(weeks.min()) - pd.Timedelta(days=7)
            for v, a in zip(
                sorted(releases["vintage"].unique()), shades, strict=True
            ):
                r = releases.loc[
                    (releases["vintage"] == v) & (releases["location"] == loc)
                ]
                x = pd.to_datetime(r["target_end_date"])
                keep = (x >= lo) & (x <= pd.Timestamp(origin) + pd.Timedelta(7))
                ax.plot(
                    x[keep],
                    r["total"][keep],
                    color=ink["data"],
                    alpha=a,
                    lw=1.2,
                    label=f"Release {v}",
                )
            for m_i, m in enumerate(models):
                rows = final.loc[
                    (final["origin"] == origin)
                    & (final["location"] == loc)
                    & (final["model"] == m)
                ]
                _draw_bars(ax, rows, palette[m_i], (m_i - mid) * width)
            ax.set_title(f"{loc}, origin {origin}")
            _date_axis(ax)
            ax.xaxis.set_major_locator(
                mdates.WeekdayLocator(byweekday=5, interval=2)
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[0].set_ylabel("Cases by onset week")
    handles, labels = axes[0].get_legend_handles_labels()
    for m_i, m in enumerate(models):
        handles.append(axes[0].plot([], [], color=palette[m_i], lw=4)[0])
        labels.append(f"{m} (50% and 90%)")
    fig.legend(
        handles,
        labels,
        fontsize=7,
        loc="outside lower center",
        ncols=min(len(labels), 5),
    )
    return fig


def plot_relative_overall(
    table: pd.DataFrame,
    models: Sequence[str],
    baseline: str = "randomwalk",
    mode: str = "light",
) -> Figure:
    """Plot each model's relative WIS and 90% coverage by pool.

    Parameters
    ----------
    table : pandas.DataFrame
        `relative_wis` output by `pool`, for one target.
    models : sequence of str
        Models in registry order. Those absent from a pool are skipped.
    baseline : str
        The model WIS is relative to.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        Two panels of bars grouped by pool. The first has relative WIS,
        with a line at 1. The second has 90% interval coverage, with a
        line at 0.9.
    """
    ink = THEMES[mode]
    palette = MODEL_COLOURS[mode]
    have_pools = set(table["pool"])
    pools = [p for p in ("national", "provinces", "zones") if p in have_pools]
    fig, axes = _grid(2, 2, width=3.8, height=2.8)
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        rows = table.loc[table["model"] == m].set_index("pool")
        have = [p for p in pools if p in rows.index]
        shift = (i - (len(models) - 1) / 2) * width
        x = [pools.index(p) + shift for p in have]
        colour = palette[i % len(palette)]
        axes[0].bar(x, rows.loc[have, "rel_wis"], width, color=colour, label=m)
        axes[1].bar(x, rows.loc[have, "coverage_90"], width, color=colour)
    axes[0].axhline(1, color=ink["muted"], lw=1, ls=":")
    axes[1].axhline(0.9, color=ink["muted"], lw=1, ls=":")
    axes[0].set_ylabel(f"WIS relative to {baseline}")
    axes[1].set_ylabel("90% interval coverage")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.set_xticks(range(len(pools)), [p.capitalize() for p in pools])
    fig.legend(
        fontsize=7, loc="outside lower center", ncols=min(len(models), 5)
    )
    return fig
