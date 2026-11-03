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

__all__ = ["plot_release_nowcasts", "plot_weekly_crosscheck"]


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
    target: str = "complete",
) -> Figure:
    """Plot each model's forecast of the recent weeks by origin.

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
    target : str
        `"complete"` for the final count, or `"reported"` for the count
        expected at the latest release.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per origin and place. Lines are the counts each release
        reports, bars the 50% and 90% intervals.
    """
    final = quantiles.loc[quantiles["target"] == target]
    final = final.loc[
        final["model"].isin(models)
        & (final["truth_vintage"] == final["truth_vintage"].max())
    ]
    if target == "reported":
        # Weeks after the release's cut-off have nothing reported.
        final = final.loc[final["truth"].notna()]
    origins = sorted(final["origin"].unique())
    fig, axes = _grid(
        len(origins) * len(locations), len(locations), width=3.4, height=2.6
    )
    ink = THEMES[mode]
    palette = MODEL_COLOURS[mode]
    shades = np.linspace(0.4, 1.0, len(releases["vintage"].unique()))
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
                hi = max(
                    pd.Timestamp(weeks.max()),
                    pd.Timestamp(origin) + pd.Timedelta(days=7),
                )
                keep = (x >= lo) & (x <= hi)
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
                shift = (m_i - (len(models) - 1) / 2) * 3.2 / len(models)
                _draw_bars(ax, rows, palette[m_i], shift)
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


def plot_weekly_crosscheck(
    quantiles: pd.DataFrame,
    weekly: pd.DataFrame,
    releases: pd.DataFrame,
    locations: Sequence[str],
    model: str = "randomwalknowcast",
    mode: str = "light",
) -> Figure:
    """Plot the weekly cross-check points over a model's nowcast bands.

    Parameters
    ----------
    quantiles : pandas.DataFrame
        Quantiles of `model` at the latest release, columns `location`,
        `target_end_date`, `q5` to `q95`.
    weekly : pandas.DataFrame
        Columns `location`, `target_end_date` and `point`.
    releases : pandas.DataFrame
        Real target rows with `vintage`, `location`, `target_end_date`
        and `total`, for the reported counts.
    locations : sequence of str
        Places to show.
    model : str
        Model name for the legend.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per place.
    """
    fig, axes = _grid(len(locations), 3, width=3.4, height=2.6)
    ink = THEMES[mode]
    palette = MODEL_COLOURS[mode]
    latest = releases["vintage"].max()
    for ax, loc in zip(axes, locations, strict=True):
        q = quantiles.loc[quantiles["location"] == loc]
        _draw_bars(ax, q, palette[0], 0.0)
        w = weekly.loc[weekly["location"] == loc]
        ax.plot(
            mdates.date2num(pd.to_datetime(w["target_end_date"])) + 0.0,
            w["point"],
            "x",
            color=palette[1],
            ms=6,
            mew=1.8,
        )
        r = releases.loc[
            (releases["vintage"] == latest) & (releases["location"] == loc)
        ]
        r = r.loc[r["target_end_date"].isin(q["target_end_date"])]
        ax.plot(
            pd.to_datetime(r["target_end_date"]),
            r["total"],
            color=ink["data"],
            lw=1.2,
        )
        ax.set_title(loc)
        ax.xaxis.set_major_locator(
            mdates.WeekdayLocator(byweekday=5, interval=1)
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[0].set_ylabel("Cases by onset week")
    axes[0].plot([], [], color=palette[0], lw=4, label=f"{model} (50% and 90%)")
    axes[0].plot([], [], "x", color=palette[1], label="Weekly cross-check")
    axes[0].plot([], [], color=ink["data"], label=f"Reported {latest}")
    axes[0].legend(fontsize=6)
    return fig
