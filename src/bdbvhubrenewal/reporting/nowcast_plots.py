# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Figures of the random walk models' nowcasts against the releases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
from bdbvhubrenewal.reporting.scorechart import OVERALL_METRICS

__all__ = [
    "plot_relative_overall",
    "plot_release_nowcasts",
    "plot_truth_changes",
]


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
    x: str = "pool",
    metrics: Mapping[str, tuple[str, float | None]] = OVERALL_METRICS,
    colours: Mapping[str, tuple[str, str]] | None = None,
) -> Figure:
    """Plot each model's overall scores by pool, a panel per metric.

    Parameters
    ----------
    table : pandas.DataFrame
        `relative_wis` output by `x`, for one target.
    models : sequence of str
        Models in registry order. Those absent from a pool are skipped.
    baseline : str
        The model WIS is relative to, named in the labels.
    mode : str
        `"light"` or `"dark"`.
    x : str
        The column of the groups, `national`, `provinces` and `zones`.
    metrics : mapping of str to (str, float or None)
        Metric column to its label and reference line, as
        `OVERALL_METRICS`.
    colours : mapping of str to (str, str), optional
        Model to its light and dark colour. Defaults to the registry
        colours by position in `models`.

    Returns
    -------
    matplotlib.figure.Figure
        A panel of bars grouped by pool per metric, each with a dotted line
        at its reference.
    """
    ink = THEMES[mode]
    palette = MODEL_COLOURS[mode]
    pick = 0 if mode == "light" else 1
    have_pools = set(table[x])
    pools = [p for p in ("national", "provinces", "zones") if p in have_pools]
    fig, axes = _grid(len(metrics), 3, width=3.4, height=2.6)
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        rows = table.loc[table["model"] == m].set_index(x)
        have = [p for p in pools if p in rows.index]
        shift = (i - (len(models) - 1) / 2) * width
        at = [pools.index(p) + shift for p in have]
        colour = colours[m][pick] if colours else palette[i % len(palette)]
        for ax, key in zip(axes, metrics, strict=True):
            ax.bar(at, rows.loc[have, key], width, color=colour, label=m)
    for ax, (key, (label, ref)) in zip(axes, metrics.items(), strict=True):
        if ref is not None:
            ax.axhline(ref, color=ink["muted"], lw=1, ls=":")
        if key.startswith("rel_"):
            label = f"{label} to {baseline}"
        if key.startswith("coverage"):
            ax.set_ylim(0, 1)
        if key == "bias":
            ax.set_ylim(-1, 1)
        ax.set_ylabel(label)
        ax.set_axisbelow(True)
        ax.set_xticks(range(len(pools)), [p.capitalize() for p in pools])
        ax.set_xlim(-0.5, len(pools) - 0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=7,
        loc="outside lower center",
        ncols=min(len(models), 5),
    )
    return fig


def plot_truth_changes(
    target: pd.DataFrame,
    names: Mapping[str, str],
    weeks: int = 16,
    mode: str = "light",
) -> Figure:
    """Plot each place's weekly counts in every release and their changes.

    Parameters
    ----------
    target : pandas.DataFrame
        Real target rows with every release.
    names : mapping of str to str
        Place names by id.
    weeks : int
        Weeks shown, up to the latest release's last week.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        Per place with cases in the weeks shown, most first, the count by
        onset week with one line per release,
        from the oldest (faint) to the newest (strong), and under it the
        change from the previous release by week, with a line at 0.
    """
    ink = THEMES[mode]
    real = target.loc[target["kind"] == "real"]
    releases = sorted(set(real["vintage"]))
    shades = np.linspace(0.3, 1.0, len(releases))
    # Newer releases are stronger: darker on light, lighter on dark.
    cmap = plt.get_cmap("Blues")
    colours = [cmap(v if mode == "light" else 1.25 - v) for v in shades]
    end = pd.to_datetime(real["target_end_date"]).max()
    start = end - pd.Timedelta(weeks=weeks)
    wide = real.assign(
        target_end_date=pd.to_datetime(real["target_end_date"])
    ).pivot_table(
        index=["location", "target_end_date"],
        columns="vintage",
        values="total",
    )
    recent = wide.loc[wide.index.get_level_values(1) > start]
    latest = recent[releases[-1]].groupby(level=0).sum()
    latest = latest.loc[latest > 0].sort_values(ascending=False, kind="stable")
    locations = list(latest.index)
    ncol = min(4, len(locations))
    nrow = int(np.ceil(len(locations) / ncol))
    fig, axes = plt.subplots(
        2 * nrow,
        ncol,
        figsize=(3.4 * ncol, 3.6 * nrow),
        squeeze=False,
        layout="constrained",
        gridspec_kw={"height_ratios": [2, 1] * nrow},
    )
    for k in range(nrow * ncol):
        row, col = 2 * (k // ncol), k % ncol
        top, low = axes[row, col], axes[row + 1, col]
        if k >= len(locations):
            top.set_visible(False)
            low.set_visible(False)
            continue
        loc = locations[k]
        rows = wide.loc[loc]
        rows = rows.loc[rows.index > start]
        change = rows.diff(axis=1)
        for i, vin in enumerate(releases):
            top.plot(
                rows.index, rows[vin], color=colours[i], lw=1.4, label=str(vin)
            )
            if i:
                low.plot(rows.index, change[vin], color=colours[i], lw=1.2)
        low.axhline(0, color=ink["muted"], lw=1, ls=":")
        top.set_title(names.get(loc, loc), fontsize=9)
        top.set_xticklabels([])
        for ax in (top, low):
            ax.set_xlim(start, end + pd.Timedelta(days=3))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        if k % ncol == 0:
            top.set_ylabel("Count by onset week")
            low.set_ylabel("Change")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Release",
        fontsize=7,
        title_fontsize=8,
        loc="outside lower center",
        ncols=len(releases),
    )
    return fig
