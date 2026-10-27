# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Figures for the data, the forecasts and the scores.

Every function returns a matplotlib `Figure` and takes `theme`, `"light"`
or `"dark"`, so the docs can show both.
Models get colours from `MODEL_COLOURS` in registry order, so a model
keeps its colour when others are added.
The vintage kind is always shown, in a panel title or the legend.

The submission phase can call `plot_nowcast(location=..., vintage=...)`
for the figure it embeds in the hub pull request.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, cast

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .data import imputed_share, newly_missing, revision_table
from .delays import delay_summary

__all__ = [
    "MODEL_COLOURS",
    "THEMES",
    "model_colour",
    "plot_comparison",
    "plot_delay_distribution",
    "plot_delays_over_time",
    "plot_forecasts",
    "plot_imputed_share",
    "plot_newly_missing",
    "plot_nowcast",
    "plot_reconstruction",
    "plot_relative_skill",
    "plot_revisions",
    "plot_scores",
    "plot_simulation",
    "plot_target",
    "save_light_dark",
    "theme",
]

THEMES: Final[dict[str, dict[str, str]]] = {
    "light": {
        "ink": "#0b0b0b",
        "text": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "data": "#0b0b0b",
    },
    "dark": {
        "ink": "#f0efec",
        "text": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "data": "#f0efec",
    },
}
"""Ink colours for each theme. Backgrounds are transparent."""

MODEL_COLOURS: Final[dict[str, tuple[str, ...]]] = {
    "light": (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ),
    "dark": (
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ),
}
"""Categorical colours in fixed order (checked for colour vision deficiency)."""

_KIND_MARKERS: Final = {"real": "o", "reconstructed": "s", "simulated": "^"}


def model_colour(name: str, models: Sequence[str], mode: str = "light") -> str:
    """Return a model's colour from its position in the model list.

    Parameters
    ----------
    name : str
        The model.
    models : sequence of str
        All models, in registry order.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    str
        A hex colour.
    """
    palette = MODEL_COLOURS[mode]
    return palette[list(models).index(name) % len(palette)]


@contextmanager
def theme(mode: str = "light") -> Iterator[dict[str, str]]:
    """Apply a light or dark theme to figures made inside the block.

    Parameters
    ----------
    mode : str
        `"light"` or `"dark"`.

    Yields
    ------
    dict
        The theme's ink colours.
    """
    ink = THEMES[mode]
    params: dict[str, Any] = {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "axes.edgecolor": ink["axis"],
        "axes.labelcolor": ink["text"],
        "axes.titlecolor": ink["ink"],
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": ink["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": ink["muted"],
        "ytick.color": ink["muted"],
        "xtick.labelcolor": ink["text"],
        "ytick.labelcolor": ink["text"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": ink["text"],
        "text.color": ink["text"],
        "font.size": 9,
        "lines.linewidth": 2,
        "svg.fonttype": "none",
    }
    with mpl.rc_context(params):  # type: ignore[arg-type]
        yield ink


def save_light_dark(
    make: Callable[..., Figure], path: Path | str
) -> list[Path]:
    """Save a figure in light and dark versions.

    Parameters
    ----------
    make : callable
        Takes the keyword `mode` (`"light"` or `"dark"`) and returns a
        figure.
    path : pathlib.Path or str
        Output path for the light version, such as `figures/x.svg`. The
        dark version gets a `-dark` suffix.

    Returns
    -------
    list of pathlib.Path
        The two paths written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = [out, out.with_name(f"{out.stem}-dark{out.suffix}")]
    for mode, p in zip(("light", "dark"), paths, strict=True):
        with theme(mode):
            fig = make(mode=mode)
            fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
    return paths


def _date_axis(ax: Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def _grid(
    n: int, ncol: int, width: float = 3.2, height: float = 2.4
) -> tuple[Figure, list[Axes]]:
    ncol = max(1, min(ncol, n))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(width * ncol, height * nrow),
        squeeze=False,
        layout="constrained",
    )
    flat = list(axes.ravel())
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def _names(target: pd.DataFrame) -> dict[str, str]:
    pairs = target[["location", "location_name"]].drop_duplicates()
    return dict(zip(pairs["location"], pairs["location_name"], strict=False))


def _locations_by_size(target: pd.DataFrame, n: int | None = None) -> list[str]:
    latest = target.loc[target["vintage"] == target["vintage"].max()]
    totals = (
        latest.groupby("location")["total"].sum().sort_values(ascending=False)
    )
    locs = list(totals.index[totals > 0])
    return locs if n is None else locs[:n]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def plot_target(
    target: pd.DataFrame,
    locations: Sequence[str] | None = None,
    mode: str = "light",  # noqa: ARG001 - shared signature
) -> Figure:
    """Plot weekly totals by onset week for each vintage, by location.

    Parameters
    ----------
    target : pandas.DataFrame
        Target rows of one kind.
    locations : sequence of str, optional
        Locations to show. Defaults to all with cases, largest first.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per location.
    """
    locs = list(locations or _locations_by_size(target))
    vints = sorted(target["vintage"].unique())
    names = _names(target)
    fig, axes = _grid(len(locs), 3)
    cmap = mpl.colormaps["Blues"]
    shades = np.linspace(0.35, 0.95, len(vints))
    for ax, loc in zip(axes, locs, strict=True):
        rows = target.loc[target["location"] == loc]
        for vin, shade in zip(vints, shades, strict=True):
            r = rows.loc[rows["vintage"] == vin]
            ax.plot(
                pd.to_datetime(r["target_end_date"]),
                r["total"],
                color=cmap(shade),
                lw=1.5,
                label=vin,
            )
        ax.set_title(f"{names.get(loc, loc)} ({loc})")
        _date_axis(ax)
    axes[0].set_ylabel("Confirmed cases by onset week")
    axes[0].legend(title="Vintage", title_fontsize=8)
    return fig


def plot_revisions(target: pd.DataFrame, mode: str = "light") -> Figure:
    """Plot how recent weeks fill in, as a share of the latest vintage.

    Parameters
    ----------
    target : pandas.DataFrame
        Real target rows.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        National share reported by weeks before the vintage.
    """
    rev = revision_table(target.loc[target["location"] == "CD"])
    latest = rev["vintage"].max()
    rev = rev.loc[(rev["vintage"] < latest) & (rev["weeks_before"] <= 10)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
    palette = MODEL_COLOURS[mode]
    for i, (vin, r) in enumerate(rev.groupby("vintage")):
        ax.plot(
            r["weeks_before"],
            r["fraction"],
            marker="o",
            ms=4,
            color=palette[i % len(palette)],
            label=str(vin),
        )
    ax.axhline(1, color=THEMES[mode]["muted"], lw=1, ls=":")
    ax.set_xlabel("Weeks from onset week to the vintage's epi week")
    ax.set_ylabel(f"Share of the {latest} count")
    ax.invert_xaxis()
    ax.legend(title="Vintage", title_fontsize=8)
    return fig


def plot_imputed_share(
    target: pd.DataFrame,
    locations: Sequence[str] | None = None,
    mode: str = "light",
) -> Figure:
    """Plot the share of cases with an imputed onset by onset week.

    Parameters
    ----------
    target : pandas.DataFrame
        Real target rows.
    locations : sequence of str, optional
        Locations to show. Defaults to the four largest and the country.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per location, one line per vintage, with point size by
        the week's total.
    """
    share = imputed_share(target)
    locs = list(locations or _locations_by_size(target, 5))
    vints = sorted(share["vintage"].unique())
    names = _names(target)
    palette = MODEL_COLOURS[mode]
    fig, axes = _grid(len(locs), 3)
    for ax, loc in zip(axes, locs, strict=True):
        rows = share.loc[(share["location"] == loc) & (share["total"] > 0)]
        for i, vin in enumerate(vints):
            r = rows.loc[rows["vintage"] == vin]
            x = pd.to_datetime(r["target_end_date"])
            col = palette[i % len(palette)]
            ax.plot(x, r["imputed_share"], color=col, lw=1, label=vin)
            ax.scatter(
                x, r["imputed_share"], s=np.sqrt(r["total"]) * 2, color=col
            )
        ax.set_ylim(0, 1)
        ax.set_title(f"{names.get(loc, loc)} ({loc})")
        _date_axis(ax)
    axes[0].set_ylabel("Imputed share of cases")
    axes[0].legend(title="Vintage", title_fontsize=8)
    return fig


def plot_newly_missing(target: pd.DataFrame, mode: str = "light") -> Figure:
    """Plot cases added between vintages, split by onset status.

    Parameters
    ----------
    target : pandas.DataFrame
        Real target rows.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        Bars of new observed and new imputed cases per vintage step, by
        province.
    """
    new = newly_missing(target)
    new = new.loc[new["location"] != "CD"]
    keep = new.groupby("location")["new_total"].sum().abs().sort_values()
    locs = list(keep.index[keep > 0][::-1])
    new = new.loc[new["location"].isin(locs)]
    steps = sorted(new["vintage"].unique())
    fig, axes = _grid(len(steps), 3, width=3.6, height=2.8)
    palette = MODEL_COLOURS[mode]
    for ax, vin in zip(axes, steps, strict=True):
        r = new.loc[new["vintage"] == vin].set_index("location").reindex(locs)
        x = np.arange(len(locs))
        ax.bar(
            x - 0.2,
            r["new_observed"],
            0.38,
            color=palette[0],
            label="Known onset",
        )
        ax.bar(
            x + 0.2,
            r["new_imputed"],
            0.38,
            color=palette[1],
            label="Imputed onset",
        )
        ax.axhline(0, color=THEMES[mode]["axis"], lw=1)
        ax.set_xticks(x, [loc.removeprefix("CD-") for loc in locs])
        prev = r["previous_vintage"].dropna()
        ax.set_title(f"{prev.iloc[0] if len(prev) else ''} to {vin}")
    axes[0].set_ylabel("Cases added")
    axes[0].legend()
    return fig


def plot_delay_distribution(
    delays: pd.DataFrame, max_delay: int = 40, mode: str = "light"
) -> Figure:
    """Plot the onset-to-report delay distribution by province.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay line data of one vintage.
    max_delay : int
        Longest delay shown.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per province with at least 10 cases, with the count,
        mean and median in the title.
    """
    counts = delays["location"].value_counts()
    locs = list(counts.index[counts >= 10])
    names = _names(delays)
    fig, axes = _grid(len(locs), 3)
    colour = MODEL_COLOURS[mode][0]
    bins = list(np.arange(-0.5, max_delay + 1.5))
    for ax, loc in zip(axes, locs, strict=True):
        d = delays.loc[delays["location"] == loc, "delay"]
        ax.hist(d.clip(upper=max_delay), bins=bins, density=True, color=colour)
        ax.set_title(
            f"{names.get(loc, loc)}: n {len(d)}, mean {d.mean():.1f}, "
            f"median {d.median():.0f}"
        )
        ax.set_xlim(-1, max_delay + 1)
    axes[0].set_ylabel("Share of cases")
    for ax in axes:
        ax.set_xlabel("Days from onset to report")
    return fig


def plot_delays_over_time(
    delays: pd.DataFrame, min_cases: int = 5, mode: str = "light"
) -> Figure:
    """Plot the mean and median delay by onset week, with counts.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay line data of one or more vintages.
    min_cases : int
        Weeks with fewer cases are not drawn.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        Top: mean (line) and median (points) delay by onset week for the
        largest provinces, latest vintage. Bottom: cases per week.
    """
    latest = delays.loc[delays["vintage"] == delays["vintage"].max()]
    summ = delay_summary(latest, by=("location", "onset_week"))
    summ = summ.loc[summ["n"] >= min_cases]
    locs = list(latest["location"].value_counts().index[:4])
    names = _names(latest)
    palette = MODEL_COLOURS[mode]
    fig, (ax, ax2) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5),
        sharex=True,
        height_ratios=(2, 1),
        layout="constrained",
    )
    for i, loc in enumerate(locs):
        s = summ.loc[summ["location"] == loc]
        x = pd.to_datetime(s["onset_week"])
        col = palette[i]
        ax.plot(x, s["mean"], color=col, label=names.get(loc, loc))
        ax.scatter(x, s["median"], color=col, s=10)
        ax2.plot(x, s["n"], color=col)
    ax.set_ylabel("Delay (days): mean line, median points")
    ax.legend()
    ax2.set_ylabel("Cases")
    ax2.set_yscale("log")
    _date_axis(ax2)
    ax.set_title(f"Delay data of the {latest['vintage'].max()} vintage")
    return fig


def plot_reconstruction(
    comparison: pd.DataFrame, location: str = "CD", mode: str = "light"
) -> Figure:
    """Plot reconstructed against real vintages for one location.

    Parameters
    ----------
    comparison : pandas.DataFrame
        Output of `bdbv_forecasts.vintages.compare_reconstruction`.
    location : str
        Location to show.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per vintage: real and reconstructed totals by week.
    """
    comp = comparison.loc[comparison["location"] == location]
    vints = sorted(comp["vintage"].unique())
    fig, axes = _grid(len(vints), 3, width=3.4)
    palette = MODEL_COLOURS[mode]
    for ax, vin in zip(axes, vints, strict=True):
        c = comp.loc[comp["vintage"] == vin]
        c = c.loc[c["target_end_date"] >= "2026-05-01"]
        x = pd.to_datetime(c["target_end_date"])
        ax.plot(x, c["total_real"], color=THEMES[mode]["data"], label="Real")
        ax.plot(
            x,
            c["total_reconstructed"],
            color=palette[1],
            ls="--",
            label="Reconstructed",
        )
        ax.set_title(f"Vintage {vin}")
        _date_axis(ax)
    axes[0].set_ylabel(f"Cases by onset week ({location})")
    axes[0].legend()
    return fig


def plot_simulation(parameters: pd.DataFrame, mode: str = "light") -> Figure:
    """Plot the true parameters of a simulated outbreak.

    Parameters
    ----------
    parameters : pandas.DataFrame
        `SimulatedOutbreak.parameters`.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        Panels for onsets, Rt, susceptible fraction, ascertainment,
        missing fraction and mean delay.
    """
    panels = [
        ("onsets", "Daily onsets"),
        ("rt", "Rt"),
        ("susceptible_fraction", "Susceptible fraction"),
        ("ascertainment", "Ascertainment"),
        ("missing", "Share missing onset"),
        ("delay_mean", "Mean delay (days)"),
    ]
    fig, axes = _grid(len(panels), 3, width=3.0, height=2.2)
    palette = MODEL_COLOURS[mode]
    locs = list(parameters["location"].unique())
    for ax, (col, title) in zip(axes, panels, strict=True):
        shared = col in ("ascertainment", "missing", "delay_mean")
        for i, loc in enumerate(locs[:1] if shared else locs):
            p = parameters.loc[parameters["location"] == loc]
            ax.plot(p["date"], p[col], color=palette[i], lw=1.5, label=loc)
        ax.set_title(title)
        _date_axis(ax)
    axes[0].legend()
    return fig


# --------------------------------------------------------------------------
# Forecasts
# --------------------------------------------------------------------------


def _bands(samples: pd.DataFrame) -> pd.DataFrame:
    grp = samples.groupby("target_end_date")["value"]
    out = grp.quantile(np.array([0.05, 0.25, 0.5, 0.75, 0.95])).unstack()
    out.index = pd.to_datetime(out.index)
    return out


def _draw_forecast(
    ax: Axes, samples: pd.DataFrame, colour: str, label: str
) -> None:
    """Draw sample trajectories, 50% and 90% ribbons and the median."""
    paths = samples.pivot_table(
        index="target_end_date", columns="sample", values="value"
    ).sort_index()
    x = pd.to_datetime(paths.index)
    for s in paths.columns:
        ax.plot(x, paths[s], color=colour, lw=0.4, alpha=0.08, zorder=1)
    q = _bands(samples)
    ax.fill_between(q.index, q[0.05], q[0.95], color=colour, alpha=0.18, lw=0)
    ax.fill_between(q.index, q[0.25], q[0.75], color=colour, alpha=0.35, lw=0)
    ax.plot(q.index, q[0.5], color=colour, lw=1.5, label=label)


def plot_forecasts(
    forecasts: pd.DataFrame,
    target: pd.DataFrame,
    location: str,
    models: Sequence[str] | None = None,
    kind: str = "real",
    truth_vintage: str | None = None,
    weeks_before: int = 10,
    mode: str = "light",
) -> Figure:
    """Plot forecasts against data at the time and later data, by date.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Output of `bdbv_forecasts.backtest.run_forecasts`.
    target : pandas.DataFrame
        Target rows holding each forecast's vintage and the truth vintage.
        For reconstructed forecasts, pass the reconstructed rows too.
    location : str
        Location to show.
    models : sequence of str, optional
        Models in registry order. Defaults to those in `forecasts`.
    kind : str
        Vintage kind to show.
    truth_vintage : str, optional
        Vintage drawn as later data. Defaults to the latest real vintage.
    weeks_before : int
        Weeks of data shown before each forecast.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per reference date: data at the time (dashed), later
        data (solid) and each model's sample trajectories, 50% and 90%
        intervals and median.
    """
    fc = forecasts.loc[
        (forecasts["location"] == location) & (forecasts["kind"] == kind)
    ]
    models = list(models or fc["model"].unique())
    real = target.loc[target["kind"] == "real"]
    truth_v = truth_vintage or real["vintage"].max()
    truth = real.loc[
        (real["vintage"] == truth_v) & (real["location"] == location)
    ]
    refs = sorted(fc["reference_date"].unique())
    ink = THEMES[mode]
    fig, axes = _grid(len(refs), 4, width=3.0, height=2.3)
    for ax, ref in zip(axes, refs, strict=True):
        f_ref = fc.loc[fc["reference_date"] == ref]
        vintage = f_ref["vintage"].iloc[0]
        start = pd.Timestamp(ref) - pd.Timedelta(weeks=weeks_before)
        end = pd.Timestamp(f_ref["target_end_date"].max())
        at_time = target.loc[
            (target["kind"] == kind)
            & (target["vintage"] == vintage)
            & (target["location"] == location)
        ]
        for rows, style, lab in (
            (truth, "-", f"Data {truth_v}"),
            (at_time, "--", "Data at the time"),
        ):
            x = pd.to_datetime(rows["target_end_date"])
            sel = (x >= start) & (x <= end)
            ax.plot(
                x[sel],
                rows["total"][sel],
                color=ink["data"],
                ls=style,
                lw=1.2,
                label=lab,
            )
        for m in models:
            s = f_ref.loc[f_ref["model"] == m]
            if not s.empty:
                _draw_forecast(ax, s, model_colour(m, models, mode), m)
        ax.axvline(
            mdates.date2num(pd.Timestamp(ref)),
            color=ink["muted"],
            lw=0.8,
            ls=":",
        )
        ax.set_title(f"{ref} ({kind})")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[0].set_ylabel(f"Cases by onset week ({location})")
    axes[0].legend(fontsize=7)
    return fig


def plot_nowcast(
    location: str,
    vintage: str | None = None,
    models: Sequence[str] = ("persistence",),
    target: pd.DataFrame | None = None,
    delays: pd.DataFrame | None = None,
    forecasts: pd.DataFrame | None = None,
    weeks_before: int = 16,
    mode: str = "light",
    data_dir: Path | str | None = None,
    config_path: Path | str = "config/backtest.toml",
    ax: Axes | None = None,
) -> Figure:
    """Plot the nowcast and forecast for one location from one vintage.

    If `forecasts` is not given, the models are run on the vintage.
    This is the figure the submission pull request embeds.

    Parameters
    ----------
    location : str
        Location to show, such as `"CD-IT"` or `"CD"`.
    vintage : str, optional
        Real vintage to forecast from. Defaults to the latest.
    models : sequence of str
        Registered models to show.
    target, delays : pandas.DataFrame, optional
        Tidy tables. Loaded from `data_dir` if not given.
    forecasts : pandas.DataFrame, optional
        Output of `run_forecasts` for this vintage.
    weeks_before : int
        Weeks of data shown before the forecast.
    mode : str
        `"light"` or `"dark"`.
    data_dir : pathlib.Path or str, optional
        Directory of the tidy tables.
    config_path : pathlib.Path or str
        Backtest settings, for horizons and samples.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on, for a grid of locations. A new figure if None.

    Returns
    -------
    matplotlib.figure.Figure
        Observed, imputed and total cases by onset week, and each model's
        sample trajectories, 50% and 90% intervals and median.
    """
    from .backtest import load_config, make_tasks, run_forecasts
    from .data import load_delays, load_target
    from .models.base import get_model

    target = load_target(data_dir) if target is None else target
    real = target.loc[target["kind"] == "real"]
    vintage = vintage or str(real["vintage"].max())
    rows = real.loc[real["vintage"] == vintage]
    if forecasts is None:
        delays = load_delays(data_dir) if delays is None else delays
        config = load_config(config_path)
        tasks = make_tasks(rows, delays, config)
        forecasts = run_forecasts([get_model(m) for m in models], tasks)
    fc = forecasts.loc[
        (forecasts["location"] == location) & (forecasts["vintage"] == vintage)
    ]
    ref = pd.Timestamp(fc["reference_date"].iloc[0])
    data = rows.loc[rows["location"] == location].copy()
    data["date"] = pd.to_datetime(data["target_end_date"])
    start = ref - pd.Timedelta(days=7 * weeks_before)
    data = data.loc[data["date"] >= start]
    ink = THEMES[mode]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    else:
        fig = cast(Figure, ax.get_figure())
    ax.bar(
        data["date"],
        data["observed"],
        width=5,
        color=ink["axis"],
        label="Known onset",
    )
    ax.bar(
        data["date"],
        data["imputed"],
        width=5,
        bottom=data["observed"],
        color=ink["muted"],
        label="Imputed onset",
    )
    for m in models:
        s = fc.loc[fc["model"] == m]
        _draw_forecast(ax, s, model_colour(m, list(models), mode), m)
    ax.axvline(mdates.date2num(ref), color=ink["muted"], lw=0.8, ls=":")
    names = _names(rows)
    ax.set_title(
        f"{names.get(location, location)} ({location}): vintage {vintage}, "
        f"reference date {ref.date()}"
    )
    ax.set_ylabel("Confirmed cases by onset week")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(fontsize=7, loc="upper left")
    return fig


def _label(model: str, source: str, names: Mapping[str, str]) -> str:
    name = names.get(model, model)
    if source == "hub":
        return name
    return f"{name} (retrospective, {source} vintage)"


def plot_comparison(
    forecasts: pd.DataFrame,
    target: pd.DataFrame,
    location: str,
    names: Mapping[str, str] | None = None,
    weeks_before: int = 10,
    mode: str = "light",
) -> Figure:
    """Plot hub and our forecasts side by side, one panel per round.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Hub and our samples, with `model`, `source`, `reference_date`,
        `location`, `target_end_date`, `sample` and `value`.
    target : pandas.DataFrame
        Target rows; the latest real vintage is drawn as data.
    location : str
        Location to show.
    names : mapping, optional
        Labels from `display_names`. Hub models not in it are left out,
        and our models keep their id.
    weeks_before : int
        Weeks of data shown before each reference date.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        Each model's sample trajectories, 50% and 90% intervals and
        median, over the latest data.
    """
    names = dict(names or {})
    fc = forecasts.loc[forecasts["location"] == location]
    fc = fc.loc[(fc["source"] != "hub") | fc["model"].isin(list(names))].assign(
        label=lambda d: [
            _label(m, s, names)
            for m, s in zip(d["model"], d["source"], strict=True)
        ]
    )
    labels = sorted(set(fc["label"]))
    real = target.loc[target["kind"] == "real"]
    data = real.loc[
        (real["vintage"] == real["vintage"].max())
        & (real["location"] == location)
    ]
    refs = sorted(set(fc["reference_date"]))
    ink = THEMES[mode]
    fig, axes = _grid(max(len(refs), 1), 3, width=3.6, height=2.6)
    for ax, ref in zip(axes, refs, strict=False):
        f_ref = fc.loc[fc["reference_date"] == ref]
        start = pd.Timestamp(ref) - pd.Timedelta(weeks=weeks_before)
        end = pd.Timestamp(f_ref["target_end_date"].max())
        x = pd.to_datetime(data["target_end_date"])
        sel = (x >= start) & (x <= end)
        ax.plot(
            x[sel],
            data["total"][sel],
            color=ink["data"],
            lw=1.2,
            label=f"Data {real['vintage'].max()}",
        )
        for lab in labels:
            s = f_ref.loc[f_ref["label"] == lab]
            if not s.empty:
                _draw_forecast(ax, s, model_colour(lab, labels, mode), lab)
        ax.axvline(
            mdates.date2num(pd.Timestamp(ref)),
            color=ink["muted"],
            lw=0.8,
            ls=":",
        )
        ax.set_title(str(ref))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[0].set_ylabel(f"Cases by onset week ({location})")
    axes[0].legend(fontsize=7)
    return fig


def read_submission(path: Path | str, vintage: str) -> pd.DataFrame:
    """Read a hub submission file as forecasts for `plot_nowcast`.

    Parameters
    ----------
    path : pathlib.Path or str
        A hub file with sample output.
    vintage : str
        The data vintage it was made from.

    Returns
    -------
    pandas.DataFrame
        Columns `model` (`"submission"`), `vintage`, `reference_date`,
        `location`, `target_end_date`, `sample` and `value`.
    """
    table = pd.read_csv(path)
    return pd.DataFrame(
        {
            "model": "submission",
            "vintage": vintage,
            "reference_date": table["reference_date"].astype(str),
            "location": table["location"].astype(str),
            "target_end_date": table["target_end_date"].astype(str),
            "sample": table["output_type_id"].astype("int64"),
            "value": table["value"].astype(float),
        }
    )


def plot_submission(
    submission: Path | str,
    vintage: str,
    target: pd.DataFrame | None = None,
    n_provinces: int = 3,
    data_dir: Path | str | None = None,
) -> Figure:
    """Plot what was submitted: the national total and the largest provinces.

    Draws the samples in the file, not a re-run of the model, with the
    data of the vintage behind them. Light mode, about 1600 by 1000 pixels
    at 100 dpi, for the submission pull request.

    Parameters
    ----------
    submission : pathlib.Path or str
        The hub file.
    vintage : str
        The data vintage the file was made from.
    target : pandas.DataFrame, optional
        Tidy target table. Loaded from `data_dir` if not given.
    n_provinces : int
        Provinces shown beside the national total, largest median first.
    data_dir : pathlib.Path or str, optional
        Directory of the tidy tables.

    Returns
    -------
    matplotlib.figure.Figure
        A grid with one panel per location.
    """
    from .data import load_target

    target = load_target(data_dir) if target is None else target
    fc = read_submission(submission, vintage)
    sizes = (
        fc.loc[fc["location"] != "CD"]
        .groupby("location")["value"]
        .median()
        .sort_values(ascending=False)
    )
    locations = [
        loc
        for loc in ["CD", *sizes.index[:n_provinces]]
        if loc in set(fc["location"])
    ]
    ncol = 2
    nrow = max(1, -(-len(locations) // ncol))
    with theme("light"):
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(16, 5 * nrow), layout="constrained"
        )
        flat = list(np.atleast_1d(axes).ravel())
        for loc, ax in zip(locations, flat, strict=False):
            plot_nowcast(
                loc,
                vintage=vintage,
                models=("submission",),
                target=target,
                forecasts=fc,
                ax=ax,
            )
        for ax in flat[len(locations) :]:
            ax.set_visible(False)
    return fig


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------


def plot_scores(
    scores: pd.DataFrame,
    by: str = "horizon",
    metric: str = "wis",
    models: Sequence[str] | None = None,
    mode: str = "light",
) -> Figure:
    """Plot mean scores by horizon or location, one panel per vintage kind.

    Parameters
    ----------
    scores : pandas.DataFrame
        Output of `score_forecasts`.
    by : str
        `"horizon"` or `"location"`.
    metric : str
        A score column, such as `"wis"`, `"log_wis"`, `"coverage_90"` or
        `"bias"`.
    models : sequence of str, optional
        Models in registry order.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        Mean score by `by`, with the number of forecasts in the title.
    """
    models = list(models or scores["model"].unique())
    kinds = [
        k
        for k in ("real", "reconstructed", "simulated")
        if k in set(scores["kind"])
    ]
    fig, axes = _grid(len(kinds), 3, width=3.6, height=2.6)
    for ax, kind in zip(axes, kinds, strict=True):
        s = scores.loc[scores["kind"] == kind]
        mean = s.groupby(["model", by])[metric].mean().unstack(0)
        if by == "location":
            order = s.groupby("location")["truth"].sum().sort_values()
            mean = mean.reindex(order.index[::-1])
        x = np.arange(len(mean))
        width = 0.8 / max(1, len(models))
        for i, m in enumerate(models):
            if m not in mean.columns:
                continue
            col = model_colour(m, models, mode)
            if by == "horizon":
                ax.plot(
                    mean.index,
                    mean[m],
                    color=col,
                    marker=_KIND_MARKERS[kind],
                    ms=5,
                    label=m,
                )
            else:
                ax.bar(x + i * width, mean[m], width, color=col, label=m)
        if by != "horizon":
            ax.set_xticks(
                x + width * (len(models) - 1) / 2,
                [str(v).removeprefix("CD-") for v in mean.index],
            )
        if metric.startswith("coverage"):
            level = 0.5 if metric.endswith("50") else 0.9
            ax.axhline(level, color=THEMES[mode]["muted"], lw=1, ls=":")
        if metric == "bias":
            ax.axhline(0, color=THEMES[mode]["muted"], lw=1, ls=":")
        ax.set_title(
            f"{kind}: {s[['vintage']].nunique().iloc[0]} vintages, n {len(s)}"
        )
        ax.set_xlabel(by.capitalize())
    axes[0].set_ylabel(_METRIC_LABELS.get(metric, metric))
    axes[0].legend()
    return fig


_METRIC_LABELS: Final[dict[str, str]] = {
    "wis": "Mean WIS",
    "log_wis": "Mean WIS, log scale",
    "coverage_50": "50% interval coverage",
    "coverage_90": "90% interval coverage",
    "bias": "Mean bias",
    "relative_wis": "WIS relative to persistence",
    "relative_log_wis": "Log WIS relative to persistence",
}


def plot_relative_skill(
    scores: pd.DataFrame,
    by: str = "horizon",
    metric: str = "wis",
    baseline: str = "persistence",
    models: Sequence[str] | None = None,
    mode: str = "light",
) -> Figure:
    """Plot each model's score relative to the baseline.

    Parameters
    ----------
    scores : pandas.DataFrame
        Output of `score_forecasts` with at least the baseline.
    by : str
        `"horizon"` or `"location"`.
    metric : str
        `"wis"` or `"log_wis"`.
    baseline : str
        The reference model.
    models : sequence of str, optional
        Models in registry order.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        Relative score by `by`, one panel per vintage kind, with a line at
        1. With only the baseline, the panels say so.
    """
    from .scoring import relative_skill

    models = list(models or scores["model"].unique())
    kinds = [
        k
        for k in ("real", "reconstructed", "simulated")
        if k in set(scores["kind"])
    ]
    fig, axes = _grid(len(kinds), 3, width=3.6, height=2.6)
    col = f"relative_{metric}"
    for ax, kind in zip(axes, kinds, strict=True):
        rel = relative_skill(
            scores.loc[scores["kind"] == kind],
            baseline=baseline,
            by=(by,),
            metrics=(metric,),
        )
        ax.axhline(1, color=THEMES[mode]["muted"], lw=1, ls=":")
        others = [m for m in models if m != baseline]
        if not others:
            ax.text(
                0.5,
                0.5,
                f"Only {baseline} so far",
                transform=ax.transAxes,
                ha="center",
                color=THEMES[mode]["muted"],
            )
        for m in others:
            r = rel.loc[rel["model"] == m]
            ax.plot(
                r[by].astype(str) if by != "horizon" else r[by],
                r[col],
                color=model_colour(m, models, mode),
                marker=_KIND_MARKERS[kind],
                label=m,
            )
        ax.set_title(kind)
        ax.set_xlabel(by.capitalize())
    axes[0].set_ylabel(_METRIC_LABELS.get(col, col))
    if len(models) > 1:
        axes[0].legend()
    return fig
