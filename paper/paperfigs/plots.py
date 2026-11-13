"""The results panel used for each study, and the SI score plots.

The score is the WIS for quantile forecasts and the CRPS for samples, and
the labels follow it.

`results_panel` follows the layout of baselinenowcast-paper's
`make_fig_hub_validation`: nowcasts over time across the top, then
summary scores, then scores by location and by horizon. Relative scores
divide by the baseline's mean score on matched forecasts and are shown on
a log axis with a dashed line at 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    MaxNLocator,
    NullFormatter,
)

from paperfigs.factors import (
    bin_factors,
    factor_label,
    relative_by_factor,
    top_factors,
)
from paperfigs.schema import DEFAULT_STRATUM
from paperfigs.scores import (
    STUDY_BASELINES,
    coverage_columns,
    metric_of,
    relative_scores,
    summarise_scores,
)
from paperfigs.style import (
    INK,
    bar_style,
    line_style,
    model_order,
    point_style,
)

__all__ = [
    "PANEL_LETTERS",
    "STUDY_HORIZON_BIN",
    "results_panel",
    "si_scores_by_factor",
    "si_scores_by_location_horizon",
    "si_wis_over_time",
]

PANEL_LETTERS: Final = ("A", "B", "C", "D", "E", "F", "G", "H")
"""Letters of the results panel, in reading order. The last is used only
when the study has strata."""

_TITLES: Final = {
    "nowcast": "Nowcasts over time",
    "rel": "Relative {m}",
    "rel_log": "Relative {m}, log scale",
    "bias": "Bias",
    "coverage": "Coverage",
    "location": "Relative {m} by location",
    "factor": "Relative {m} by scenario factor",
    "stratum": "Relative {m} by stratum",
    "horizon": "Relative {m} by horizon",
}

_RATIO_TICKS: Final = (0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10)

_STEP_UNIT: Final = {1: "days", 7: "weeks"}

STUDY_HORIZON_BIN: Final[dict[str, int]] = {"synthetic": 7}
"""Studies whose horizons, in days, are binned to weeks for the panels.

The synthetic targets are weekly totals with origins on varying weekdays,
so day horizons are ragged; `horizon // 7` gives weeks from the origin's
week.
"""


def _label(ax: Axes, letter: str, title: str = "") -> None:
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=12)
    if title:
        ax.set_title(title, loc="center")


def _one_study(forecasts: pd.DataFrame) -> str:
    studies = forecasts["study"].unique()
    if len(studies) != 1:
        raise ValueError("The results panel shows one study at a time.")
    return str(studies[0])


def _step_days(forecasts: pd.DataFrame) -> int:
    nonzero = forecasts.loc[forecasts["horizon"] != 0]
    if nonzero.empty:
        return 1
    row = nonzero.iloc[0]
    delta = (row["target_date"] - row["origin_date"]).days
    return abs(int(delta / row["horizon"]))


def _log_ratio_axis(ax: Axes, axis: str = "y") -> None:
    """Log axis for ratios with plain tick labels and a line at 1."""
    which = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    which.set_major_locator(FixedLocator(_RATIO_TICKS))
    which.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    which.set_minor_formatter(NullFormatter())
    line = ax.axhline if axis == "y" else ax.axvline
    line(1, color=INK["muted"], linestyle="--", linewidth=1, zorder=0)


def _pad_log_limits(ax: Axes, values: np.ndarray, axis: str = "y") -> None:
    v = values[np.isfinite(values) & (values > 0)]
    if v.size == 0:
        return
    lo, hi = min(v.min(), 1.0), max(v.max(), 1.0)
    lims = (lo / 1.15, hi * 1.15)
    (ax.set_ylim if axis == "y" else ax.set_xlim)(*lims)


def _ribbon_models(models: Sequence[str], baseline: str) -> list[str]:
    keep = [m for m in models if m == baseline or m.endswith("_ft")]
    keep += [m for m in models if m == "hub_ensemble"]
    return model_order(keep)


def _spaced_origins(forecasts: pd.DataFrame, n_max: int = 5) -> list:
    """Origins far enough apart that their nowcast windows do not overlap."""
    origins = np.sort(forecasts["origin_date"].unique())
    span = forecasts["horizon"].max() - forecasts["horizon"].min() + 1
    step = pd.Timedelta(days=_step_days(forecasts))
    per = pd.Timedelta(origins[1] - origins[0]) if len(origins) > 1 else step
    gap = max(1, int(np.ceil(span * step / per)))
    chosen = origins[::gap]
    if len(chosen) > n_max:
        chosen = chosen[np.linspace(0, len(chosen) - 1, n_max).astype(int)]
    return list(chosen)


def _quantile_table(one: pd.DataFrame) -> pd.DataFrame:
    """Quantiles by target date; sample forecasts use linear quantiles."""
    if one["quantile_level"].notna().all():
        return one.pivot_table(
            index="target_date", columns="quantile_level", values="value"
        )
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    return one.groupby("target_date")["value"].quantile(levels).unstack()


def _main_stratum(df: pd.DataFrame) -> str:
    """Return `all` if present, else the first stratum."""
    strata = sorted(df["stratum"].unique())
    return DEFAULT_STRATUM if DEFAULT_STRATUM in strata else strata[0]


def _largest_location(observations: pd.DataFrame) -> str:
    final = observations.loc[
        observations["as_of_date"] == observations["as_of_date"].max()
    ]
    return str(final.groupby("location")["value"].sum().idxmax())


def plot_nowcasts(
    ax: Axes,
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    models: Sequence[str],
    location: str,
    all_models: Sequence[str],
) -> None:
    """Draw 50% and 90% ribbons and medians at spaced origin dates.

    Data as of each origin are in light grey, the final data in black.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    forecasts : pandas.DataFrame
        Forecasts for one study.
    observations : pandas.DataFrame
        Observations for the same study.
    models : sequence of str
        Models to draw ribbons for.
    location : str
        Location to show.
    all_models : sequence of str
        Every model in the figure, for styles.
    """
    fc = forecasts.loc[forecasts["location"] == location]
    obs = observations.loc[observations["location"] == location]
    origins = _spaced_origins(fc)
    final = obs.loc[obs["as_of_date"] == obs["as_of_date"].max()]
    step = pd.Timedelta(days=_step_days(fc))
    hmin, hmax = fc["horizon"].min(), fc["horizon"].max()
    start = origins[0] + (hmin - abs(hmin) // 2) * step
    end = origins[-1] + (max(hmax, 0) + 3) * step
    window = final.loc[final["target_date"].between(start, end)]
    for origin in origins:
        asof = obs.loc[
            (obs["as_of_date"] == origin)
            & (obs["target_date"] >= origin + (hmin - abs(hmin) // 2) * step)
        ]
        ax.plot(
            asof["target_date"],
            asof["value"],
            color=INK["asof"],
            lw=1.4,
            zorder=4.5,
        )
        ax.axvline(origin, color=INK["muted"], lw=0.8, ls=":", zorder=1)
        for model in models:
            one = fc.loc[(fc["model"] == model) & (fc["origin_date"] == origin)]
            if one.empty:
                continue
            wide = _quantile_table(one)
            colour = line_style(model, all_models)["color"]
            for lo, hi, alpha in ((0.05, 0.95, 0.15), (0.25, 0.75, 0.3)):
                if lo in wide.columns and hi in wide.columns:
                    ax.fill_between(
                        wide.index,
                        wide[lo],
                        wide[hi],
                        color=colour,
                        alpha=alpha,
                        lw=0,
                        zorder=3,
                    )
            ax.plot(
                wide.index,
                wide[0.5],
                lw=1.2,
                zorder=4,
                **line_style(model, all_models),
            )
    ax.plot(
        window["target_date"],
        window["value"],
        color=INK["data"],
        lw=1.0,
        marker="o",
        ms=1.8,
        zorder=5,
    )
    ax.set_xlim(start, end)
    # Cap the axis so one wide ribbon does not flatten the data.
    ax.set_ylim(0, 1.6 * float(window["value"].max()))
    ax.set_ylabel(f"Count, {location}")
    locator = mdates.AutoDateLocator(maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _model_rows(ax: Axes, models: Sequence[str], show_names: bool) -> None:
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models if show_names else [])
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)


def _bars_relative(
    ax: Axes, rel: pd.DataFrame, col: str, models: Sequence[str]
) -> None:
    values = rel.set_index("model")[col].reindex(models)
    for i, m in enumerate(models):
        ax.barh(i, values[m] - 1, left=1, height=0.7, **bar_style(m, models))
    _log_ratio_axis(ax, "x")
    _pad_log_limits(ax, values.to_numpy(dtype=float), "x")


def _bars_bias(ax: Axes, summary: pd.DataFrame, models: Sequence[str]) -> None:
    values = summary.set_index("model")["bias"].reindex(models)
    for i, m in enumerate(models):
        ax.barh(i, values[m], height=0.7, **bar_style(m, models))
    ax.axvline(0, color=INK["muted"], ls="--", lw=1, zorder=0)
    lim = max(0.25, float(np.nanmax(np.abs(values))) * 1.15)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Mean bias")


def _bars_coverage(
    ax: Axes, summary: pd.DataFrame, models: Sequence[str]
) -> None:
    """Narrowest interval solid, widest light, with lines at nominal."""
    cols = coverage_columns(summary)
    narrow, wide = cols[0], cols[-1]
    levels = [int(c.split("_")[1]) / 100 for c in (narrow, wide)]
    s = summary.set_index("model").reindex(models)
    for i, m in enumerate(models):
        style = bar_style(m, models)
        ax.barh(i - 0.18, s.loc[m, wide], height=0.34, alpha=0.45, **style)
        ax.barh(i + 0.18, s.loc[m, narrow], height=0.34, **style)
    for level in levels:
        ax.axvline(level, color=INK["muted"], ls="--", lw=1, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, *levels])
    ax.set_xlabel(f"Coverage ({levels[0]:.0%} solid, {levels[1]:.0%} light)")


def _points_by(
    ax: Axes,
    rel: pd.DataFrame,
    by: str,
    models: Sequence[str],
    col: str = "rel_score",
    lines: bool = False,
    rotate: bool = False,
    order: Sequence | None = None,
) -> None:
    cats = (
        list(order)
        if order is not None
        else sorted(rel[by].unique(), key=lambda v: (str(type(v)), v))
    )
    x = np.arange(len(cats))
    width = 0.7 / max(len(models), 1)
    for j, m in enumerate(models):
        one = rel.loc[rel["model"] == m].set_index(by)[col].reindex(cats)
        if lines:
            ax.plot(cats, one.to_numpy(), lw=1.4, **line_style(m, models))
        else:
            offset = (j - (len(models) - 1) / 2) * width
            ax.plot(
                x + offset,
                one.to_numpy(),
                ls="none",
                marker="o",
                ms=4.5,
                **point_style(m, models),
            )
    if not lines:
        ax.set_xticks(x)
        if rotate:
            ax.set_xticklabels([str(c) for c in cats], rotation=45, ha="right")
        else:
            ax.set_xticklabels([str(c) for c in cats])
        ax.grid(axis="x", visible=False)
    _log_ratio_axis(ax, "y")
    _pad_log_limits(ax, rel[col].to_numpy(dtype=float), "y")


def _legend(fig: Figure, models: Sequence[str]) -> None:
    handles = [
        Line2D(
            [], [], marker="s", ms=6, lw=1.6, label=m, **line_style(m, models)
        )
        for m in models
    ]
    handles.append(
        Line2D([], [], color=INK["asof"], lw=1.2, label="Data as of origin")
    )
    handles.append(
        Line2D(
            [],
            [],
            color=INK["data"],
            lw=1,
            marker="o",
            ms=2,
            label="Final data",
        )
    )
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(int(np.ceil(len(handles) / 2)), 6),
        handlelength=2.6,
    )


def _level_order(levels: Sequence[str]) -> list[str]:
    """Order levels by their leading number if they have one."""

    def key(v: str) -> tuple[int, float, str]:
        try:
            return (0, float(v.split(" to ")[0]), v)
        except ValueError:
            return (1, 0.0, v)

    return sorted(levels, key=key)


def _factor_points(
    ax: Axes,
    scores: pd.DataFrame,
    factors: pd.DataFrame,
    baseline: str,
    models: Sequence[str],
    n_factors: int,
) -> None:
    """Draw relative scores by level for the most spread factors."""
    rel = relative_by_factor(scores, bin_factors(factors), baseline)
    keep = top_factors(rel, n_factors)
    rel = rel.loc[rel["factor"].isin(keep)].copy()
    cats: list[str] = []
    groups = []
    for f in keep:
        levels = _level_order(rel.loc[rel["factor"] == f, "level"].unique())
        groups.append((f, len(cats), len(cats) + len(levels) - 1))
        cats += [f"{f}|{lv}" for lv in levels]
    rel["cat"] = rel["factor"] + "|" + rel["level"]
    _points_by(ax, rel, "cat", models, rotate=True, order=cats)
    ax.set_xticklabels([c.split("|", 1)[1] for c in cats])
    for i, (f, lo, hi) in enumerate(groups):
        if i:
            ax.axvline(lo - 0.5, color=INK["axis"], lw=0.8)
        ax.text(
            (lo + hi) / 2,
            0.98,
            factor_label(f),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=INK["text"],
        )


def results_panel(
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    scores: pd.DataFrame,
    baseline: str | None = None,
    location: str | None = None,
    ribbon_models: Sequence[str] | None = None,
    factors: pd.DataFrame | None = None,
    n_factors: int = 4,
) -> Figure:
    """Plot the results panel for one study.

    Panels: A nowcasts over time at spaced origin dates; B relative score;
    C relative score on the log scale; D mean bias; E coverage at the
    study's two widths; F relative score by location (main stratum); then,
    for a study with strata, G relative score by stratum; and last the
    relative score by horizon.
    The score is the WIS for quantile forecasts and the CRPS for samples.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Forecasts for one study, in the schema.
    observations : pandas.DataFrame
        Observations for that study.
    scores : pandas.DataFrame
        `score_forecasts` output for the forecasts.
    baseline : str, optional
        Model relative scores are divided by. Defaults to the study's
        entry in `STUDY_BASELINES`.
    location : str, optional
        Location for panel A. Defaults to the one with most cases.
    factors : pandas.DataFrame, optional
        Scenario factor values by `location` (synthetic study). When
        given, the panel by location is replaced by one by the `n_factors`
        factors whose levels spread the relative score most.
    n_factors : int
        Factors to show when `factors` is given.
    ribbon_models : sequence of str, optional
        Models with ribbons in panel A. Defaults to the baseline, the
        fine-tuned models and `hub_ensemble` if present.

    Returns
    -------
    matplotlib.figure.Figure
        The panel.

    Raises
    ------
    ValueError
        If the forecasts span more than one study.
    """
    study = _one_study(forecasts)
    obs = observations.loc[observations["study"] == study]
    sc = scores.loc[scores["study"] == study]
    baseline = baseline or STUDY_BASELINES[study]
    name = metric_of(sc).upper()
    models = model_order(sc["model"].unique())
    main = _main_stratum(sc)
    stratified = len(sc["stratum"].unique()) > 1
    location = location or _largest_location(obs.loc[obs["stratum"] == main])
    ribbons = list(ribbon_models or _ribbon_models(models, baseline))
    where = "location" if factors is None else "factor"
    if stratified:
        bottom = [where] * 5 + ["stratum"] * 3 + ["horizon"] * 4
    else:
        bottom = [where] * 7 + ["horizon"] * 5
    top = [["nowcast"] * 12, ["rel"] * 3 + ["rel_log"] * 3]
    top[1] += ["bias"] * 3 + ["coverage"] * 3
    fig, axd = plt.subplot_mosaic(
        [*top, bottom],
        figsize=(12, 11),
        height_ratios=[1.25, 1, 1],
        layout="constrained",
    )
    plot_nowcasts(
        axd["nowcast"],
        forecasts.loc[forecasts["stratum"] == main],
        obs.loc[obs["stratum"] == main],
        ribbons,
        location,
        models,
    )
    rel = relative_scores(sc, baseline=baseline)
    summary = summarise_scores(sc)
    _bars_relative(axd["rel"], rel, "rel_score", models)
    axd["rel"].set_xlabel(f"Relative to {baseline}")
    _bars_relative(axd["rel_log"], rel, "rel_log_score", models)
    axd["rel_log"].set_xlabel(f"Relative to {baseline}")
    _bars_bias(axd["bias"], summary, models)
    _bars_coverage(axd["coverage"], summary, models)
    for key in ("rel", "rel_log", "bias", "coverage"):
        _model_rows(axd[key], models, show_names=key == "rel")
    # Locations are compared on their main stratum (all ages).
    if factors is None:
        _points_by(
            axd["location"],
            relative_scores(
                sc.loc[sc["stratum"] == main], ["location"], baseline
            ),
            "location",
            models,
        )
    else:
        _factor_points(axd["factor"], sc, factors, baseline, models, n_factors)
    if stratified:
        _points_by(
            axd["stratum"],
            relative_scores(sc, ["stratum"], baseline),
            "stratum",
            models,
            rotate=True,
        )
    bin_days = STUDY_HORIZON_BIN.get(study)
    hsc = (
        sc if bin_days is None else sc.assign(horizon=sc["horizon"] // bin_days)
    )
    _points_by(
        axd["horizon"],
        relative_scores(hsc, ["horizon"], baseline),
        "horizon",
        models,
        lines=True,
    )
    unit = _STEP_UNIT.get(bin_days or _step_days(forecasts), "steps")
    axd["horizon"].xaxis.set_major_locator(MaxNLocator(integer=True))
    axd["horizon"].set_xlabel(f"Horizon ({unit})")
    order = ["nowcast", "rel", "rel_log", "bias", "coverage", where]
    order += ["stratum", "horizon"] if stratified else ["horizon"]
    for letter, key in zip(PANEL_LETTERS, order, strict=False):
        _label(axd[key], letter, _TITLES[key].format(m=name))
    for key in bottom:
        axd[key].set_ylabel(f"Relative {name}")
    _legend(fig, models)
    return fig


def si_scores_by_location_horizon(
    scores: pd.DataFrame,
    baseline: str | None = None,
    col: str = "rel_score",
    ncol: int = 4,
) -> Figure:
    """Plot the relative score by horizon for every location and stratum.

    Parameters
    ----------
    scores : pandas.DataFrame
        `score_forecasts` output for one study.
    baseline : str, optional
        Reference model. Defaults to the study's in `STUDY_BASELINES`.
    col : str
        `rel_score` or `rel_log_score`.
    ncol : int
        Panels per row.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per location, sharing axes.
    """
    name = metric_of(scores).upper()
    rel = relative_scores(scores, ["location", "stratum", "horizon"], baseline)
    main = _main_stratum(rel)
    rel["series"] = rel["location"].where(
        rel["stratum"] == main, rel["location"] + " " + rel["stratum"]
    )
    models = model_order(rel["model"].unique())
    # Each location's main stratum comes before its other strata.
    order = rel.assign(_sub=rel["stratum"] != main).sort_values(
        ["location", "_sub", "stratum"]
    )
    locs = list(dict.fromkeys(order["series"]))
    nrow = int(np.ceil(len(locs) / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3 * ncol, 2.4 * nrow + 0.6),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    flat = list(axes.ravel())
    for ax in flat[len(locs) :]:
        ax.remove()
    for ax, loc in zip(flat, locs, strict=False):
        _points_by(
            ax,
            rel.loc[rel["series"] == loc],
            "horizon",
            models,
            col,
            lines=True,
        )
        ax.set_title(loc)
    _pad_log_limits(flat[0], rel[col].to_numpy(dtype=float))
    scale = "log scale" if col == "rel_log_score" else "natural scale"
    fig.supylabel(f"Relative {name} ({scale})", fontsize=9)
    for ax in flat[max(0, len(locs) - ncol) : len(locs)]:
        ax.set_xlabel("Horizon")
    _legend_models(fig, models)
    return fig


def _legend_models(fig: Figure, models: Sequence[str]) -> None:
    handles = [
        Line2D([], [], lw=1.6, label=m, **line_style(m, models)) for m in models
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(len(handles), 4),
        handlelength=2.6,
    )


def si_wis_over_time(scores: pd.DataFrame) -> Figure:
    """Plot the mean score by origin date, natural and log scale.

    Parameters
    ----------
    scores : pandas.DataFrame
        `score_forecasts` output for one study.

    Returns
    -------
    matplotlib.figure.Figure
        Two panels: the score on a log axis, and the log-scale score.
    """
    name = metric_of(scores).upper()
    summary = summarise_scores(scores, ["origin_date"])
    models = model_order(summary["model"].unique())
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 3.8), sharex=True, layout="constrained"
    )
    for ax, col, letter, title in (
        (axes[0], "score", "A", name),
        (axes[1], "log_score", "B", f"{name}, log scale"),
    ):
        for m in models:
            one = summary.loc[summary["model"] == m]
            ax.plot(
                one["origin_date"], one[col], lw=1.4, **line_style(m, models)
            )
        _label(ax, letter, title)
        locator = mdates.AutoDateLocator(maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_xlabel("Origin date")
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].set_ylabel(f"Mean {name}")
    axes[1].set_ylabel(f"Mean {name} of log(x + 1)")
    _legend_models(fig, models)
    return fig


def si_scores_by_factor(
    scores: pd.DataFrame,
    factors: pd.DataFrame,
    baseline: str | None = None,
    col: str = "rel_score",
    ncol: int = 3,
) -> Figure:
    """Plot relative scores by level for every scenario factor.

    Parameters
    ----------
    scores : pandas.DataFrame
        `score_forecasts` output for the synthetic study.
    factors : pandas.DataFrame
        Scenario factor values by `location`.
    baseline : str, optional
        Reference model. Defaults to the study's in `STUDY_BASELINES`.
    col : str
        `rel_score` or `rel_log_score`.
    ncol : int
        Panels per row.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per factor, sharing the y axis.
    """
    name = metric_of(scores).upper()
    rel = relative_by_factor(scores, bin_factors(factors), baseline)
    models = model_order(rel["model"].unique())
    names = list(dict.fromkeys(rel["factor"]))
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.6 * ncol, 2.8 * nrow + 0.6),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    flat = list(axes.ravel())
    for ax in flat[len(names) :]:
        ax.remove()
    for ax, f in zip(flat, names, strict=False):
        one = rel.loc[rel["factor"] == f]
        order = _level_order(one["level"].unique())
        _points_by(ax, one, "level", models, col, rotate=True, order=order)
        ax.set_title(factor_label(f))
    _pad_log_limits(flat[0], rel[col].to_numpy(dtype=float))
    scale = "log scale" if col == "rel_log_score" else "natural scale"
    fig.supylabel(f"Relative {name} ({scale})", fontsize=9)
    _legend_models(fig, models)
    return fig
