# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Figures and tables of the Reporting lag comparison page.

The page compares `renewal`, with a report-to-publication lag uniform on 0
to 3 days, with `renewalnolag`, which has none. It shows the release check
behind the lag, both arms' forecasts against later releases, their scores
by release and their convergence.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from bdbvhubrenewal.evaluation.nowcast import relative_wis
from bdbvhubrenewal.forecasters import list_models

from .plots import HUB_GREYS, model_colour, save_light_dark, theme

__all__ = [
    "ARMS",
    "late_reports",
    "make_lag_comparison",
    "plot_lag_fans",
    "plot_lag_scores",
    "plot_late_reports",
]

ARMS: Final = ("renewal", "renewalnolag")
"""The models compared: with the lag, and without it."""

PLACES: Final = ("CD", "CD-IT", "CD-NK")
"""Places shown in the fans."""

DAYS_BEFORE: Final = (
    ("0", (0, 0)),
    ("1", (1, 1)),
    ("2", (2, 2)),
    ("3", (3, 3)),
    ("4 to 6", (4, 6)),
    ("7 to 13", (7, 13)),
)
"""Groups of days from a delay row's report day to the earlier cut-off."""


def late_reports(delays: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Share of delay rows reported by a cut-off that missed that release.

    For each pair of consecutive delay files, a row's report day is the
    end of its onset week plus its delay, the latest it can be. Rows in
    the later file but not the earlier one, with that day on or before
    the earlier file's cut-off, were reported in time but published late.

    Parameters
    ----------
    delays : pandas.DataFrame
        Delay rows with `vintage`, `location`, `onset_week` and `delay`.
    daily : pandas.DataFrame
        Daily onsets by release, whose latest onset per vintage is its
        cut-off.

    Returns
    -------
    pandas.DataFrame
        One row per pair (and `"All"`) and group of days before the
        cut-off: `published`, `late` and `share_late`.
    """
    cut = pd.to_datetime(daily.groupby("vintage")["onset_date"].max())
    key = ["location", "onset_week", "delay"]
    vintages = sorted(delays["vintage"].unique())
    rows = []
    for a, b in pairwise(vintages):
        before = delays.loc[delays["vintage"] == a].groupby(key).size()
        after = delays.loc[delays["vintage"] == b].groupby(key).size()
        added = after.sub(before, fill_value=0).clip(lower=0)
        for label, counts in (("published", before), ("late", added)):
            frame = counts.rename("n").reset_index()
            report = pd.to_datetime(frame["onset_week"]) + pd.to_timedelta(
                frame["delay"], unit="D"
            )
            frame["days"] = (cut[a] - report).dt.days
            for name, (lo, hi) in DAYS_BEFORE:
                n = frame.loc[frame["days"].between(lo, hi), "n"].sum()
                rows.append({"pair": f"{a} to {b}", "days": name, label: n})
    long = pd.DataFrame(rows).groupby(["pair", "days"]).sum().reset_index()
    total = long.groupby("days")[["published", "late"]].sum().reset_index()
    out = pd.concat([long, total.assign(pair="All")], ignore_index=True)
    out[["published", "late"]] = out[["published", "late"]].astype(int)
    out["share_late"] = out["late"] / (out["published"] + out["late"])
    order = {name: i for i, (name, _) in enumerate(DAYS_BEFORE)}
    return out.sort_values(
        ["pair", "days"], key=lambda s: s.map(order) if s.name == "days" else s
    ).reset_index(drop=True)


def plot_late_reports(table: pd.DataFrame, mode: str = "light") -> Figure:
    """Plot the late share by days before the cut-off, for each pair.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of `late_reports`.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    with theme(mode) as ink:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        names = [name for name, _ in DAYS_BEFORE]
        x = np.arange(len(names))
        pairs = [p for p in table["pair"].unique() if p != "All"]
        greys = HUB_GREYS[mode]
        for i, pair in enumerate(pairs):
            rows = table.loc[table["pair"] == pair].set_index("days")
            ax.plot(
                x, 100 * rows.loc[names, "share_late"], marker="o",
                color=greys[i % len(greys)], lw=1, label=pair,
            )  # fmt: skip
        rows = table.loc[table["pair"] == "All"].set_index("days")
        ax.plot(
            x, 100 * rows.loc[names, "share_late"], marker="o",
            color=ink["ink"], lw=2.2, label="All pairs",
        )  # fmt: skip
        ax.set_xticks(x, names)
        ax.set_xlabel("Days from report to the earlier release's cut-off")
        ax.set_ylabel("Missed that release (%)")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
    return fig


def _quantiles(samples: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "vintage", "location", "target_end_date"]
    probs = {"q5": 0.05, "q25": 0.25, "q50": 0.5, "q75": 0.75, "q95": 0.95}
    grouped = samples.groupby(keys)["value"]
    return pd.DataFrame(
        {name: grouped.quantile(p) for name, p in probs.items()}
    ).reset_index()


def plot_lag_fans(
    samples: pd.DataFrame,
    target: pd.DataFrame,
    origins: Sequence[str],
    models: Sequence[str],
    names: dict[str, str],
    mode: str = "light",
) -> Figure:
    """Plot each arm's forecasts with the counts of each later release.

    Parameters
    ----------
    samples : pandas.DataFrame
        Final-count samples with `model`, `vintage`, `location`,
        `target_end_date` and `value`.
    target : pandas.DataFrame
        The tidy target table, for the counts each release reported.
    origins : sequence of str
        Origins, one per column.
    models : sequence of str
        Registered models in registry order, for the colours.
    names : dict
        Location code to name.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    quant = _quantiles(samples.loc[samples["model"].isin(ARMS)])
    real = target.loc[target["kind"] == "real"]
    with theme(mode) as ink:
        fig, axes = plt.subplots(
            len(PLACES), len(origins), figsize=(3.6 * len(origins), 8.4),
            squeeze=False,
        )  # fmt: skip
        greys = HUB_GREYS[mode]
        for col, origin in enumerate(origins):
            later = sorted(v for v in real["vintage"].unique() if v >= origin)
            for row, place in enumerate(PLACES):
                ax = axes[row, col]
                cell = quant.loc[
                    (quant["vintage"] == origin) & (quant["location"] == place)
                ]
                weeks = sorted(cell["target_end_date"].unique())
                for k, vintage in enumerate(later):
                    seen = real.loc[
                        (real["vintage"] == vintage)
                        & (real["location"] == place)
                        & real["target_end_date"].isin(weeks)
                    ].sort_values("target_end_date")
                    last = k == len(later) - 1
                    ax.plot(
                        pd.to_datetime(seen["target_end_date"]),
                        seen["total"], marker="o", ms=3,
                        lw=1.6 if last else 0.9,
                        color=ink["data"] if last else greys[k % len(greys)],
                        label=f"Release {vintage}",
                    )  # fmt: skip
                for model in ARMS:
                    arm = cell.loc[cell["model"] == model].sort_values(
                        "target_end_date"
                    )
                    t = pd.to_datetime(arm["target_end_date"])
                    colour = model_colour(model, models, mode)
                    ax.fill_between(
                        t, arm["q5"], arm["q95"], color=colour, alpha=0.15,
                        lw=0,
                    )  # fmt: skip
                    ax.fill_between(
                        t, arm["q25"], arm["q75"], color=colour, alpha=0.3,
                        lw=0,
                    )  # fmt: skip
                    ax.plot(t, arm["q50"], color=colour, lw=1.6, label=model)
                ax.axvline(
                    pd.Timestamp(origin), color=ink["muted"], lw=0.8, ls=":"
                )
                ax.set_ylim(bottom=0)
                ax.tick_params(axis="x", labelrotation=45, labelsize=7)
                if row == 0:
                    ax.set_title(f"Origin {origin}", fontsize=9)
                if col == 0:
                    ax.set_ylabel(names.get(place, place))
                if row == len(PLACES) - 1:
                    ax.legend(frameon=False, fontsize=6, loc="upper left")
        fig.tight_layout()
    return fig


def plot_lag_scores(table: pd.DataFrame, mode: str = "light") -> Figure:
    """Plot relative WIS and coverage by origin and truth release.

    Parameters
    ----------
    table : pandas.DataFrame
        `relative_wis` by `origin` and `truth_vintage`, for the arms.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    models = [s.name for s in list_models()]
    cells = (
        table[["origin", "truth_vintage"]]
        .drop_duplicates()
        .sort_values(["origin", "truth_vintage"])
    )
    labels = [f"{o}\nto {t}" for o, t in cells.itertuples(index=False)]
    x = np.arange(len(labels))
    panels = (
        ("rel_wis", "WIS relative to randomwalk", 1.0),
        ("coverage_50", "50% coverage", 0.5),
        ("coverage_90", "90% coverage", 0.9),
    )
    with theme(mode) as ink:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
        for ax, (col, title, ref) in zip(axes, panels, strict=True):
            for i, model in enumerate(ARMS):
                rows = cells.merge(
                    table.loc[table["model"] == model], how="left"
                )
                ax.bar(
                    x + (i - 0.5) * 0.38, rows[col], width=0.38,
                    color=model_colour(model, models, mode), label=model,
                )  # fmt: skip
            ax.axhline(ref, color=ink["muted"], lw=0.8, ls="--")
            ax.set_xticks(x, labels, fontsize=7)
            ax.set_title(title, fontsize=9)
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
    return fig


CALIBRATION_MODELS: Final = (
    "renewal",
    "renewalnolag",
    "renewalsp",
    "renewalbvd",
    "randomwalknowcast",
)
"""Models in the coverage check of the newest weeks."""

NEWEST: Final = (0, 3)
"""Weeks back from the reference week counted as the newest."""


def newest_weeks(quantiles: pd.DataFrame) -> pd.DataFrame:
    """Nowcasts of the newest weeks with where each truth fell.

    Parameters
    ----------
    quantiles : pandas.DataFrame
        `nowcast-quantiles.csv`, with `q5` to `q95`, `reported` (the count
        at the origin) and `truth` (the count a later release reported).

    Returns
    -------
    pandas.DataFrame
        The `reported` rows of `CALIBRATION_MODELS` for weeks back 0 to 3,
        with `in_50`, `in_90`, `above_90` and `below_90`, the interval
        widths over the median `width_50` and `width_90`, and the counts
        added after the origin, `added` (truth) and `added_median`.
    """
    q = quantiles.loc[
        (quantiles["target"] == "reported")
        & quantiles["weeks_back"].between(*NEWEST)
        & quantiles["model"].isin(CALIBRATION_MODELS)
    ].copy()
    median = q["q50"].clip(lower=1)
    return q.assign(
        in_50=q["truth"].between(q["q25"], q["q75"]),
        in_90=q["truth"].between(q["q5"], q["q95"]),
        above_90=q["truth"] > q["q95"],
        below_90=q["truth"] < q["q5"],
        width_50=(q["q75"] - q["q25"]) / median,
        width_90=(q["q95"] - q["q5"]) / median,
        added=q["truth"] - q["reported"],
        added_median=q["q50"] - q["reported"],
    )


def calibration_table(rows: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """Summarise coverage, misses and width of the newest weeks.

    Parameters
    ----------
    rows : pandas.DataFrame
        Output of `newest_weeks`.
    by : sequence of str
        Columns to group by besides `model`.

    Returns
    -------
    pandas.DataFrame
        `n`, the 50% and 90% coverage, the shares above and below the 90%
        interval, the median widths over the median, and the counts added
        after the origin, realised and at the median.
    """
    return (
        rows.groupby([*by, "model"])
        .agg(
            n=("in_90", "size"),
            coverage_50=("in_50", "mean"),
            coverage_90=("in_90", "mean"),
            above_90=("above_90", "mean"),
            below_90=("below_90", "mean"),
            width_50=("width_50", "median"),
            width_90=("width_90", "median"),
            added=("added", "sum"),
            added_median=("added_median", "sum"),
        )
        .reset_index()
    )


def plot_calibration(rows: pd.DataFrame, mode: str = "light") -> Figure:
    """Plot coverage and width of the newest weeks by weeks back.

    Parameters
    ----------
    rows : pandas.DataFrame
        Output of `newest_weeks`.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    models = [s.name for s in list_models()]
    table = calibration_table(rows, ("weeks_back",))
    panels = (
        ("coverage_50", "50% coverage", 0.5),
        ("coverage_90", "90% coverage", 0.9),
        ("above_90", "Above the 90% interval", 0.05),
        ("width_90", "90% width over the median", None),
    )
    with theme(mode) as ink:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
        for ax, (col, title, ref) in zip(axes, panels, strict=True):
            for model in CALIBRATION_MODELS:
                line = table.loc[table["model"] == model]
                if line.empty:
                    continue
                ax.plot(
                    line["weeks_back"], line[col], marker="o", lw=1.4,
                    color=model_colour(model, models, mode), label=model,
                )  # fmt: skip
            if ref is not None:
                ax.axhline(ref, color=ink["muted"], lw=0.8, ls="--")
            ax.set_xticks(range(NEWEST[0], NEWEST[1] + 1))
            ax.set_xlabel("Weeks back")
            ax.set_title(title, fontsize=9)
        axes[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
    return fig


def _calibration_md(rows: pd.DataFrame, by: Sequence[str]) -> str:
    table = calibration_table(rows, by)
    cols = {
        "coverage_50": "50% coverage",
        "coverage_90": "90% coverage",
        "above_90": "above 90%",
        "below_90": "below 90%",
        "width_50": "50% width",
        "width_90": "90% width",
        "added": "added after origin",
        "added_median": "added, median nowcast",
        "weeks_back": "weeks back",
        "truth_vintage": "truth release",
    }
    return _table(table.rename(columns=cols))


def _table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".2f")


def make_lag_comparison(
    fig_dir: Path,
    gen_dir: Path,
    res: Path,
    target: pd.DataFrame,
    delays: pd.DataFrame,
    daily: pd.DataFrame,
    names: dict[str, str],
) -> None:
    """Write the figures and tables of the Reporting lag comparison page.

    Parameters
    ----------
    fig_dir, gen_dir : pathlib.Path
        Where SVGs and Markdown snippets go.
    res : pathlib.Path
        Results: `forecasts.csv.gz`, `nowcast-scores.csv`,
        `backtest-fits.csv` and `lag-comparison/` (the arms' fits and
        forecasts at the latest release, which is held out).
    target, delays, daily : pandas.DataFrame
        The tidy data.
    names : dict
        Location code to name.
    """
    late = late_reports(delays, daily)
    save_light_dark(
        lambda mode: plot_late_reports(late, mode), fig_dir / "lag-late.svg"
    )
    shown = late.assign(share_late=(100 * late["share_late"]).round(1))
    (gen_dir / "lag-late.md").write_text(
        _table(
            shown.rename(
                columns={
                    "days": "days before cut-off",
                    "share_late": "late (%)",
                }
            )
        )
        + "\n"
    )

    stored = pd.read_csv(res / "forecasts.csv.gz", dtype={"as_of": object})
    stored = stored.loc[
        stored["model"].isin(ARMS)
        & (stored["kind"] == "real")
        & stored["as_of"].isna()
    ]
    latest = pd.read_csv(res / "lag-comparison" / "forecasts.csv.gz")
    samples = pd.concat([stored, latest], ignore_index=True)
    scores = pd.read_csv(res / "nowcast-scores.csv")
    scored = sorted(scores.loc[scores["model"].isin(ARMS), "origin"].unique())
    origins = [*scored, latest["vintage"].max()]
    models = [s.name for s in list_models()]
    save_light_dark(
        lambda mode: plot_lag_fans(
            samples, target, origins, models, names, mode
        ),
        fig_dir / "lag-fans.svg",
    )

    rel = relative_wis(
        scores.loc[scores["target"] == "reported"],
        by=("origin", "truth_vintage"),
    )
    rel = rel.loc[rel["model"].isin(ARMS)]
    save_light_dark(
        lambda mode: plot_lag_scores(rel, mode), fig_dir / "lag-scores.svg"
    )
    (gen_dir / "lag-scores.md").write_text(
        _table(
            rel[
                [
                    "origin",
                    "truth_vintage",
                    "model",
                    "n",
                    "rel_wis",
                    "coverage_50",
                    "coverage_90",
                ]
            ].rename(
                columns={
                    "truth_vintage": "truth release",
                    "rel_wis": "relative WIS",
                    "coverage_50": "50% coverage",
                    "coverage_90": "90% coverage",
                }
            )
        )
        + "\n"
    )

    fits = pd.concat(
        [
            pd.read_csv(res / "backtest-fits.csv"),
            pd.read_csv(res / "lag-comparison" / "fits.csv"),
        ],
        ignore_index=True,
    )
    fits = fits.loc[fits["model"].isin(ARMS) & fits["seconds"].notna()]
    conv = (
        fits.sort_values(["vintage", "model"])[
            [
                "vintage",
                "model",
                "max_rhat",
                "min_ess_bulk",
                "divergent_share",
                "seconds",
            ]
        ]
        .assign(
            divergent_share=lambda d: (100 * d["divergent_share"]).round(1),
            seconds=lambda d: (d["seconds"] / 60).round(0).astype(int),
        )
        .rename(
            columns={
                "max_rhat": "max R-hat",
                "min_ess_bulk": "min bulk ESS",
                "divergent_share": "divergent (%)",
                "seconds": "minutes",
            }
        )
    )
    (gen_dir / "lag-convergence.md").write_text(_table(conv) + "\n")

    newest = newest_weeks(pd.read_csv(res / "nowcast-quantiles.csv"))
    main = newest.loc[newest["location"].isin(PLACES)]
    save_light_dark(
        lambda mode: plot_calibration(main, mode),
        fig_dir / "lag-calibration.svg",
    )
    for stem, rows, by in (
        ("overall", newest, ()),
        ("weeks", newest, ("weeks_back",)),
        ("origin", newest, ("origin", "truth_vintage")),
        ("places", main, ("location",)),
        ("places-weeks", main, ("weeks_back",)),
    ):
        (gen_dir / f"lag-calibration-{stem}.md").write_text(
            _calibration_md(rows, by) + "\n"
        )
