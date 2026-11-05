# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Figures and tables of the Reporting model comparison page.

The page compares the renewal models before and after the learnt
publication lag and the partially pooled province delays, with two arms
that change one part each. Their fits were made once on a branch that
registers the arms, and their scores and summaries are in
`results/reporting-comparison/` (`COMPARISON_FILES`). The page reads only
those files.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from bdbvhubrenewal.evaluation.nowcast import relative_wis
from bdbvhubrenewal.evaluation.scoring import relative_skill

from .plots import HUB_GREYS, MODEL_COLOURS, save_light_dark, theme

__all__ = [
    "ARMS",
    "COMPARISON_FILES",
    "make_reporting_comparison",
    "newest_places",
    "newest_weeks",
    "plot_fans",
    "plot_learnt_lag",
    "plot_province_delays",
]

ARMS: Final[dict[str, str]] = {
    "renewal-before": "renewal before",
    "renewalfixedlag": "fixed lag, pooled delays",
    "renewalshareddelay": "learnt lag, shared delay",
    "renewal": "renewal after",
    "renewalsp-before": "renewalsp before",
    "renewalsp": "renewalsp after",
    "renewalbvd-before": "renewalbvd before",
    "renewalbvd": "renewalbvd after",
}
"""Each fitted model's label. `-before` fits are of the earlier code."""

_COLOURS: Final[dict[str, int]] = {
    "renewal-before": 0,
    "renewalfixedlag": 1,
    "renewalshareddelay": 6,
    "renewal": 2,
    "renewalsp-before": 4,
    "renewalsp": 7,
    "renewalbvd-before": 3,
    "renewalbvd": 9,
}
"""Each model's place in `MODEL_COLOURS` on this page."""

BASELINES: Final = ("randomwalk", "randomwalknowcast")
"""Models scored beside the fits."""

PLACES: Final = ("CD", "CD-IT", "CD-NK")
"""Places in the fans and the place scores."""

ZONES: Final = ("CD-NK-20", "CD-NK-03")
"""Zones in the newest-week table: Lubero, and Beni, where North Kivu's
reporting slowed release by release."""

FIXED_LAG: Final = 3
"""The earlier lag was uniform on 0 to this many days."""

NEWEST: Final = (0, 3)
"""Weeks back from the reference week counted as the newest."""

COMPARISON_FILES: Final = (
    "scores.csv",
    "quantiles.csv",
    "zone-scores.csv",
    "zone-forecasts.csv.gz",
    "fits.csv",
    "delays.csv",
    "learnt-lag.csv",
)
"""Files in `results/reporting-comparison/`."""


def _colour(model: str, mode: str) -> str:
    return MODEL_COLOURS[mode][_COLOURS[model]]


def _table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".2f")


def plot_learnt_lag(table: pd.DataFrame, mode: str = "light") -> Figure:
    """Plot the learnt lag at each origin against the earlier fixed lag.

    Parameters
    ----------
    table : pandas.DataFrame
        `model`, `vintage`, `lag` and the lag pmf's `q5`, `q50` and
        `q95`, for `renewal`.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rows = table.loc[table["model"] == "renewal"]
    with theme(mode):
        fig, ax = plt.subplots(figsize=(7, 3.4))
        greys = HUB_GREYS[mode]
        origins = sorted(rows["vintage"].unique())
        for i, origin in enumerate(origins):
            one = rows.loc[rows["vintage"] == origin].sort_values("lag")
            last = i == len(origins) - 1
            colour = _colour("renewal", mode) if last else greys[i]
            ax.fill_between(
                one["lag"], one["q5"], one["q95"], color=colour,
                alpha=0.25 if last else 0.12, lw=0, step="mid",
            )  # fmt: skip
            ax.step(
                one["lag"], one["q50"], where="mid", color=colour,
                lw=1.8 if last else 1.0, label=f"Learnt at {origin}",
            )  # fmt: skip
        lags = np.arange(int(rows["lag"].max()) + 1)
        fixed = np.where(lags <= FIXED_LAG, 1 / (FIXED_LAG + 1), 0.0)
        ax.step(
            lags, fixed, where="mid", color=_colour("renewal-before", mode),
            lw=1.4, ls="--", label=f"Earlier fixed lag, 0 to {FIXED_LAG}",
        )  # fmt: skip
        ax.set_xlabel("Days from report to publication in a release")
        ax.set_ylabel("Probability")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
    return fig


def plot_province_delays(
    delays: pd.DataFrame, names: dict[str, str], mode: str = "light"
) -> Figure:
    """Plot each province's delay over onset weeks against the national.

    Parameters
    ----------
    delays : pandas.DataFrame
        `delays.csv` rows of `renewal` at one origin, `kind`
        `"onset_week"` and `quantity` `"share_within_7"`.
    names : dict
        Location code to name.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rows = delays.copy()
    rows["x"] = pd.to_datetime(rows["x"])
    with theme(mode) as ink:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        places = sorted(set(rows["location"]) - {"CD"})
        for place in places:
            one = rows.loc[rows["location"] == place].sort_values("x")
            main = place in PLACES
            colour = (
                MODEL_COLOURS[mode][PLACES.index(place)]
                if main
                else HUB_GREYS[mode][2]
            )
            if main:
                ax.fill_between(
                    one["x"], one["q5"], one["q95"], color=colour,
                    alpha=0.2, lw=0,
                )  # fmt: skip
            ax.plot(
                one["x"], one["q50"], color=colour, lw=1.6 if main else 0.8,
                label=names.get(place, place) if main else None,
            )  # fmt: skip
        nat = rows.loc[rows["location"] == "CD"].sort_values("x")
        ax.fill_between(
            nat["x"], nat["q5"], nat["q95"], color=ink["data"], alpha=0.12,
            lw=0,
        )  # fmt: skip
        ax.plot(nat["x"], nat["q50"], color=ink["data"], lw=2, label="National")
        ax.plot(
            [], [], color=HUB_GREYS[mode][2], lw=0.8, label="Other provinces"
        )
        ax.set_ylabel("Share reported within 7 days")
        ax.set_xlabel("Onset week ending")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelrotation=30)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
    return fig


def plot_fans(
    quantiles: pd.DataFrame,
    target: pd.DataFrame,
    models: Sequence[str],
    names: dict[str, str],
    mode: str = "light",
) -> Figure:
    """Plot models' nowcasts at each origin with every later release.

    Parameters
    ----------
    quantiles : pandas.DataFrame
        Final-count quantiles (`model`, `vintage`, `location`,
        `target_end_date`, `q5` to `q95`).
    target : pandas.DataFrame
        The tidy target table.
    models : sequence of str
        Models shown, each in its colour.
    names : dict
        Location code to name.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    real = target.loc[target["kind"] == "real"]
    origins = sorted(quantiles["vintage"].unique())
    with theme(mode) as ink:
        fig, axes = plt.subplots(
            len(PLACES), len(origins), figsize=(3.4 * len(origins), 8),
            squeeze=False,
        )  # fmt: skip
        greys = HUB_GREYS[mode]
        for col, origin in enumerate(origins):
            later = sorted(v for v in real["vintage"].unique() if v >= origin)
            for row, place in enumerate(PLACES):
                ax = axes[row, col]
                cell = quantiles.loc[
                    (quantiles["vintage"] == origin)
                    & (quantiles["location"] == place)
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
                for model in models:
                    fan = cell.loc[cell["model"] == model].sort_values(
                        "target_end_date"
                    )
                    t = pd.to_datetime(fan["target_end_date"])
                    colour = _colour(model, mode)
                    ax.fill_between(
                        t, fan["q5"], fan["q95"], color=colour, alpha=0.18,
                        lw=0,
                    )  # fmt: skip
                    ax.plot(
                        t, fan["q50"], color=colour, lw=1.6,
                        label=ARMS[model],
                    )  # fmt: skip
                ax.axvline(
                    pd.Timestamp(origin), color=ink["muted"], lw=0.8, ls=":"
                )
                ax.set_ylim(bottom=0)
                ax.tick_params(axis="x", labelrotation=45, labelsize=7)
                if row == 0:
                    ax.set_title(f"Origin {origin}", fontsize=9)
                if col == 0:
                    ax.set_ylabel(names.get(place, place))
        handles, labels = axes[-1, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncol=4, frameon=False,
            fontsize=8,
        )  # fmt: skip
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def newest_weeks(quantiles: pd.DataFrame) -> pd.DataFrame:
    """Coverage and width of the newest weeks, against later releases.

    Parameters
    ----------
    quantiles : pandas.DataFrame
        `quantiles.csv` of the release backtest, with `reported` (the
        count at the origin) and `truth` (a later release's count).

    Returns
    -------
    pandas.DataFrame
        By model, the number of nowcasts `n` of weeks 0 to 3 back in
        `PLACES` on the `reported` target, the 50% and 90% coverage, the
        share above the 90% interval and the median 90% width over the
        median.
    """
    q = quantiles.loc[
        (quantiles["target"] == "reported")
        & quantiles["weeks_back"].between(*NEWEST)
        & quantiles["location"].isin(PLACES)
    ]
    q = q.assign(
        in_50=q["truth"].between(q["q25"], q["q75"]),
        in_90=q["truth"].between(q["q5"], q["q95"]),
        above_90=q["truth"] > q["q95"],
        width_90=(q["q95"] - q["q5"]) / q["q50"].clip(lower=1),
    )
    return (
        q.groupby("model")
        .agg(
            n=("in_90", "size"),
            coverage_50=("in_50", "mean"),
            coverage_90=("in_90", "mean"),
            above_90=("above_90", "mean"),
            width_90=("width_90", "median"),
        )
        .reset_index()
    )


def newest_places(
    forecasts: pd.DataFrame,
    zones: pd.DataFrame,
    target: pd.DataFrame,
    names: dict[str, str],
) -> pd.DataFrame:
    """Nowcasts of North Kivu and `ZONES` before and after, with releases.

    Parameters
    ----------
    forecasts, zones : pandas.DataFrame
        Final-count quantiles of provinces and of zones (`q5`, `q50`,
        `q95`) by `model`, `vintage`, `location` and `target_end_date`.
    target : pandas.DataFrame
        The tidy target table, for North Kivu's later counts.
    names : dict
        Location code to name.

    Returns
    -------
    pandas.DataFrame
        One row per origin, place and week 0 to 3 back: the median and
        90% interval of `renewal` before and after, and the latest
        count of each place, at the latest release.
    """
    both = pd.concat([forecasts, zones], ignore_index=True)
    rows = both.loc[
        both["location"].isin(("CD-NK", *ZONES))
        & both["model"].isin(("renewal-before", "renewal"))
    ].copy()
    rows["weeks_back"] = (
        pd.to_datetime(rows["vintage"])
        - pd.to_datetime(rows["target_end_date"])
    ).dt.days // 7
    rows = rows.loc[rows["weeks_back"].between(*NEWEST)]
    rows["value"] = [
        f"{m:.0f} ({lo:.0f} to {hi:.0f})"
        for m, lo, hi in rows[["q50", "q5", "q95"]].to_numpy()
    ]
    wide = rows.pivot_table(
        index=["vintage", "location", "target_end_date"],
        columns="model",
        values="value",
        aggfunc="first",
    ).reset_index()
    real = target.loc[target["kind"] == "real"]
    latest = real.loc[real["vintage"] == real["vintage"].max()]
    counts = latest.set_index(["location", "target_end_date"])["total"]
    zone_counts = _zone_counts()
    wide["latest count"] = [
        counts.get((loc, week), zone_counts.get((loc, week), 0))
        for loc, week in wide[["location", "target_end_date"]].to_numpy()
    ]
    wide["location"] = wide["location"].map(lambda c: names.get(c, c))
    return wide.rename(
        columns={
            "vintage": "origin",
            "location": "place",
            "target_end_date": "week ending",
            "renewal-before": ARMS["renewal-before"],
            "renewal": ARMS["renewal"],
        }
    )


def _zone_counts() -> pd.Series:
    """Zone counts at the latest release, by zone and week."""
    from bdbvhubrenewal.data.tidy import load_daily_zones
    from bdbvhubrenewal.data.zones import weekly_zone_counts

    daily = load_daily_zones()
    counts = weekly_zone_counts(daily, daily["vintage"].max())
    return counts.set_index(["location", "target_end_date"])["total"]


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    shown = [*ARMS, *BASELINES]
    out = df.loc[df["model"].isin(shown)].copy()
    out["order"] = out["model"].map({m: i for i, m in enumerate(shown)})
    out["model"] = out["model"].map(lambda m: ARMS.get(m, m))
    return out.sort_values([*[c for c in out if c == "scale"], "order"]).drop(
        columns="order"
    )


def make_reporting_comparison(
    fig_dir: Path,
    gen_dir: Path,
    res: Path,
    target: pd.DataFrame,
    names: dict[str, str],
) -> None:
    """Write the figures and tables of the Reporting model comparison page.

    Parameters
    ----------
    fig_dir, gen_dir : pathlib.Path
        Where SVGs and Markdown snippets go.
    res : pathlib.Path
        `results/reporting-comparison/` (`COMPARISON_FILES`).
    target : pandas.DataFrame
        The tidy target table.
    names : dict
        Location code to name.
    """
    lag = pd.read_csv(res / "learnt-lag.csv")
    save_light_dark(
        lambda mode: plot_learnt_lag(lag, mode), fig_dir / "rc-lag.svg"
    )
    delays = pd.read_csv(res / "delays.csv")
    latest = delays["vintage"].max()
    shown = delays.loc[
        (delays["model"] == "renewal")
        & (delays["vintage"] == latest)
        & (delays["quantity"] == "share_within_7")
    ]
    save_light_dark(
        lambda mode: plot_province_delays(shown, names, mode),
        fig_dir / "rc-delays.svg",
    )

    forecasts = pd.read_csv(res / "forecasts.csv.gz")
    for stem, models in (
        ("renewal", ("renewal-before", "renewal")),
        ("arms", ("renewalfixedlag", "renewalshareddelay")),
        ("renewalsp", ("renewalsp-before", "renewalsp")),
        ("renewalbvd", ("renewalbvd-before", "renewalbvd")),
    ):
        save_light_dark(
            lambda mode, models=models: plot_fans(
                forecasts.loc[forecasts["model"].isin(models)],
                target,
                models,
                names,
                mode,
            ),
            fig_dir / f"rc-fans-{stem}.svg",
        )

    scores = pd.read_csv(res / "scores.csv")
    reported = scores.loc[scores["target"] == "reported"]
    scale = reported.assign(
        scale=np.where(reported["location"] == "CD", "national", "province")
    )
    zones = pd.read_csv(res / "zone-scores.csv")
    zone_rel = relative_skill(
        zones,
        "randomwalk",
        match_on=("origin", "truth_vintage", "location", "target_end_date"),
        means=("wis", "log_wis", "coverage_90", "bias"),
    ).rename(
        columns={"relative_wis": "rel_wis", "relative_log_wis": "rel_log_wis"}
    )
    cols = ["model", "n", "rel_wis", "rel_log_wis", "bias", "coverage_90"]
    header = {
        "rel_wis": "relative WIS",
        "rel_log_wis": "relative WIS, log scale",
        "coverage_90": "90% coverage",
        "truth_vintage": "truth release",
    }
    by_scale = pd.concat(
        [
            relative_wis(scale, by=("scale",)),
            zone_rel.assign(scale="zone"),
        ],
        ignore_index=True,
    )
    tables = {
        "rc-scores-scale.md": _labelled(by_scale)[["scale", *cols]],
        "rc-scores-release.md": _labelled(
            relative_wis(reported, by=("origin", "truth_vintage"))
        )[["origin", "truth_vintage", *cols]],
        "rc-scores-places.md": _labelled(
            relative_wis(
                reported.loc[reported["location"].isin(PLACES)],
                by=("location",),
            ).replace({"location": names})
        )[["location", *cols]],
        "rc-scores-complete.md": _labelled(
            relative_wis(scores.loc[scores["target"] == "complete"])
        )[cols],
        "rc-newest.md": _labelled(
            newest_weeks(pd.read_csv(res / "quantiles.csv"))
        ),
    }
    for name, table in tables.items():
        (gen_dir / name).write_text(_table(table.rename(columns=header)) + "\n")

    (gen_dir / "rc-newest-places.md").write_text(
        _table(
            newest_places(
                forecasts,
                pd.read_csv(res / "zone-forecasts.csv.gz"),
                target,
                names,
            )
        )
        + "\n"
    )
    latest_fc = forecasts.loc[
        (forecasts["vintage"] == forecasts["vintage"].max())
        & (forecasts["location"] == "CD-IT")
        & forecasts["model"].isin(ARMS)
    ]
    ituri = latest_fc.assign(
        model=latest_fc["model"].map(ARMS),
        forecast=[
            f"{m:.0f} ({lo:.0f} to {hi:.0f})"
            for m, lo, hi in latest_fc[["q50", "q5", "q95"]].to_numpy()
        ],
    ).pivot(index="target_end_date", columns="model", values="forecast")
    (gen_dir / "rc-ituri.md").write_text(
        ituri.reset_index()
        .rename(columns={"target_end_date": "week ending"})
        .to_markdown(index=False)
        + "\n"
    )

    fits = pd.read_csv(res / "fits.csv")
    conv = fits.loc[fits["model"].isin(ARMS)].assign(
        model=lambda d: d["model"].map(ARMS),
        divergent=lambda d: (100 * d["divergent_share"]).round(1),
        minutes=lambda d: (d["seconds"] / 60).round(0).astype(int),
    )
    (gen_dir / "rc-convergence.md").write_text(
        _table(
            conv[
                [
                    "vintage",
                    "model",
                    "fit",
                    "max_rhat",
                    "min_ess_bulk",
                    "divergent",
                    "minutes",
                ]
            ].rename(
                columns={
                    "max_rhat": "max R-hat",
                    "min_ess_bulk": "min bulk ESS",
                    "divergent": "divergent (%)",
                }
            )
        )
        + "\n"
    )
