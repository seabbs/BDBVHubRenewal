# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Weekly cross-check of the daily nowcast.

The latest two releases whose cut-offs are one week apart give one week
of growth for every hub week.
The port's chain ladder (`pair_sums`, `pair_factors`, `chain_growth`)
runs on those hub weeks, with national factors for every place and a
step of one week.
Weeks up to six old at the earlier release are grown to the 42 day window.
Two releases give no right-truncated incremental triangle, so
`estimate_delay` itself cannot run.
The points are drawn over the `randomwalknowcast` bands of the later
of the two releases.
This is a check and not a model in the package.
`scripts/weekly_crosscheck.py` runs it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bdbvhubrenewal.baselinenowcast import (
    chain_growth,
    exact_plan,
    pair_factors,
    pair_sums,
)
from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.forecasters.base import add_national, get_model
from bdbvhubrenewal.reporting.nowcast_plots import plot_weekly_crosscheck
from bdbvhubrenewal.reporting.plots import save_light_dark

DELAY_WEEKS = 6
PLACES = ("CD", "CD-IT", "CD-NK", "CD-HU", "CD-TO")


def weekly_pair(daily: pd.DataFrame) -> tuple[str, str, str, str]:
    """Return the latest two releases whose cut-offs are 7 days apart.

    Gives the two releases, then their cut-off days.
    """
    cut = pd.to_datetime(daily["onset_date"]).groupby(daily["vintage"]).max()
    cut = cut.sort_index()
    for i in range(len(cut) - 1, 0, -1):
        if (cut.iloc[i] - cut.iloc[i - 1]).days == 7:
            return (
                str(cut.index[i - 1]),
                str(cut.index[i]),
                cut.iloc[i - 1].date().isoformat(),
                cut.iloc[i].date().isoformat(),
            )
    raise ValueError("No two releases have cut-offs one week apart.")


def weekly_points(
    real: pd.DataFrame,
    earlier: str,
    later: str,
    cut_earlier: str,
    cut_later: str,
) -> pd.DataFrame:
    """Grow the latest weeks of the later release with the weekly ladder."""
    wide = real.loc[real["vintage"].isin([earlier, later])].pivot_table(
        index=["location", "target_end_date"],
        columns="vintage",
        values="total",
        fill_value=0.0,
    )
    places = sorted(wide.index.get_level_values("location").unique())
    weeks = sorted(wide.index.get_level_values("target_end_date").unique())
    counts = np.zeros((len(places), 2, len(weeks)))
    for key, row in wide.iterrows():
        loc, week = str(key[0]), str(key[1])  # type: ignore[index]
        counts[places.index(loc), :, weeks.index(week)] = [
            row[earlier],
            row[later],
        ]
    cut = np.array([weeks.index(cut_earlier), weeks.index(cut_later)])
    observed = np.arange(len(weeks))[None, :] <= cut[:, None]
    counts = counts * observed[None]
    num, den = pair_sums(counts, observed, cut, np.array([0]), DELAY_WEEKS)
    national = places.index("CD")
    factors, _ = pair_factors(
        num[national : national + 1], den[national : national + 1],
        np.array([0]), 1,
    )  # fmt: skip
    ages = np.arange(DELAY_WEEKS)
    growth = np.asarray(
        chain_growth(factors, exact_plan(ages, 1, 0, DELAY_WEEKS))
    )[0]
    out = []
    for loc in PLACES:
        for age in range(4):
            week = weeks[cut[1] - age]
            seen = counts[places.index(loc), 1, cut[1] - age]
            out.append(
                {
                    "location": loc,
                    "target_end_date": week,
                    "weeks_back": age,
                    "reported": seen,
                    "point": seen * growth[age],
                }
            )
    return pd.DataFrame(out)


def make_crosscheck(
    data_dir: Path | str = "data",
    config_path: Path | str = "config/backtest.toml",
    figures: Path | str = "docs/figures",
    results: Path | str = "results",
) -> pd.DataFrame:
    """Write the cross-check points and figure.

    Parameters
    ----------
    data_dir : pathlib.Path or str
        The tidy data.
    config_path : pathlib.Path or str
        Backtest settings.
    figures : pathlib.Path or str
        Where the figure goes.
    results : pathlib.Path or str
        Where the points go.

    Returns
    -------
    pandas.DataFrame
        The points, columns `location`, `target_end_date`, `weeks_back`,
        `reported` and `point`.
    """
    config = bt.load_config(config_path)
    target, delays, daily = (
        load_target(data_dir),
        load_delays(data_dir),
        load_daily(data_dir),
    )
    real = target.loc[target["kind"] == "real"]
    earlier, later, cut_earlier, cut_later = weekly_pair(daily)
    weekly = weekly_points(real, earlier, later, cut_earlier, cut_later)
    task = bt.make_tasks(
        real.loc[real["vintage"] == later],
        delays.assign(kind="real"),
        config,
        daily=daily,
    )[0]
    samples = add_national(get_model("randomwalknowcast").forecast(task))
    weeks = set(weekly["target_end_date"])
    rows = samples.loc[samples["target_end_date"].isin(weeks)]
    q = (
        rows.groupby(["location", "target_end_date"])["value"]
        .quantile(np.array([0.05, 0.25, 0.5, 0.75, 0.95]))
        .unstack()
        .rename(columns=lambda p: f"q{round(100 * p)}")
        .reset_index()
    )
    Path(results).mkdir(exist_ok=True)
    weekly.to_csv(Path(results) / "weekly-crosscheck.csv", index=False)
    save_light_dark(
        lambda mode: plot_weekly_crosscheck(q, weekly, real, PLACES, mode=mode),
        Path(figures) / "weekly-crosscheck.svg",
    )
    return weekly
