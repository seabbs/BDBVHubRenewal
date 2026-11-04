# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Getting started
#
# Forecast with two baselines, plot them and score them.
#
# Every figure and table below comes from the Python code shown above it, which calls the package's functions directly.
# [Setup](../setup/) installs the package and shows how to run this code in a REPL, a notebook or a script.
# The tidy data in `data/` are committed, so nothing needs downloading.
# This page is a notebook, and `task getting-started` runs it again.

# %% [markdown]
# ## Load the data
#
# `load_target` reads the weekly confirmed cases by onset week of every hub vintage.
# `load_daily` reads the daily onsets of every hub release, which the baselines need.
# `load_delays` reads the reporting delays the hub published, which the renewal models use.

# %%
import dataclasses
from pathlib import Path

from bdbvhubrenewal.data import (
    delays_for_vintage,
    load_daily,
    load_delays,
    load_target,
)
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation.nowcast import relative_wis, release_backtest
from bdbvhubrenewal.forecasters import NATIONAL, get_model
from bdbvhubrenewal.reporting.plots import plot_location_grid, save_light_dark

target = load_target("data")
daily = load_daily("data")
delays = load_delays("data")
real = target.loc[target["kind"] == "real"]
sorted(real["vintage"].unique())

# %% [markdown]
# ## Build the forecast task
#
# The origin is the vintage 2026-10-17.
# The latest vintage, 2026-11-01, holds the same weeks as they were reported afterwards.
# The task holds what a model sees at the origin, including the daily onsets of every release up to it, and the weeks to forecast.
# Its reference date is the next Saturday, 2026-10-24, and the forecast weeks run from six weeks before it to three weeks after it.
# `min_horizon=-6` asks for the six weeks before the reference date as well, as the submission does.

# %%
ORIGIN = "2026-10-17"
LATER = "2026-11-01"

config = dataclasses.replace(
    bt.load_config("config/backtest.toml"), min_horizon=-6
)
origin_delays, _ = delays_for_vintage(delays, ORIGIN)
origin = real.loc[real["vintage"] == ORIGIN]
task = bt.make_tasks(
    origin, origin_delays, config, history=real, daily=daily
)[0]
print(task.vintage, task.reference_date, len(task.locations), "provinces")
print(task.target_end_dates[0], "to", task.target_end_dates[-1])

# %% [markdown]
# ## Forecast
#
# `get_model` returns a model from the registry by name.
# `randomwalk` carries the last complete week forward with random walk noise on the log scale.
# `randomwalknowcast` nowcasts the recent weeks first ([Baseline models](../methods/baselines/)).
# `randomwalknowcast` is the baseline we submit to the hub as `seabbs-rwnowcast`.
# `run_forecasts` runs each model, checks the samples, adds the national total and returns 100 trajectories per location.

# %%
models = [get_model("randomwalk"), get_model("randomwalknowcast")]
forecasts = bt.run_forecasts(models, [task])
forecasts.groupby("model")["sample"].nunique()

# %% [markdown]
# ## Plot the nowcast
#
# Recent weeks are still being reported, so the weeks before the reference date are nowcasts.
# The first figure keeps only those weeks, to show what the nowcast adds to the data at the origin.
# Each panel shows the data at the origin and as later reported, with the forecast fans ([reading the figures](../methods/evaluation/#reading-the-figures)).
# The panels are the national total and the three provinces with the most cases.
# `save_light_dark` writes a light and a dark version of each figure.

# %%
LATEST = real.loc[real["vintage"] == LATER]
provinces = (
    LATEST.loc[LATEST["location"] != NATIONAL]
    .groupby("location")["total"]
    .sum()
    .nlargest(3)
    .index
)
panels = [NATIONAL, *provinces]
OUT = Path("docs/assets/getting-started")


def save(stem, **kwargs):
    save_light_dark(
        lambda mode: plot_location_grid(
            forecasts,
            target,
            panels,
            truth_vintage=LATER,
            weeks_before=12,
            mode=mode,
            **kwargs,
        ),
        OUT / f"{stem}.svg",
    )


save("nowcast", max_horizon=-1)
save("forecast")
panels

# %% [markdown]
# ![Nowcasts from randomwalk and randomwalknowcast at 2026-10-17 against the data as later reported.](../assets/getting-started/nowcast.svg#only-light)
# ![Nowcasts from randomwalk and randomwalknowcast at 2026-10-17 against the data as later reported.](../assets/getting-started/nowcast-dark.svg#only-dark)

# %% [markdown]
# ## Plot the forecast
#
# The second figure adds the reference week and the three forecast weeks.
#
# ![Nowcasts and forecasts from randomwalk and randomwalknowcast at 2026-10-17 against the data as later reported.](../assets/getting-started/forecast.svg#only-light)
# ![Nowcasts and forecasts from randomwalk and randomwalknowcast at 2026-10-17 against the data as later reported.](../assets/getting-started/forecast-dark.svg#only-dark)

# %% [markdown]
# ## Score
#
# Each forecast is scored on what each later release reported ([how scoring works](../methods/evaluation/#real-releases)).
# `release_backtest` maps the forecast of final counts to the counts a later release would report, using the reporting delays fitted at the origin.
# It scores the mapped samples with the weighted interval score (WIS) and interval coverage.
# The table gives each model's mean WIS relative to `randomwalk`, for the national total and over the provinces.
# `as reported` uses the data at the origin as the forecast.

# %%
scores = release_backtest(
    [task], daily, target, models=(), stored=forecasts
).scores
scores = scores.loc[scores["target"] == "reported"]
scores = scores.assign(
    level=scores["location"].map(
        lambda x: "National" if x == NATIONAL else "Province"
    )
)
relative_wis(scores, by=["level"]).round(2)

# %% [markdown]
# ## Next steps
#
# The [API reference](../reference/) documents every function used here, grouped by task.
# The [CLI page](../development/cli/) lists the commands the submission workflow runs.
# They fetch the data, run the backtest, forecast from the latest vintage and write the hub file.
#
# The renewal model is not fitted here.
# It needs a sampler and is described on the [Renewal models page](../methods/model/).
