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
# This page is a notebook.
# `task getting-started` runs it again and writes the outputs into `docs/getting-started.ipynb`.
# The code calls the same library functions as the command line.
# `task setup` installs the package.
# The tidy data in `data/` are committed, so nothing needs downloading.

# %% [markdown]
# ## Load the data
#
# `load_target` reads the weekly confirmed cases by onset week of every hub vintage.
# `load_delays` reads the reporting delays the hub published.

# %%
import dataclasses
from pathlib import Path

from bdbvhubrenewal.data import delays_for_vintage, load_delays, load_target
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation.scoring import summarise_scores
from bdbvhubrenewal.forecasters import NATIONAL, get_model
from bdbvhubrenewal.reporting.plots import plot_location_grid, save_light_dark

target = load_target("data")
delays = load_delays("data")
real = target.loc[target["kind"] == "real"]
sorted(real["vintage"].unique())

# %% [markdown]
# ## Build the forecast task
#
# The origin is the vintage 2026-10-09.
# Later vintages hold the same weeks as they were reported afterwards.
# The task holds what a model sees at the origin and the weeks to forecast.
# Its reference date is the next Saturday, 2026-10-17, and the forecast weeks run from six weeks before it to three weeks after it.
# `min_horizon=-6` asks for the six weeks before the reference date as well, as the submission does.

# %%
ORIGIN = "2026-10-09"
LATER = "2026-10-24"

config = dataclasses.replace(
    bt.load_config("config/backtest.toml"), min_horizon=-6
)
origin_delays, _ = delays_for_vintage(delays, ORIGIN)
origin = real.loc[real["vintage"] == ORIGIN]
task = bt.make_tasks(origin, origin_delays, config, history=real)[0]
print(task.vintage, task.reference_date, len(task.locations), "provinces")
print(task.target_end_dates[0], "to", task.target_end_dates[-1])

# %% [markdown]
# ## Forecast
#
# `get_model` returns a model from the registry by name.
# `persistence` carries the last complete week forward with random walk noise on the log scale.
# `persistencenow` first nowcasts the recent weeks for reporting delays, then does the same.
# `persistencenow` is the baseline we submit to the hub as `seabbs-baseline`.
# `run_forecasts` runs each model, checks the samples, adds the national total and returns 100 trajectories per location.

# %%
models = [get_model("persistence"), get_model("persistencenow")]
forecasts = bt.run_forecasts(models, [task])
forecasts.groupby("model")["sample"].nunique()

# %% [markdown]
# ## Plot the nowcast
#
# Recent weeks are still being reported, so the weeks before the reference date are nowcasts.
# The first figure keeps only those weeks, to show what the nowcast adds to the data at the origin.
# Each panel shows the data at the origin (dashed), the data as later reported (solid), 100 trajectories, the 50% and 90% intervals and the median.
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
# ![Nowcasts from persistence and persistencenow at 2026-10-09 against the data as later reported.](../assets/getting-started/nowcast.svg#only-light)
# ![Nowcasts from persistence and persistencenow at 2026-10-09 against the data as later reported.](../assets/getting-started/nowcast-dark.svg#only-dark)

# %% [markdown]
# ## Plot the forecast
#
# The second figure adds the reference week and the three forecast weeks.
#
# ![Nowcasts and forecasts from persistence and persistencenow at 2026-10-09 against the data as later reported.](../assets/getting-started/forecast.svg#only-light)
# ![Nowcasts and forecasts from persistence and persistencenow at 2026-10-09 against the data as later reported.](../assets/getting-started/forecast-dark.svg#only-dark)

# %% [markdown]
# ## Score on complete weeks
#
# The truth is the 2026-10-24 vintage.
# A week counts as complete once it ended at least `complete_after_weeks` weeks before it, so most of its cases are reported.
# `score_tasks` scores every forecast sample set with the weighted interval score (WIS) and interval coverage.
# The table averages over those weeks, for the national total and over the provinces.
# Relative WIS is `persistencenow` divided by `persistence`, so values below 1 favour the nowcast.

# %%
scores = bt.score_tasks(
    forecasts,
    bt.truth_from_vintage(target, LATER),
    LATER,
    config.complete_after_weeks,
)
complete = scores.loc[scores["complete"]]
complete = complete.assign(
    level=complete["location"].map(
        lambda x: "National" if x == NATIONAL else "Province"
    )
)
by_level = summarise_scores(complete, by=["level", "model"])
table = by_level.pivot(
    index="level", columns="model", values=["wis", "coverage_90"]
)
table[("wis", "relative")] = (
    table[("wis", "persistencenow")] / table[("wis", "persistence")]
)
table.round(2)

# %% [markdown]
# ## Next steps
#
# The [CLI page](../development/cli/) lists the commands the submission workflow runs.
# They fetch the data, run the backtest, forecast from the latest vintage and write the hub file.
#
# The renewal model is not fitted here.
# It needs a sampler and is described on the [Renewal models page](../methods/model/).
