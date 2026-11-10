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
# Forecast with two baselines, load the renewal model's stored fit at the same origin, then plot and score all three.
#
# Every figure and table below comes from the Python code shown above it, which calls the package's functions directly.
# [Setup](../setup/) installs the package and shows how to run this code in a REPL, a notebook or a script.
# The tidy data in `data/` are committed, and `task fit-store` fetches the stored renewal fit.
# This page is a notebook, and `task getting-started` runs it again.

# %% [markdown]
# ## Load the data
#
# `load_target` reads the weekly confirmed cases by onset week of every hub vintage.
# `load_daily` reads the daily onsets of every hub release, which the baselines need.
# `load_delays` reads the reporting delays the hub published, which the renewal models use.

# %%
import json
from pathlib import Path

import pandas as pd

from bdbvhubrenewal.data import load_daily, load_delays, load_target
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation import store
from bdbvhubrenewal.evaluation.nowcast import relative_wis, release_backtest
from bdbvhubrenewal.evaluation.scoring import relative_skill
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
# The origin is the second newest release.
# The newest release scores it.
# The task holds what a model sees at the origin, including the daily onsets of every release up to it, and the weeks to forecast.
# Its reference date is the Saturday after the origin, and the forecast weeks run from four weeks before it to three weeks after it.
# `origin_tasks` builds the task of every real origin as the fit store fits it, so the baselines and the stored renewal fit see the same task.

# %%
ORIGIN, LATER = bt.real_vintages(target)[-2:]

config = bt.load_config("config/backtest.toml")
task = store.origin_tasks(target, delays, config, daily)[ORIGIN]
print(task.vintage, task.reference_date, len(task.locations), "provinces")
print(task.target_end_dates[0], "to", task.target_end_dates[-1])
print("scored on", LATER)

# %% [markdown]
# ## Forecast with the baselines
#
# `get_model` returns a model from the registry by name.
# `randomwalk` carries the last complete week forward with random walk noise on the log scale.
# `randomwalknowcast` nowcasts the recent weeks first ([Baseline models](../methods/baselines/)).
# `randomwalknowcast` is the baseline we submit to the hub as `seabbs-rwnowcast`.
# `run_forecasts` runs each model, checks the samples, adds the national total and returns 100 trajectories per location.
# `learnt` collects each model's modelled delay, which scoring uses.

# %%
models = [get_model("randomwalk"), get_model("randomwalknowcast")]
learnt: list[pd.DataFrame] = []
baselines = bt.run_forecasts(models, [task], learnt=learnt)
baselines.groupby("model")["sample"].nunique()

# %% [markdown]
# ## Fit the renewal model
#
# The [renewal model](../methods/model/) is fitted by MCMC (NUTS), which takes about half an hour, so this page does not fit it.
# The [fit store](../development/cli/#fit-store) fits it in CI at every real origin and stores the result.
# This is how it is fitted, with the sampler settings of the submission:
#
# ```python
# fitted = store.fit(
#     "renewal", task, store.samplers(), store.code_digest("renewal")
# )
# store.write_fit("output/fit-store", fitted)
# ```
#
# The page loads the stored fit instead.

# %% [markdown]
# ## Load the stored renewal fit
#
# `bdbvhubrenewal fit-store load` (`task fit-store`) fetches the store and records the fit it uses for each model and origin in `results/fit-store.json`.
# That is the fit for this code and data, or, until that fit is made, the newest fit of older code.
# Each stored fit is a gzipped JSON file, which `read_fit` reads into tables.
# Its `forecasts` table has the same columns as the baselines' forecasts, and its `delays` table holds the fit's modelled delay.

# %%
status = json.loads(Path("results/fit-store.json").read_text())
entry = next(
    s for s in status if s["model"] == "renewal" and s["origin"] == ORIGIN
)
assert entry["used"], f"The fit store has no renewal fit at {ORIGIN}."
path = Path("output/fit-store/renewal") / ORIGIN / f"{entry['used']}.json.gz"
stored = store.read_fit(path)
renewal = stored.forecasts
print("fitted at commit", stored.meta["commit"][:7])
print("fit of this code" if entry["current"] else "fit of older code")
forecasts = pd.concat([baselines, renewal], ignore_index=True)
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
# ![Nowcasts from randomwalk, randomwalknowcast and renewal at the origin against the data as later reported.](../assets/getting-started/nowcast.svg#only-light)
# ![Nowcasts from randomwalk, randomwalknowcast and renewal at the origin against the data as later reported.](../assets/getting-started/nowcast-dark.svg#only-dark)

# %% [markdown]
# ## Plot the forecast
#
# The second figure adds the reference week and the three forecast weeks.
#
# ![Nowcasts and forecasts from randomwalk, randomwalknowcast and renewal at the origin against the data as later reported.](../assets/getting-started/forecast.svg#only-light)
# ![Nowcasts and forecasts from randomwalk, randomwalknowcast and renewal at the origin against the data as later reported.](../assets/getting-started/forecast-dark.svg#only-dark)

# %% [markdown]
# ## Score
#
# Each forecast is scored on what each later release reported ([how scoring works](../methods/evaluation/#real-releases)).
# `randomwalk` is scored on its samples as they are, with no delay.
# `release_backtest` maps each other model's forecast of final counts to the counts a later release would report, using the model's own modelled delay.
# It scores the samples with the weighted interval score (WIS) and interval coverage.
# The table gives each model's mean WIS relative to `randomwalk`, for the national total and over the provinces.

# %%
result = release_backtest(
    [task],
    daily,
    target,
    models=(),
    stored=forecasts,
    delays=pd.concat([*learnt, stored.delays], ignore_index=True),
)
scores = result.scores.loc[result.scores["target"] == "reported"]
scores = scores.assign(
    level=scores["location"].map(
        lambda x: "National" if x == NATIONAL else "Province"
    )
)
relative_wis(scores, by=["level"]).round(2)

# %% [markdown]
# The [energy score](../methods/evaluation/#energy-score) scores each scale's forecast of every place and week at once, from the same samples.
# The table gives it and its log version relative to `randomwalk`, with $n$ the releases scored.

# %%
energy = result.energy.loc[result.energy["target"] == "reported"]
relative_skill(
    energy,
    by=("scale",),
    metrics=("energy", "log_energy"),
    match_on=("origin", "truth_vintage"),
).round(2)

# %% [markdown]
# ## Next steps
#
# The [API reference](../reference/) documents every function used here, grouped by task.
# The [CLI page](../development/cli/) lists the commands the submission workflow runs.
# They fetch the data, run the backtest, forecast from the latest vintage and write the hub file.
