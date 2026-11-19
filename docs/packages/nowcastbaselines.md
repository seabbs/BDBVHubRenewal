# nowcastbaselines

The random walk and the random walk nowcast, for weekly counts revised as later releases report more cases.
{ .page-subtitle }

Until the package split, nowcastbaselines is `bdbvhubrenewal.forecasters` (`randomwalk`, `randomwalknowcast`) with `bdbvhubrenewal.baselinenowcast` ([Packages](index.md)).

## Install

```sh
uv pip install "git+https://github.com/seabbs/bdbv-hub-modelling"
```

## Getting started

We make a small outbreak in two places, reported with a delay, and forecast its weekly counts with both baselines.

The data are daily onsets by release: each release (`vintage`) lists every onset day up to its cut-off with the cases reported by then.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(2)
days = pd.date_range("2026-01-04", "2026-04-25")
final = {
    place: rng.poisson(30 * scale * np.exp(0.01 * np.arange(days.size)))
    for place, scale in (("A-1", 1.0), ("A-2", 0.6))
}
releases = pd.date_range("2026-03-07", "2026-04-25", freq="7D")
rows = []
for release in releases:
    for place, counts in final.items():
        for i, day in enumerate(days[days <= release]):
            # Older days are more completely reported.
            age = (release - day).days
            reported = rng.binomial(counts[i], 1 - np.exp(-age / 6))
            rows.append((release, place, day, reported))
daily = pd.DataFrame(
    rows, columns=["vintage", "location", "onset_date", "observed"]
).assign(imputed=0)
for col in ("vintage", "onset_date"):
    daily[col] = daily[col].dt.date.astype(str)
daily["cutoff"] = daily["vintage"]
```

The weekly counts of each release, by week ending on Saturday, are the forecast target.

```python
onset = pd.to_datetime(daily["onset_date"])
saturday = onset + pd.to_timedelta((5 - onset.dt.weekday) % 7, unit="D")
target = (
    daily.assign(target_end_date=saturday.dt.date.astype(str))
    .groupby(["vintage", "location", "target_end_date", "cutoff"], as_index=False)[
        ["observed", "imputed"]
    ]
    .sum()
    .assign(kind="real")
)
target["total"] = target["observed"] + target["imputed"]
target["location_name"] = target["location"]
```

A task is what a model sees at one release.
Both models return 100 sample trajectories for each place and week.

```python
from nowcasteval.backtest import BacktestConfig, make_tasks
from bdbvhubrenewal.forecasters import get_model

latest = target.loc[target["vintage"] == target["vintage"].max()]
task = make_tasks(latest, None, BacktestConfig(seed=1), daily=daily)[0]
forecasts = {
    name: get_model(name).forecast(task)
    for name in ("randomwalk", "randomwalknowcast")
}
for samples in forecasts.values():
    assert list(samples.columns) == ["location", "target_end_date", "sample", "value"]
    assert samples.groupby(["location", "target_end_date"]).size().eq(100).all()
```

The random walk forecasts each place's weekly count on the log scale from its latest complete week.
The random walk nowcast first nowcasts the recent weeks, by estimating the reporting delay from how counts grew between releases, then forecasts each nowcast draw with the random walk.

## How to

### Score the baselines

The forecasts are sample tables, so nowcasteval scores them ([nowcasteval](nowcasteval.md#score-a-table-of-forecasts)).

### Run them over many releases

`make_tasks` gives one task per release in the target it is passed, so a backtest is a loop over its tasks.
`nowcasteval.backtest.run_forecasts` runs a list of models over a list of tasks and skips a model where it cannot forecast.

## Reference

[Forecasters API](../reference/forecasters.md) and [Baselinenowcast API](../reference/baselinenowcast.md).
The methods are in the paper's [supporting information](../paper/si.html#baselines).

## How it is used in the workflow

Every other model is scored relative to the random walk nowcast, and both baselines are the last models in the hub submission chain.

Next: [nowcastutils](nowcastutils.md).
