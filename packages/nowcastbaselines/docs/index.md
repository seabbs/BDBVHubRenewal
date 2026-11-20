# nowcastbaselines

The random walk and the random walk nowcast, for weekly counts revised as later releases report more cases.
{ .page-subtitle }

nowcastbaselines is the package in `packages/nowcastbaselines` of the repository, and builds on [nowcastutils](https://samabbott.co.uk/bdbv-hub-modelling/nowcastutils/) ([Packages](https://samabbott.co.uk/bdbv-hub-modelling/packages/)).

## Install

```sh
uv pip install "git+https://github.com/seabbs/bdbv-hub-modelling#subdirectory=packages/nowcastbaselines"
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
    .groupby(
        ["vintage", "location", "target_end_date", "cutoff"], as_index=False
    )[["observed", "imputed"]]
    .sum()
    .assign(kind="real")
)
target["total"] = target["observed"] + target["imputed"]
target["location_name"] = target["location"]
```

A task is what a model sees at one release: its data, the places and the weeks to forecast.
The weeks run from four weeks before the release's reference week, which the data have not yet completed, to three weeks after.
Importing `nowcastbaselines.randomwalk` adds both models to the registry.
Each returns 100 sample trajectories for each place and week.

```python
import datetime as dt

import nowcastbaselines.randomwalk  # noqa: F401
from nowcastutils.base import ForecastTask, get_model
from nowcastutils.dates import reference_date_for_vintage

vintage = daily["vintage"].max()
reference = reference_date_for_vintage(vintage)
task = ForecastTask(
    vintage=vintage,
    reference_date=reference,
    kind="real",
    cutoff=vintage,
    target=target.loc[target["vintage"] == vintage].reset_index(drop=True),
    delays=None,
    locations=("A-1", "A-2"),
    target_end_dates=tuple(
        (reference + dt.timedelta(weeks=h)).isoformat() for h in range(-4, 4)
    ),
    seed=1,
    daily_history=daily,
)
forecasts = {
    name: get_model(name).forecast(task)
    for name in ("randomwalk", "randomwalknowcast")
}
for samples in forecasts.values():
    assert list(samples.columns) == [
        "location",
        "target_end_date",
        "sample",
        "value",
    ]
    assert samples.groupby(["location", "target_end_date"]).size().eq(100).all()
```

The random walk forecasts each place's weekly count on the log scale from its latest complete week.
The random walk nowcast first nowcasts the recent weeks, by estimating the reporting delay from how counts grew between releases, then forecasts each nowcast draw with the random walk.

Next: [How to](how-to.md).
