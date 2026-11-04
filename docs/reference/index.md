# API reference

The Python API of `bdbvhubrenewal`, grouped by task. See [Setup](../setup.md) to install it and [Getting started](../getting-started.ipynb) for a worked example.
{ .page-subtitle }

| Task | Pages | Start with |
|---|---|---|
| Data | [Data](data.md) | `load_target`, `load_daily`, `load_delays`, `delays_for_vintage` |
| Models | [Forecasters](forecasters.md), [Renewal model](model.md), [Baselinenowcast](baselinenowcast.md) | `get_model`, `list_models`, `ForecastTask`, `ForecastModel.forecast` |
| Evaluation | [Evaluation](evaluation.md) | `make_tasks`, `run_forecasts`, `score_forecasts`, `release_backtest` |
| Reporting | [Figures and tables](reporting.md), [Submission](submission.md) | `plot_location_grid`, `plot_forecasts`, `to_hub_table`, `write_submission` |
| CLI | [CLI](cli.md) | `bdbvhubrenewal --help` |

## A forecast in five steps

Run from the repository root after `bdbvhubrenewal data` has written the tidy tables to `data/`.

```python
import dataclasses

from bdbvhubrenewal.data import (
    delays_for_vintage,
    load_daily,
    load_delays,
    load_target,
)
from bdbvhubrenewal.evaluation import backtest as bt
from bdbvhubrenewal.evaluation.nowcast import release_backtest
from bdbvhubrenewal.forecasters import get_model
from bdbvhubrenewal.reporting.plots import plot_location_grid

# 1. Load the data.
target, daily, delays = load_target(), load_daily(), load_delays()
real = target.loc[target["kind"] == "real"]

# 2. Build the task a model sees at one release.
config = dataclasses.replace(
    bt.load_config("config/backtest.toml"), min_horizon=-6
)
vintage = "2026-10-17"
task = bt.make_tasks(
    real.loc[real["vintage"] == vintage],
    delays_for_vintage(delays, vintage)[0],
    config,
    history=real,
    daily=daily,
)[0]

# 3. Forecast with models from the registry.
models = [get_model("randomwalk"), get_model("randomwalknowcast")]
forecasts = bt.run_forecasts(models, [task])

# 4. Score against later releases.
scores = release_backtest(
    [task], daily, target, models=(), stored=forecasts
).scores

# 5. Plot.
fig = plot_location_grid(forecasts, real, ["CD"])
```
