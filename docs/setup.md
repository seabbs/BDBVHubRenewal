# Setup

Install the package, then run it from a script, a REPL or a notebook.
{ .page-subtitle }

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), which manages Python and the packages for you.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install the package and its dependencies into `.venv/`.

```sh
git clone https://github.com/seabbs/BDBVHubRenewal.git
cd BDBVHubRenewal
uv sync
```

Run every command below from the `BDBVHubRenewal` folder.
The hub data are committed in `data/`, so nothing needs downloading.
`uv run bdbvhubrenewal data` fetches the latest hub data when you want it.

## Work interactively

`uv run` runs a command inside the project's environment.

| To | Run |
|---|---|
| Open a Python REPL | `uv run python` |
| Open an IPython REPL | `uv run --with ipython ipython` |
| Open Jupyter Lab | `uv run --with jupyterlab jupyter lab` |
| Run a script | `uv run python my_forecast.py` |

In VS Code, open the folder and pick `.venv/bin/python` as the interpreter or notebook kernel.
VS Code offers to install `ipykernel` the first time you run a notebook cell.

## Run a forecast from a script

Save this as `my_forecast.py` in the `BDBVHubRenewal` folder.
It forecasts with the random walk nowcast from one release, scores the forecast against later releases and saves a table and a figure.
[Getting started](getting-started.ipynb) walks through the same steps, and the [API reference](reference/index.md) documents each function.

```python
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
from bdbvhubrenewal.forecasters import NATIONAL
from bdbvhubrenewal.forecasters.chain import Settings
from bdbvhubrenewal.forecasters.randomwalk import RandomWalk, RandomWalkNowcast
from bdbvhubrenewal.reporting.plots import plot_location_grid

# Settings to change.
ORIGIN = "2026-10-17"  # the release to forecast from
N_SAMPLES = 200  # sample trajectories per location
WALK_WEEKS = 6  # weeks of data that set the random walk's spread
OUT = Path("output/my-forecast")

target, daily, delays = load_target(), load_daily(), load_delays()
real = target.loc[target["kind"] == "real"]

config = dataclasses.replace(
    bt.load_config("config/backtest.toml"), min_horizon=-6, n_samples=N_SAMPLES
)
task = bt.make_tasks(
    real.loc[real["vintage"] == ORIGIN],
    delays_for_vintage(delays, ORIGIN)[0],
    config,
    history=real,
    daily=daily,
)[0]

models = [
    RandomWalk(),
    RandomWalkNowcast(settings=Settings(walk_weeks=WALK_WEEKS)),
]
forecasts = bt.run_forecasts(models, [task])
scores = release_backtest(
    [task], daily, target, models=(), stored=forecasts
).scores

OUT.mkdir(parents=True, exist_ok=True)
forecasts.to_csv(OUT / "forecasts.csv", index=False)
table = relative_wis(scores.loc[scores["target"] == "reported"])
table.to_csv(OUT / "relative-wis.csv", index=False)
fig = plot_location_grid(forecasts, real, [NATIONAL, "CD-IT"])
fig.savefig(OUT / "forecast.png", dpi=150)
print(table)
```

Run it.

```sh
uv run python my_forecast.py
```

To change a setting, edit it at the top of the script and run it again.
`ORIGIN` takes any release date in `real["vintage"]`.
`get_model("randomwalknowcast")` builds the same model with its default settings, and `bdbvhubrenewal.forecasters.list_models()` lists every model.
The renewal models fit with a sampler and take minutes per task (see [Renewal models](methods/model.md)).

## Where outputs land

| Output | Folder |
|---|---|
| The script above | `output/my-forecast/` |
| `uv run bdbvhubrenewal backtest` | forecasts and scores in `results/` |
| `uv run bdbvhubrenewal nowcast` | the hub file in `output/submission/` |
| `uv run bdbvhubrenewal report` | the docs figures and tables in `docs/figures/` and `docs/generated/` |

`output/` is not committed, so anything there is yours to delete.
