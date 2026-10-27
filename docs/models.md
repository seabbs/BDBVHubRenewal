# Models

The models in the registry and how to add one.
{ .page-subtitle }

## Persistence

`persistence` is the baseline every other model is compared against.
For each province with cases it:

1. finds the last week judged complete, using the 90th centile of the delays published with the vintage (or 10 days if none);
2. returns the published counts for that week and earlier;
3. carries the last complete week forward on the log scale as a random walk, with the weekly spread estimated from the last 8 complete weeks;
4. draws Poisson counts around that path.

The national value is the sum of the province samples, sample by sample.
Each sample is one joint trajectory across weeks and provinces.
The maths is on the [Methods page](methods.md#persistence-baseline).

It ignores right truncation in weeks it judges complete.
In the backtest those weeks are still revised upwards, so it underpredicts them (see [Backtest](backtest.md)).

## Adding a model

A model is a class with a `name` and a `forecast(task)` method.
`task` is a [`ForecastTask`][bdbv_forecasts.models.base.ForecastTask]: the target rows and delay data of one vintage, the provinces and weeks to forecast, the number of samples and a seed.
`forecast` returns one row per province, week and sample, with columns `location`, `target_end_date`, `sample` and `value`.

```python
from dataclasses import dataclass

import pandas as pd

from bdbv_forecasts.models import ForecastTask, register_model


@dataclass
class MyModel:
    name: str = "my_model"

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        ...


register_model("my_model", MyModel, "One line describing it.")
```

Import the module in `bdbv_forecasts/models/__init__.py`.
Then:

- `task backtest` scores it on every development vintage;
- the [Backtest](backtest.md) page shows its scores and its WIS relative to persistence;
- the [Nowcast](nowcast.md) page shows its current forecast;
- `bdbv-forecasts nowcast --model my_model` writes and checks its hub file.

The harness adds the national total, the horizons and the vintage kind.
Model colours follow registry order, so a model keeps its colour as others are added.

## Hub submission

`bdbv-forecasts nowcast` writes `model-output/<model_id>/<reference_date>-<model_id>.csv` under `output/submission/`.
The model id is `epiforecasts-<model>`.
It checks the file with `validate_submission`, which mirrors the hub's checks:

- column names and order free, no extra columns;
- one reference date, a Saturday, listed in `tasks.json` for live rounds;
- known locations, the one target and `sample` output only;
- target weeks are Saturdays from 2026-02-14 to three weeks after the reference date;
- exactly 100 samples per location, and every sample covers the same weeks;
- integer sample ids, finite non-negative values, no duplicate rows;
- the submission window, Sunday to Tuesday in Kinshasa time, when a time is given.

`task validate-hub` then runs the hub's own `hubValidations::validate_submission()` in R against a fresh clone of the hub.
The persistence file for 2026-10-31 passes every check.

`bdbv-forecasts metadata` writes `hub/model-metadata/epiforecasts-<model>.yml` and checks it against the hub's schema.
The contributor list, licence and model names are placeholders until Sam confirms them.

!!! note "Sample ids"
    Sample ids run from 1 to 100 within each location and are shared across locations.
    `hubValidations` reads this as samples that are joint across locations, which they are: the national sample is the sum of the province samples with the same id.
