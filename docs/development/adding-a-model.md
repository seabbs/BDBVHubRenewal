# Adding a model

How to add a model to the registry, so every evaluation picks it up.
{ .page-subtitle }

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
- the [Backtest](../backtest.md) page shows its scores and its WIS relative to persistence;
- the [Nowcast](../nowcast.md) page shows its current forecast;
- `bdbv-forecasts compare-hub` adds its retrospective forecasts to the comparison with hub forecasts;
- `bdbv-forecasts nowcast --model my_model` writes and checks its hub file.

Once it is chosen for submission, add an entry to `results/model-choice.yml`.
The [held-out evaluation](../heldout.md) then scores it on every later vintage.

The harness adds the national total, the horizons and the vintage kind.
Model colours follow registry order, so a model keeps its colour as others are added.

To take part in the leave-one-province-out check, set `supports_mask = True` on the class.
The model must then drop the counts of the provinces in `task.mask` from its likelihood, while still forecasting them.
