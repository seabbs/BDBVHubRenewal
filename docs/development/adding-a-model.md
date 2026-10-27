# Adding a model

Adding a model to the registry, so every evaluation includes it.
{ .page-subtitle }

A model is a class with a `name` and a `forecast(task)` method.
`task` is a [`ForecastTask`][bdbv_hub_renewal.models.base.ForecastTask]: the target rows and delay data of one vintage, the provinces and weeks to forecast, the number of samples and a seed.
`forecast` returns one row per province, week and sample, with columns `location`, `target_end_date`, `sample` and `value`.

```python
from dataclasses import dataclass

import pandas as pd

from bdbv_hub_renewal.models import ForecastTask, register_model


@dataclass
class MyModel:
    name: str = "my_model"

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        ...


register_model("my_model", MyModel, "One line describing it.")
```

Import the module in `bdbv_hub_renewal/models/__init__.py`.
Then:

- `task backtest` scores it on every development vintage;
- the [Backtest](../backtest.md) page shows its scores and its WIS relative to persistence;
- the [Nowcast](../nowcast.md) page shows its current forecast;
- `bdbv-hub-renewal compare-hub` adds its retrospective forecasts to the comparison with hub forecasts;
- `bdbv-hub-renewal nowcast --model my_model` writes and checks its hub file.

Once it is chosen for submission, add an entry to `results/model-choice.yml`.
The [held-out evaluation](../heldout.md) then scores it on every later vintage.

The harness adds the national total, the horizons and the vintage kind.
Model colours follow registry order, so a model keeps its colour as others are added.

To take part in the leave-one-province-out check, set `supports_mask = True` on the class.
The model must then drop the counts of the provinces in `task.mask` from its likelihood, while still forecasting them.

## Renewal variants

A new renewal variant is one entry in `VARIANTS` in [`models.renewal`][bdbv_hub_renewal.models.renewal].
It names a base configuration (`MAIN` or `FALLBACK`) and the `ModelConfig` fields it changes.
Registered variants are marked `fitted`, because each forecast is an MCMC fit.
So the docs build and the smoke tests run only the models with no fit.
Run the fitted variants apart, for example on a bigger machine, and commit what they write to `results/`:

```bash
BDBV_RENEWAL_SAMPLER='{"num_chains": 2, "num_warmup": 300, "num_samples": 300}' \
  task backtest -- --models renewal
```

A run with `--models` replaces only those models' rows in `results/`.
`BDBV_RENEWAL_SAMPLER` sets the NUTS settings of every registered variant.

The PyRenew model code in `bdbv_hub_renewal.pyrenew` and its tests in `tests/pyrenew_model/` are synced from the prototype:

```bash
uv run python scripts/sync_pyrenew.py <prototype dir>
task format
```

The adapter reads only `ModelConfig`, `MAIN`, `FALLBACK`, `build_model`, `fit`, `forecast_trajectories` and `adapter.vintage_from_tables` from it, so a new model version drops in if it keeps those.
