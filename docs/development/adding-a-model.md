# Adding a model

Adding a model to the registry, so every evaluation includes it.
{ .page-subtitle }

A model is a class with a `name` and a `forecast(task)` method.
`task` is a [`ForecastTask`][bdbvhubrenewal.forecasters.base.ForecastTask]: the target rows and delay data of one vintage, the provinces and weeks to forecast, the number of samples and a seed.
`forecast` returns one row per province, week and sample, with columns `location`, `target_end_date`, `sample` and `value`.

```python
from dataclasses import dataclass

import pandas as pd

from bdbvhubrenewal.forecasters import ForecastTask, register_model


@dataclass
class MyModel:
    name: str = "my_model"

    def forecast(self, task: ForecastTask) -> pd.DataFrame: ...


register_model("my_model", MyModel, "One line describing it.")
```

Import the module in `bdbvhubrenewal/forecasters/__init__.py`.
The backtest, the [Backtest](../backtest.md) and [Nowcast](../nowcast.md) pages, `compare-hub` and `nowcast --model my_model` then include it.
Once it is chosen for submission, an entry in `results/model-choice.yml` adds it to the [held-out evaluation](../heldout.md).
The harness adds the national total, the horizons and the vintage kind.

For the leave-one-province-out check, set `supports_mask = True` and drop the provinces in `task.mask` from the likelihood while still forecasting them.

## Renewal variants

A new renewal variant is one entry in `VARIANTS` in [`forecasters.renewal`][bdbvhubrenewal.forecasters.renewal].
It names a base `ModelConfig` and the fields it changes.
Fitted variants are not run by the docs build or the smoke tests.
Run them apart and commit what they write to `results/`:

```bash
BDBV_RENEWAL_SAMPLER='{"num_chains": 2, "num_warmup": 300, "num_samples": 300}' \
  task backtest -- --models renewal
```

A run with `--models` replaces only those models' rows, and `BDBV_RENEWAL_SAMPLER` sets the NUTS settings.
