# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Nowcasts and forecasts for the BDBV Modeling Hub.

The package has these parts:

- `data`: the hub data, dates, reconstructed and simulated vintages,
  empirical delays and BVDOutbreakSize releases.
- `models`: the model interface, registry, persistence and the renewal
  variants (`models.renewal`).
- `model`: the PyRenew renewal model (JAX), imported lazily.
- `hub`: write and check submission files and model metadata.
- `scoring` and `backtest`: score forecasts over vintages, with a
  development and held-out split.
- `plots` and `report`: figures and docs tables.

- Repository: <https://github.com/seabbs/BDBVHubRenewal>
- Docs: <https://seabbs.github.io/BDBVHubRenewal/>
"""

from bdbv_hub_renewal import (
    backtest,
    data,
    hub,
    models,
    plots,
    scoring,
)
from bdbv_hub_renewal.backtest import (
    BacktestConfig,
    load_config,
    run_development,
)
from bdbv_hub_renewal.data import load_delays, load_target
from bdbv_hub_renewal.hub import to_hub_table, validate_submission
from bdbv_hub_renewal.models import (
    ForecastModel,
    ForecastTask,
    get_model,
    register_model,
)
from bdbv_hub_renewal.plots import plot_nowcast
from bdbv_hub_renewal.scoring import relative_skill, score_forecasts

__all__ = [
    "BacktestConfig",
    "ForecastModel",
    "ForecastTask",
    "backtest",
    "data",
    "get_model",
    "hub",
    "load_config",
    "load_delays",
    "load_target",
    "models",
    "plot_nowcast",
    "plots",
    "register_model",
    "relative_skill",
    "run_development",
    "score_forecasts",
    "scoring",
    "to_hub_table",
    "validate_submission",
]
__version__ = "0.1.0"
