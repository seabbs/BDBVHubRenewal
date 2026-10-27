# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Nowcasts and forecasts for the BDBV Modeling Hub.

The package has these parts:

- `data`: download, tidy and load every stream the hub publishes.
- `vintages` and `simulate`: reconstructed and simulated vintages.
- `models`: the model interface, registry, persistence and the renewal
  variants (`models.renewal`).
- `model`: the PyRenew renewal model (JAX), imported lazily.
- `bvd`: BVDOutbreakSize estimates as of a forecast's cut-off.
- `delays`, `latent` and `observation`: NumPy references for the model's
  delay, renewal and observation layers.
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
    dates,
    delays,
    hub,
    latent,
    models,
    observation,
    plots,
    scoring,
    simulate,
    vintages,
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
    "dates",
    "delays",
    "get_model",
    "hub",
    "latent",
    "load_config",
    "load_delays",
    "load_target",
    "models",
    "observation",
    "plot_nowcast",
    "plots",
    "register_model",
    "relative_skill",
    "run_development",
    "score_forecasts",
    "scoring",
    "simulate",
    "to_hub_table",
    "validate_submission",
    "vintages",
]
__version__ = "0.1.0"
