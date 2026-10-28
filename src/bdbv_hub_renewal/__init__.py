# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Nowcasts and forecasts for the BDBV Modeling Hub.

The package has these parts:

- `data`: the hub data, dates, reconstructed and simulated vintages,
  empirical delays and BVDOutbreakSize releases.
- `forecasters`: the forecaster interface, registry, persistence and the
  renewal variants (`forecasters.renewal`).
- `model`: the PyRenew renewal model (JAX), imported lazily.
- `hub`: write and check submission files and model metadata.
- `evaluation`: scoring, the backtest, held-out evaluation, posterior
  predictive checks and the comparison with hub forecasts.
- `plots` and `report`: figures and docs tables.

- Repository: <https://github.com/seabbs/BDBVHubRenewal>
- Docs: <https://seabbs.github.io/BDBVHubRenewal/>
"""

from bdbv_hub_renewal import (
    data,
    evaluation,
    forecasters,
    hub,
    plots,
)
from bdbv_hub_renewal.data import load_delays, load_target
from bdbv_hub_renewal.evaluation.backtest import (
    BacktestConfig,
    load_config,
    run_development,
)
from bdbv_hub_renewal.evaluation.scoring import relative_skill, score_forecasts
from bdbv_hub_renewal.forecasters import (
    ForecastModel,
    ForecastTask,
    get_model,
    register_model,
)
from bdbv_hub_renewal.hub import to_hub_table, validate_submission
from bdbv_hub_renewal.plots import plot_nowcast

__all__ = [
    "BacktestConfig",
    "ForecastModel",
    "ForecastTask",
    "data",
    "evaluation",
    "forecasters",
    "get_model",
    "hub",
    "load_config",
    "load_delays",
    "load_target",
    "plot_nowcast",
    "plots",
    "register_model",
    "relative_skill",
    "run_development",
    "score_forecasts",
    "to_hub_table",
    "validate_submission",
]
__version__ = "0.1.0"
