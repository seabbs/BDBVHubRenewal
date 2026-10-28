# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Nowcasts and forecasts for the BDBV Modeling Hub.

The package has these parts:

- `data`: the hub data, dates, reconstructed and simulated vintages,
  empirical delays and BVDOutbreakSize releases.
- `forecasters`: the forecaster interface, registry, persistence and the
  renewal variants (`forecasters.renewal`).
- `model`: the PyRenew renewal model (JAX), imported lazily.
- `submission`: the hub file writer and checks, model metadata, the CI
  fit configurations and the submitted rounds.
- `evaluation`: scoring, the backtest, held-out evaluation, posterior
  predictive checks and the comparison with hub forecasts.
- `reporting`: figures and docs tables.
- `cli`: the `bdbvhubrenewal` command.

- Repository: <https://github.com/seabbs/BDBVHubRenewal>
- Docs: <https://samabbott.co.uk/BDBVHubRenewal/>
"""

from bdbvhubrenewal import (
    data,
    evaluation,
    forecasters,
    reporting,
    submission,
)
from bdbvhubrenewal.data import load_delays, load_target
from bdbvhubrenewal.evaluation.backtest import (
    BacktestConfig,
    load_config,
    run_development,
)
from bdbvhubrenewal.evaluation.scoring import relative_skill, score_forecasts
from bdbvhubrenewal.forecasters import (
    ForecastModel,
    ForecastTask,
    get_model,
    register_model,
)
from bdbvhubrenewal.reporting.plots import plot_nowcast
from bdbvhubrenewal.submission.hub import to_hub_table, validate_submission

__all__ = [
    "BacktestConfig",
    "ForecastModel",
    "ForecastTask",
    "data",
    "evaluation",
    "forecasters",
    "get_model",
    "load_config",
    "load_delays",
    "load_target",
    "plot_nowcast",
    "register_model",
    "relative_skill",
    "reporting",
    "run_development",
    "score_forecasts",
    "submission",
    "to_hub_table",
    "validate_submission",
]
__version__ = "0.1.0"
