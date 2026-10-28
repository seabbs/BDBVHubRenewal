# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Evaluation: how forecasts are scored and compared.

- `scoring`: WIS, coverage, bias and relative skill.
- `backtest`: the rolling-origin development backtest, the recovery
  check on simulated vintages and leave one province out.
- `heldout`: the model choice record and the rolling held-out evaluation.
- `ppc`: posterior predictive check summaries and their plot.
- `hubcompare`: comparison with the forecasts submitted to the hub.
"""
