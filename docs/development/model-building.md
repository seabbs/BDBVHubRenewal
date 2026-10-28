# Model building

How the renewal model was built, one component at a time, with the checks at each step.
{ .page-subtitle }

The model was built as a ladder of rungs, following a Bayesian workflow.
Each rung adds one component to the one before, from a single national renewal to the full province model.
Each is fitted to a simulated outbreak with known parameters and to a real vintage.
The real-vintage columns come from forecasts at one origin, the 2026-10-09 vintage, scored against the 2026-10-24 vintage as reported.
They are a check on each step, not a comparison of models.
The [Backtest](../backtest.md) compares the models.
A component is kept only if the sampler stays healthy, the simulated parameters are recovered, and the real forecasts are no worse.
The rungs and their settings are in `bdbv_hub_renewal.model.model.LADDER`, and `scripts/ladder.py` runs them into `results/ladder.csv`.
The [Model](../methods/model.md) page describes the result.

--8<-- "ladder.md"
