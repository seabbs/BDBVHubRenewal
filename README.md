# Evidence for #221 and #237: the copula's arms

Made at fm-copula-rolling 1a97791 (#237) by `scripts/paper_rows.py`, which calls `bdbvhubrenewal.evaluation.trajectories` (`copula_arms`, `copula_scores`, `copula_rho`).

- `results/copula-scores.csv.gz`: one row per study, strata, model, arm and origin. The columns are those of the paper's evidence contract (`SCORE_COLUMNS`).
- `results/copula-rho.csv.gz`: each arm's correlations by model and origin (`RHO_COLUMNS`). It has `arm` and `n_scores` and no `place`.
- `results/day-grid.txt`: the daily Chronos-2 models' weekly median bias, WIS and coverage on the BDBV backtest for each correlation between adjacent days (`scripts/day_grid.py`). 0.985 gave the lowest WIS and median bias and is `CHRONOS_DAY_RHO`.

Studies:
- `ebola`: the BDBV release backtest, 5 origins, rolling truth at D = 35 days. Covers `randomwalknowcast` and the six Chronos-2 models.
- `germany`: the German development half, 2021-11-22 to 2022-02-09, 40-day truth. Covers `randomwalknowcast`, with `states` and `ages` scored apart.

Arms:
- `unpaired`: each place and week drawn on its own (the Chronos-2 models with `GaussianCopula()`).
- `fixed`: the prior. For Chronos-2 this is the model's own copula, so the daily models pair their days by `CHRONOS_DAY_RHO`.
- `rolling`: the rolling copula.
- `rolling_w25`: the rolling copula, with the prior worth 25 pairs.

The Chronos-2 marginal draws come from `scripts/evidence.py draws`, and the German samples from `scripts/rolling.py`. Both ran on archie's CPU.
