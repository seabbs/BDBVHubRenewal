# CLI

The `bdbvhubrenewal` command.
{ .page-subtitle }

Paths to the tidy data default to `$BDBV_DATA_DIR` or `data/`.
`bdbvhubrenewal <command> --help` lists each command's options.

| Command | What it does |
|---|---|
| `data` | Download the hub data and write the tidy tables to `data/`. |
| `backtest` | Run the development backtest and the recovery check. With `--models`, only those models run and the others' rows in `results/` are kept. |
| `backtest --evaluate-holdout` | Score every held-out vintage under the model chosen at the time, with the other models alongside. |
| `backtest --no-fit` | Run only the models with no fit. Fitted models' rows in `results/` are kept, and their held-out forecasts are scored again. The docs build uses this, so it never fits a model. |
| `backtest --lopo` | Leave one province out: forecast each province with its counts masked, for the variants that take a mask. |
| `compare-hub [--fetch]` | Score our retrospective forecasts and the hub's submitted ones on the same truth and weeks. |
| `nowcast` | Forecast from a real vintage, write the hub file and check it. With `--config` and `--fit`, forecast from a fit configuration. |
| `fit` | Fit a configuration and write the fit directory and its diagnostics, for the submission CI. |
| `plot-submission` | Plot a submitted hub file. |
| `plot-ppc` | Plot a fit's posterior predictive checks (`ppc.json`). |
| `coherence` | Compare the national data with the sum of the provinces. |
| `validate` | Run the hub's checks on files. |
| `metadata` | Write and check a model's metadata YAML. |
| `report` | Make the docs figures and tables. |

`--version` prints the package version.

## Hub files

`bdbvhubrenewal nowcast` writes `model-output/<model_id>/<reference_date>-<model_id>.csv` under `output/submission/`.
The model id is `epiforecasts-<model>`.
It checks the file with `validate_submission`, which mirrors the hub's checks:

- column names and order free, no extra columns;
- one reference date, a Saturday, listed in `tasks.json` for live rounds;
- known locations, the one target and `sample` output only;
- target weeks are Saturdays from 2026-02-14 to three weeks after the reference date;
- exactly 100 samples per location, and every sample covers the same weeks;
- integer sample ids, finite non-negative values, no duplicate rows;
- the submission window, Sunday to Tuesday in Kinshasa time, when a time is given.

`task validate-hub` then runs the hub's own `hubValidations::validate_submission()` in R against a fresh clone of the hub.
The persistence file for 2026-10-31 passes every check.

`bdbvhubrenewal metadata` writes `hub/model-metadata/epiforecasts-<model>.yml` and checks it against the hub's schema.
Sam confirms the contributor list, licence and model name before a model is first submitted.

!!! note "Sample ids"
    Sample ids run from 1 to 100 within each location and are shared across locations.
    `hubValidations` reads this as samples that are joint across locations, which they are.
    The national sample is the sum of the province samples with the same id.
