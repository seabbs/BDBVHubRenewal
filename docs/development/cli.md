# CLI

The `bdbvhubrenewal` command.
{ .page-subtitle }

Paths to the tidy data default to `$BDBV_DATA_DIR` or `data/`.
`bdbvhubrenewal <command> --help` lists each command's options.

| Command | What it does |
|---|---|
| `data` | Download the hub data and write the tidy tables to `data/`. |
| `backtest` | Run the development backtest, the evaluation on the real releases (`results/nowcast-*.csv`) and the recovery check. With `--models`, only those models run and the others' rows in `results/` are kept. |
| `backtest --evaluate-holdout` | Score every held-out vintage under the model chosen at the time, with the other models alongside. |
| `backtest --no-fit` | Run only the models with no fit. Fitted models' rows in `results/` are kept, and their held-out forecasts are scored again. The docs build uses this, so it never fits a model. |
| `backtest --lopo` | Leave one province out: forecast each province with its counts masked, for the variants that take a mask. |
| `backtest --scenarios-only` | Run only the scenarios calibrated to the real data, from `--scenario-as-of` and `--scenario-dates`. |
| `bvd` | Cache the BVDOutbreakSize releases each vintage may use. |
| `compare-hub [--fetch]` | Score our retrospective forecasts and the hub's submitted ones on the same truth and weeks. |
| `nowcast` | Forecast from a real vintage, write the hub file and check it. With `--config` and `--fit`, forecast from a fit configuration. |
| `zone-backtest` | Score `randomwalk` and `randomwalknowcast` run at zone level on the real vintages. |
| `fit` | Fit a configuration and write the fit directory and its diagnostics, for the submission CI. |
| `plot-submission` | Plot a submitted hub file. |
| `plot-ppc` | Plot a fit's posterior predictive checks (`ppc.json`). |
| `coherence` | Compare the national data with the sum of the provinces. |
| `validate` | Run the hub's checks on files. |
| `metadata` | Write and check a model's metadata YAML. |
| `hub-id` | Print the hub model id that submissions use. |
| `freeze-round` | Write a round's docs page to the ledger once its pull request has merged. |
| `report` | Make the docs figures and tables. |

`--version` prints the package version.

## Evaluation

```bash
task backtest           # development backtest, real releases and recovery check
task backtest-lopo      # leave one province out
task backtest-scenarios # simulated scenarios calibrated to the real data
task backtest-zones     # random walk models at health-zone level
task backtest-heldout   # score the held-out vintages
task docs               # build the site into _site/
```

The fitted models use the `[sampler]` settings in `config/backtest.toml`, the same as the submission's main fit.
`BDBV_RENEWAL_SAMPLER` overrides them, for example `'{"num_chains": 2, "num_warmup": 300, "num_samples": 300}'`.
Each forecast of a renewal variant is a fit.
`--models` runs some models and keeps the others' rows in `results/`, which is committed.
The docs build runs only the random walk models and reads the fitted models from there.

The comparison with hub forecasts downloads other teams' files, so it is run on its own:

```bash
uv run bdbvhubrenewal compare-hub --fetch
```

## Hub files

`bdbvhubrenewal nowcast` writes `model-output/<model_id>/<reference_date>-<model_id>.csv` under `output/submission/`.
The model id is `seabbs-` plus the model's `model_abbr` in `config/hub.toml`.
When the reference date is an open round in `tasks.json`, it checks the file with `submission.validate`, the Python port of the hub's checks:

- column names and order free, no extra columns;
- one reference date, listed in `tasks.json`;
- known locations, the one target and `sample` output only;
- target weeks listed in `tasks.json`;
- exactly 100 samples per location, and every sample covers the same weeks;
- integer sample ids, finite non-negative values, no duplicate rows.

It skips the submission window and the model metadata, which `bdbvhubrenewal metadata` checks.
For any other reference date it prints a note and skips the checks.

`task validate-hub` then runs the hub's own `hubValidations::validate_submission()` in R against a fresh clone of the hub.

`bdbvhubrenewal metadata` writes `hub/model-metadata/<model id>.yml` and checks it against the hub's schema.

!!! note "Sample ids"
    Sample ids run from 1 to 100 and are shared across locations, so `hubValidations` reads samples as joint across locations.

The [CLI reference](../reference/cli.md) documents `main` and `build_parser`.
