# CLI

The `bdbvhubrenewal` command.
{ .page-subtitle }

Paths to the tidy data default to `$BDBV_DATA_DIR` or `data/`.
`bdbvhubrenewal <command> --help` lists each command's options.

| Command | What it does |
|---|---|
| `data` | Download the hub data and write the tidy tables to `data/`. |
| `backtest` | Run the development backtest, the evaluation on the real releases (`results/nowcast-*.csv`) and the recovery check. Fitted models are never run: their rows in `results/` come from the fit store and are kept. With `--models`, only those models run and the others' rows are kept. |
| `backtest --lopo` | Leave one province out: forecast each province with its counts masked, for the variants that take a mask. |
| `backtest --scenarios-only` | Run only the scenarios calibrated to the real data, from `--scenario-as-of` and `--scenario-dates`. `--scenarios` runs only the named scenarios. |
| `backtest --recovery-only`, `--no-recovery` | Run only the recovery check on the simulated outbreak, or leave it out, so it can run apart. |
| `backtest --chunk K/N` | With `--recovery-only` or `--scenarios-only`, run only block K of N (from 0) of the dates. Stacking the blocks' results (`concat-results`) gives one run's. |
| `bvd` | Cache the BVDOutbreakSize releases each vintage may use. |
| `compare-hub [--fetch]` | Score our retrospective forecasts and the hub's submitted ones on the same truth and weeks. |
| `nowcast` | Forecast from a real vintage, write the hub file and check it. With `--config` and `--fit`, forecast from a fit configuration. |
| `zone-backtest` | Score `randomwalk`, `randomwalknowcast` and `pprandomwalknowcast` run at zone level on the real vintages. `--chunk K/N` runs only block K of N of the origins. |
| `fit-store plan`, `fit`, `load`, `unpack` | List the fits the fit store lacks (`--origins` for some origins), make one, load the stored fits into `results/`, or write a submission candidate's stored fit directory for the submission. |
| `fit` | Fit a configuration and write the fit directory and its diagnostics, for the submission CI. |
| `plot-submission` | Plot a submitted hub file. |
| `plot-ppc` | Plot a fit's posterior predictive checks (`ppc.json`). |
| `coherence` | Compare the national data with the sum of the provinces. |
| `validate` | Run the hub's checks on files. |
| `metadata` | Write and check a model's metadata YAML. |
| `hub-id` | Print the hub model id that submissions use. |
| `freeze-round` | Write a round's docs page to the ledger once its pull request has merged. |
| `report` | Make the docs figures and tables. |
| `digest [--with PATH...] [--text TEXT] [--model M]` | Print a hash of what a backtest reads, which keys its cached outputs in the Pages workflow. With `--model`, only that model's code (its import closure, less the other models' modules) with the data. |
| `backtest --forecast-only`, `zone-backtest --forecast-only` | Forecast without scoring and write the forecasts to `--parts`, so each model can run apart. |
| `score PART...` | Score the forecasts of `--forecast-only` runs, every model together, and write the results files one run of each backtest writes. |
| `concat-results --out DIR [--dedupe FILE...] PART...` | Stack the results files of runs made apart, such as blocks of dates, in the order given. `--dedupe` keeps one copy of rows every part writes in full. |

`--version` prints the package version.

## Evaluation

```bash
task backtest           # development backtest, real releases and recovery check
task backtest-lopo      # leave one province out
task backtest-scenarios # simulated scenarios calibrated to the real data
task backtest-zones     # random walk models at health-zone level
task results            # every derived file in results/, as the Pages build makes them
task docs               # build the site into _site/ (runs task results, about 25 min)
task serve              # serve the site (run task results once first)
```

`--models` runs some models and keeps the others' rows in `results/`.
The files in `results/` are not committed, apart from `development-log.csv`, `ladder.csv` and `model-choice.yml`.
The Pages build makes the rest with the steps of `task results`, from the code, the data and the fit store.

## Fit store

The backtest and the docs build fit no model, apart from the zone backtest, which fits `pprandomwalknowcast` at each origin ([Evaluation](../methods/evaluation.md#health-zones)).
The Fit store workflow (`.github/workflows/fit-store.yml`) fits each fitted model once at each real origin.
Each fit is one file on the `fit-store` branch, `<model>/<origin>/<key>.json.gz`.
The key hashes:

- the task at the origin, which holds only the releases up to it;
- the model as built, including its variant and sampler settings;
- the BVDOutbreakSize release, for `renewalbvd`;
- the code the model's fit can run (its import closure with `evaluation/store.py` and `evaluation/backtest.py`), without docstrings or comments;
  the zone split (`forecasters/zones.py`, `evaluation/zones.py`) is left out, as a stored fit holds national and province samples only and its zones are split from them at load;
- the runtime dependencies in `uv.lock`.

The renewal variants use the sampler table of the submission candidate that runs them, in `ci/policy.toml`.
`pprandomwalknowcast` uses its own.
A model code or sampler change refits every origin.
A new hub release adds one fit per model.
The task at each origin is the submission's (`backtest.vintage_task`): weeks from 6 before the reference date to one week past `[forecast] max_horizon`, so the next round can be forecast from the same fit when no new release comes, and the latest earlier delays for a release with none of its own.
The backtest loads the weeks from `[forecast] min_horizon` to `max_horizon`.
A submission candidate's fit also holds the fit directory the submission reads (samples, the unassigned national part, `ppc.json` and `diagnostics.json`), and the submission forecasts from it (`fit-store unpack`).

The workflow runs on pushes to `main` that change these inputs, every 6 hours for new hub releases, and when the submission finds a round's fits missing.
It fits and stores the newest origin first, then the other origins, so a backfill never holds up a round.
A push never cancels a running refit: the latest push waits and runs after it.
The fit jobs are read-only, and one job with no project code pushes their files and prunes the branch.
It keeps the current key's file and the newest good fit of older code for each model and origin.
A fit that fails is stored as failed, and is fitted again only when its inputs change or a dispatch sets `retry_failed`.
On a pull request the workflow only lists the fits the change needs.
Only `main` writes to `fit-store`.
To test a branch, dispatch the workflow on it with `store_branch` set to a name starting `fit-store-scratch`, and `models` to limit it. Any other `store_branch` stops the run.
When fits are stored, the workflow rebuilds the docs.

```bash
task fit-store-plan  # the fits the store lacks for this code and data
task fit-store       # load the stored fits into results/
```

`task backtest` loads the store first.
Where the store has no fit for the current key, `load` uses the newest stored fit of that model and origin.
Pages with fitted-model results list the commits of their fits and the fits not yet made.
A full refit is 20 fits (4 models at 5 origins): 43 minutes and 5.3 runner-hours, with the slowest fit 39 minutes (`renewalbvd`).
A new release adds 4 fits: about 40 minutes and 1.3 runner-hours.

To fit a model locally, for example a new variant:

```bash
uv run bdbvhubrenewal fit-store fit --model renewal --origin 2026-10-24 --store output/fit-store
uv run bdbvhubrenewal fit-store load --store output/fit-store
```

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
