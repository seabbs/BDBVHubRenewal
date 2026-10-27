# Getting started

Install the package, run the pipeline and build this site.
{ .page-subtitle }

## Install

The package needs Python 3.11 or later, [uv](https://docs.astral.sh/uv/) and [Task](https://taskfile.dev).
`task validate-hub` also needs R with `hubValidations`.

```bash
task setup      # install the package with the dev and docs groups
task test       # run the tests
task lint       # ruff and mypy
```

## Run

```bash
task data               # download the hub data and rebuild data/
task backtest           # development backtest and recovery check
task backtest-heldout   # score the held-out vintages under the chosen model
task nowcast            # forecast from the latest vintage and write a hub file
task docs               # build this site into _site/
```

The tidy data are committed, so `task backtest` works from a clean clone without downloading.
`results/` holds the score tables and the model choice record, and is committed.
`output/` holds forecasts and other run output, and is not.

The comparison with hub forecasts downloads other teams' files, so it is run on its own:

```bash
uv run bdbv-forecasts compare-hub --fetch
```

Every command is listed on the [CLI page](development/cli.md).
