# Getting started

Install the package, fit the model to the latest data and write a hub file.
{ .page-subtitle }

## Install

The package needs Python 3.13 or later, [uv](https://docs.astral.sh/uv/) and [Task](https://taskfile.dev).
`task validate-hub` also needs R with `hubValidations`.

```bash
task setup      # the package, PyRenew (JAX, NumPyro) and the dev and docs groups
task test       # the tests
task test-slow  # small real model fits
task lint       # ruff and mypy
```

The tidy data in `data/` are committed, so nothing needs downloading to start.
`task data` refreshes them from the hub.

## Fit the model

This fits the main variant, `renewal`, to the latest vintage, with the sampler settings the submission workflow uses.

```console
$ uv run bdbvhubrenewal fit --model renewal --config main \
    --out output/fits/main
Fitted config 'main' on vintage 2026-10-24 (sampler nuts); wrote output/fits/main.
```

The fit directory holds what the submission workflow reads:

| File | What it holds |
|---|---|
| `samples.csv.gz` | 100 joint trajectories for every province and week, drawn at fit time |
| `diagnostics.json` | R-hat and ESS of every sampled parameter, and per-chain sampler checks |
| `ppc.json` | In-sample posterior predictive checks of the known and imputed onsets |
| `trajectories.json` | The variant and the provinces it reports |
| `fit.json`, `progress.log` | What was fitted, and progress lines |

Each chain in `diagnostics.json` reports divergences, the share of draws at the tree depth cap, the mean acceptance, E-BFMI and the step size.
`worst_rhat` lists the ten parameters with the highest R-hat.

## Forecast from the fit

```console
$ uv run bdbvhubrenewal nowcast --model renewal --config main \
    --fit output/fits/main --out output/candidates/main
Wrote output/candidates/main/model-output/epiforecasts-seabbsrenewal/2026-10-31-epiforecasts-seabbsrenewal.csv (9000 rows, locations incl. CD).
```

The hub file has 100 samples for the 8 provinces with cases and the national total, over 10 weeks.
National sample $k$ is the sum of province sample $k$.
`national.json` beside it lists the modelled provinces and any national cases not assigned to a province.

```console
$ uv run bdbvhubrenewal plot-submission \
    --submission output/candidates/main/model-output/epiforecasts-seabbsrenewal/2026-10-31-epiforecasts-seabbsrenewal.csv \
    --out nowcast.png
```

![Submitted samples for the country and the three largest provinces: 100 faint trajectories, 50% and 90% intervals and the median, over the known and imputed onsets.](assets/getting-started/nowcast.png)

## Check the fit

```console
$ uv run bdbvhubrenewal plot-ppc --ppc output/fits/main/ppc.json --out ppc.png
```

![Posterior predictive intervals of the known and imputed onsets in the data vintage, by province, against the data.](assets/getting-started/ppc.png)

The [CLI page](development/cli.md) lists every command, including the backtest and the other evaluations.
