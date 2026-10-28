<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="BDBVHubRenewal" width="275" height="44">
  </picture>
</h1>

**Renewal-model nowcasts and forecasts for the BDBV Modeling Hub.**

[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://seabbs.github.io/BDBVHubRenewal/)
[![Pages](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/pages.yml/badge.svg)](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/seabbs/BDBVHubRenewal/branch/main/graph/badge.svg)](https://codecov.io/gh/seabbs/BDBVHubRenewal)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

| Component | Tests | Lint | Coverage |
| --- | --- | --- | --- |
| Package (`src/`) | [![Tests](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/test.yml/badge.svg)](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/test.yml) | [![Lint](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/lint.yml/badge.svg)](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/lint.yml) | [![package coverage](https://codecov.io/gh/seabbs/BDBVHubRenewal/branch/main/graph/badge.svg?flag=package)](https://codecov.io/gh/seabbs/BDBVHubRenewal?flags%5B0%5D=package) |
| Submission CI (`ci/`) | [![Submission CI tests](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/submission-ci-tests.yml/badge.svg)](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/submission-ci-tests.yml) | [![Lint](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/lint.yml/badge.svg)](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/lint.yml) | [![ci coverage](https://codecov.io/gh/seabbs/BDBVHubRenewal/branch/main/graph/badge.svg?flag=ci)](https://codecov.io/gh/seabbs/BDBVHubRenewal?flags%5B0%5D=ci) |

Weekly nowcasts and forecasts of confirmed Bundibugyo virus disease cases for the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
The hub's target is weekly confirmed cases by symptom-onset week, by province and nationally, as 100 joint sample trajectories per location.
Forecasts are submitted as `epiforecasts-seabbsrenewal`, set in `config/hub.toml`.

## Models

All models are renewal processes built on [PyRenew](https://github.com/CDCgov/PyRenew) and fitted to each data vintage ([Model](https://seabbs.github.io/BDBVHubRenewal/methods/model/)).
Each fits the hub's known and imputed onsets together, with time-varying delays, missingness and ascertainment, and models the hub's imputation as the hub makes it.
National forecasts are the sum of the province trajectories.

| Model | What it is |
|---|---|
| `renewal` | Spatial and partially pooled. Each province with cases has its own renewal process. The reproduction numbers share a national trend with pooled province deviations. Provinces import cases from each other through a gravity kernel and deplete their susceptibles. |
| `renewalsp` | Single patch. One national renewal process, with each national trajectory split across provinces by their recent share of onsets. |
| `renewalbvd` | The spatial model informed by the latest posterior estimates of the [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize) model. |
| `persistence` | The baseline: the last complete week carried forward. |

## Contents

- Downloads and tidies every data stream the hub publishes, with every vintage, and builds reconstructed and simulated vintages for testing ([Data](https://seabbs.github.io/BDBVHubRenewal/methods/data/)).
- Scores every model against persistence on past vintages, then on every vintage after the model choice as a rolling held-out evaluation, re-scored as later data arrive ([Evaluation](https://seabbs.github.io/BDBVHubRenewal/methods/evaluation/), [Backtest](https://seabbs.github.io/BDBVHubRenewal/backtest/), [Held-out evaluation](https://seabbs.github.io/BDBVHubRenewal/heldout/)).
- Checks how well each province is predicted from the others by leaving one province out.
- Writes and checks the hub's submission files and model metadata.
- Submits each week through a gated workflow ([Submission workflow](https://seabbs.github.io/BDBVHubRenewal/submission/workflow/)).
  It fits the main model, refits it once if it has not converged, and falls back to a simpler variant and then to persistence.
  Each candidate is checked for convergence, posterior predictive fit and file validity, and the result is opened as a pull request.
- Publishes the latest nowcast and forecast ([Nowcast](https://seabbs.github.io/BDBVHubRenewal/nowcast/)).

Docs: <https://seabbs.github.io/BDBVHubRenewal/>

## Run it

You need Python 3.13 or later, [uv](https://docs.astral.sh/uv/) and [Task](https://taskfile.dev).
`task validate-hub` also needs R with `hubValidations`.

```bash
task setup             # install the package, PyRenew and the dev tools
task test              # run the tests (task test-slow runs small model fits)
task lint              # ruff and mypy
task data              # download every hub vintage and rebuild data/
task backtest          # development backtest and recovery check
task backtest-lopo     # leave one province out
task backtest-heldout  # score held-out vintages under the chosen model
task nowcast           # forecast from the latest vintage and write a hub file
task ci-fit            # fit and forecast one config as a CI candidate does
task metadata          # write the model metadata YAML
task validate-hub      # check both with hub_validate and hubValidations (R)
task ci-test           # test the submission CI scripts in ci/
task ci-dry            # dry run of the submission workflow on fixtures
task submission-preflight  # list what a live submission still needs
task docs              # build the docs site into _site/
```

The tidy data in `data/` and the backtest results in `results/` are committed, so the docs build from a clean clone without refitting.
The [Getting started](https://seabbs.github.io/BDBVHubRenewal/getting-started/) page walks through a small fit.

## Layout

```
.github/         CI, docs and the submission workflow (submit.yml)
ci/              submission CI: fallback chain, gate checks, PR body
config/          backtest and hub comparison settings
data/            tidy hub data (committed); data/raw/ is not
docs/            MkDocs site; figures are built at docs build time,
                 the model diagram by scripts/build_diagrams.sh
hub/             model metadata YAML
results/         score tables and the model choice record
scripts/         hubValidations runner (R), diagram build
src/bdbv_hub_renewal/
  data/          hub data, dates, reconstructed and simulated vintages,
                 delays and BVDOutbreakSize releases
  model/         the PyRenew renewal model (JAX)
  forecasters/   the forecaster interface, registry, persistence and the
                 renewal variants
  evaluation/    scoring, backtest, held-out evaluation, PPC, hub comparison
  submission/    hub file writer and checks, metadata, CI fit configs, rounds
  reporting/     figures and docs tables
  cli.py         the bdbv-hub-renewal command
tests/           mirrors src/
```

## Data credit

Surveillance data: Institut National de Santé Publique (INSP) and the Incident Management Support Team (IMST), Democratic Republic of the Congo.
Processed by the Atlantic Coast Center for Infectious Disease Dynamics and Analytics (ACCIDDA) and the Kinshasa School of Public Health, University of Kinshasa, and distributed through the BDBV Modeling Hub, <https://github.com/InsightNet-US/BDBV-Modeling-Hub>, accessed 2026-10-27.

## Citation and licence

To cite this software, use [CITATION.cff](CITATION.cff), or the "Cite this repository" button on GitHub.
The code is under the MIT licence, see [LICENSE](LICENSE).
