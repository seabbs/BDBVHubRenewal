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
| Submission CI (`ci/`) | [![Submission CI tests](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/submission-ci-tests.yml/badge.svg)](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/submission-ci-tests.yml) | [![Submission CI tests](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/submission-ci-tests.yml/badge.svg)](https://github.com/seabbs/BDBVHubRenewal/actions/workflows/submission-ci-tests.yml) | [![ci coverage](https://codecov.io/gh/seabbs/BDBVHubRenewal/branch/main/graph/badge.svg?flag=ci)](https://codecov.io/gh/seabbs/BDBVHubRenewal?flags%5B0%5D=ci) |

Weekly nowcasts and forecasts of confirmed Bundibugyo virus disease cases for the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
The hub's target is weekly confirmed cases by symptom-onset week, by province and nationally, as 100 joint sample trajectories per location.
The model submits as `epiforecasts-hubrenewal`.

The model is a province renewal process built on [PyRenew](https://github.com/CDCgov/PyRenew).
Provinces share a pooled reproduction number, import cases from each other through a gravity kernel, and deplete their susceptibles.
It fits the hub's known and imputed onsets together, with time-varying delays, missingness and ascertainment, and models the hub's imputation as the hub makes it.
National forecasts are the sum of the province trajectories.

This repository holds:

- the model and its variants: `hubrenewal` (spatial, pooled), `hubrenewalsp` (single patch) and a fallback, with persistence as the baseline;
- a data layer for every hub data stream and vintage, plus reconstructed and simulated vintages;
- a backtest against persistence, a rolling held-out evaluation and a leave-one-province-out check, scored by WIS, coverage and bias;
- a writer and checks for the hub's submission files and model metadata;
- an automated, gated submission workflow with a fallback chain from the main model to persistence ([Submissions](https://seabbs.github.io/BDBVHubRenewal/submission/workflow/));
- a docs site with the methods, results and the latest nowcast.

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
scripts/         hubValidations runner (R), model sync, diagram build
src/bdbv_hub_renewal/
  data.py        download, tidy and load the hub data
  dates.py       epi weeks, reference dates and submission windows
  vintages.py    reconstructed vintages
  simulate.py    simulated outbreaks published as vintages
  models/        model interface, registry, persistence and the renewal
                 variants (renewal.py)
  pyrenew/       the PyRenew model (JAX), synced by scripts/sync_pyrenew.py
  delays.py      delay interface and the empirical estimator
  latent.py      renewal with importation and depletion (reference)
  observation.py known onsets and the hub's imputation (reference)
  hub.py         submission writer, checks and metadata
  hub_validate.py the hub's checks in Python (hub validation bundle)
  fitting.py     fit configs for the submission workflow
  ppc.py         posterior predictive check summaries and plot
  scoring.py     WIS, coverage, bias and relative skill
  backtest.py    rolling-origin backtest, scored against each later vintage
  heldout.py     model choice record and rolling held-out evaluation
  hubcompare.py  comparison with the forecasts submitted to the hub
  plots.py       figures, including plot_nowcast
  report.py      docs figures and tables
  cli.py         the bdbv-hub-renewal command
tests/
```

## Data credit

Surveillance data: Institut National de Santé Publique (INSP) and the Incident Management Support Team (IMST), Democratic Republic of the Congo.
Processed by the Atlantic Coast Center for Infectious Disease Dynamics and Analytics (ACCIDDA) and the Kinshasa School of Public Health, University of Kinshasa, and distributed through the BDBV Modeling Hub, <https://github.com/InsightNet-US/BDBV-Modeling-Hub>, accessed 2026-10-27.

## Citation and licence

To cite this software, use [CITATION.cff](CITATION.cff), or the "Cite this repository" button on GitHub.
The code is under the MIT licence, see [LICENSE](LICENSE).
