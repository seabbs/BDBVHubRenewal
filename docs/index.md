# ![BDBVHubRenewal](assets/logo.svg#only-light){ .title-logo width="275" height="44" } ![](assets/logo-dark.svg#only-dark){ .title-logo width="275" height="44" }

Nowcasts and forecasts of confirmed Bundibugyo virus disease cases for the BDBV Modeling Hub.
{ .page-subtitle }

The [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub) target is weekly confirmed cases by symptom-onset week, nationally, by province and by health zone, as 100 joint sample trajectories.
Two hub models are submitted, `seabbs-rwnowcast` and `seabbs-renewal`.

## Models

`randomwalknowcast` is baselinenowcast ([Johnson et al. 2025](https://doi.org/10.12688/wellcomeopenres.25027.2), Wellcome Open Research) with a random walk on top, and `randomwalk` is the same walk with no nowcast ([Baseline models](methods/baselines.md)).
`pprandomwalknowcast` is a generative variant that fits the delay and the walk jointly, partially pooled over places, and is not submitted.
The renewal models (`renewal`, `renewalsp` and `renewalbvd`) are built on [PyRenew](https://github.com/CDCgov/PyRenew) and fit the hub's known and imputed onsets ([Renewal models](methods/model.md)).
Each week `randomwalknowcast` is submitted as `seabbs-rwnowcast` and the first renewal model to pass its checks as `seabbs-renewal` ([Submission workflow](submission/workflow.md)).

## Pages

| Section | What it holds |
|---|---|
| [Setup](setup.md) | Install the package and run it from a script, a REPL or a notebook. |
| [Getting started](getting-started.ipynb) | Forecast with the two baselines, load a stored renewal fit, and plot and score them in Python. |
| Methods | [Data](methods/data.md), [Baseline models](methods/baselines.md), [Renewal models](methods/model.md), [Evaluation](methods/evaluation.md) and [Implementation](methods/implementation.md). |
| Results | [Data](data.md), [nowcast](nowcast.md), [backtest](backtest.md), [simulation checks](simulation.md) and [comparison with hub forecasts](comparison.md). |
| Submission | The [workflow](submission/workflow.md), [hub specification](submission/hub-spec.md), [opting in](submission/opting-in.md) and [submission log](submission/rounds/index.md). |
| Development | The [model building](model-development/model-building.md) ladder, the [development log](model-development/log.md), [contributing](development/contributing.md), [adding a model](development/adding-a-model.md) and the [CLI](development/cli.md). |
| [API reference](reference/index.md) | The Python package. |

## Data credit

Surveillance data: Institut National de Santé Publique (INSP) and the Incident Management Support Team (IMST), Democratic Republic of the Congo.
Processed by ACCIDDA and the Kinshasa School of Public Health, University of Kinshasa, and distributed through the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
