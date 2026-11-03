# ![BDBVHubRenewal](assets/logo.svg#only-light){ .title-logo width="275" height="44" } ![](assets/logo-dark.svg#only-dark){ .title-logo width="275" height="44" }

Nowcasts and forecasts of confirmed Bundibugyo virus disease cases for the BDBV Modeling Hub.
{ .page-subtitle }

The [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub) target is weekly confirmed cases by symptom-onset week, nationally, by province and by health zone, as 100 joint sample trajectories.
Two hub models are submitted, `seabbs-rwnowcast` and `seabbs-renewal`.

## Models

`randomwalknowcast` is baselinenowcast ([Johnson et al. 2025](https://doi.org/10.12688/wellcomeopenres.25027.2), Wellcome Open Research) with a random walk on top.
It nowcasts the hub's daily release snapshots with a JAX port of baselinenowcast 0.2.2 and carries each nowcast forward with a random walk.
It runs at province and zone level, and the national value is the sum of the provinces ([Baseline models](methods/baselines.md)).
`randomwalk` runs the same walk from the latest week judged complete, with no nowcast.
It is a comparison model and is not submitted.
The renewal models (`renewal`, `renewalsp` and `renewalbvd`) are built on [PyRenew](https://github.com/CDCgov/PyRenew) and fit the hub's known and imputed onsets, modelling the hub's imputation ([Renewal models](methods/model.md)).

Each week the [submission](submission/workflow.md) submits `randomwalknowcast` as `seabbs-rwnowcast`, and skips the round if it fails its checks.
It submits one renewal model as `seabbs-renewal`, taking the first in the chain in `ci/policy.toml` that passes its checks.

## Pages

| Section | What it holds |
|---|---|
| [Getting started](getting-started.ipynb) | Forecast, plot and score the two baselines in Python. |
| Methods | [Data](methods/data.md), [Baseline models](methods/baselines.md), [Renewal models](methods/model.md), [Evaluation](methods/evaluation.md) and [Implementation](methods/implementation.md). |
| Results | [Data](data.md), [nowcast](nowcast.md), [backtest](backtest.md), [held-out evaluation](heldout.md) and [comparison with hub forecasts](comparison.md). |
| Submission | The [workflow](submission/workflow.md), [hub specification](submission/hub-spec.md), [opting in](submission/opting-in.md) and [submission log](submission/rounds/index.md). |
| Development | The [model building](model-development/model-building.md) ladder, the [development log](model-development/log.md), [contributing](development/contributing.md), [adding a model](development/adding-a-model.md) and the [CLI](development/cli.md). |
| [API reference](reference/index.md) | The Python package. |

## Data credit

Surveillance data: Institut National de Santé Publique (INSP) and the Incident Management Support Team (IMST), Democratic Republic of the Congo.
Processed by ACCIDDA and the Kinshasa School of Public Health, University of Kinshasa, and distributed through the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
