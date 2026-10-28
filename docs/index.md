# ![BDBVHubRenewal](assets/logo.svg#only-light){ .title-logo width="275" height="44" } ![](assets/logo-dark.svg#only-dark){ .title-logo width="275" height="44" }

Nowcasts and forecasts of confirmed Bundibugyo virus disease cases for the BDBV Modeling Hub.
{ .page-subtitle }

The [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub) target is weekly confirmed cases by symptom-onset week, by province and nationally, as 100 joint sample trajectories.
Forecasts are submitted as `epiforecasts-seabbsrenewal`.

## Models

The renewal models are built on [PyRenew](https://github.com/CDCgov/PyRenew) and fit the hub's known and imputed onsets, modelling the hub's imputation ([Model](methods/model.md)).

| Model | What it is |
|---|---|
| `renewal` | One renewal per province, with pooled Rt, gravity importation and depletion. |
| `renewalsp` | One renewal per location, the nation included, fitted on its own. |
| `renewalbvd` | `renewal` informed by the latest [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize) Rt. |
| `persistence` | The baseline: the last complete week carried forward. |

Each week the [submission](submission/workflow.md) takes the first of `renewal`, one refit of it, `renewalsp` and `persistence` that passes its checks.

## Pages

| Section | What it holds |
|---|---|
| [Getting started](getting-started.md) | Install, fit the model and write a hub file. |
| Methods | [Data](methods/data.md), [Model](methods/model.md), [Evaluation](methods/evaluation.md) and [Implementation](methods/implementation.md). |
| [Data](data.md) | The hub data, vintage by vintage. |
| [Backtest](backtest.md) | Development scores against later data. |
| [Held-out evaluation](heldout.md) | Forecasts after the model choice, scored as later data arrive. |
| [Nowcast](nowcast.md) | The latest nowcast and forecast. |
| Submission | The [workflow](submission/workflow.md), [hub specification](submission/hub-spec.md), [setup](submission/setup.md) and [rounds](submission/rounds/index.md). |
| Model development | The [model building](model-development/model-building.md) ladder. |
| Development | [Contributing](development/contributing.md), [adding a model](development/adding-a-model.md) and the [CLI](development/cli.md). |
| [API reference](reference/index.md) | The Python package. |

## Data credit

Surveillance data: Institut National de Santé Publique (INSP) and the Incident Management Support Team (IMST), Democratic Republic of the Congo.
Processed by ACCIDDA and the Kinshasa School of Public Health, University of Kinshasa, and distributed through the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
