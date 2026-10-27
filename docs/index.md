# ![BDBVHubRenewal](assets/logo.svg#only-light){ .title-logo width="275" height="44" } ![](assets/logo-dark.svg#only-dark){ .title-logo width="275" height="44" }

Nowcasts and forecasts of confirmed Bundibugyo virus disease cases for the BDBV Modeling Hub.
{ .page-subtitle }

Weekly nowcasts and forecasts of confirmed Bundibugyo virus disease cases for the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
The hub's target is weekly confirmed cases by symptom-onset week, by province and nationally, as 100 joint sample trajectories per location.
Forecasts are submitted as `epiforecasts-hubrenewal`.

## Models

All models are renewal processes built on [PyRenew](https://github.com/CDCgov/PyRenew) and fitted to each data vintage ([Model](methods/model.md)).
Each fits the hub's known and imputed onsets together, with time-varying delays, missingness and ascertainment, and models the hub's imputation as the hub makes it.
National forecasts are the sum of the province trajectories.

| Model | What it is |
|---|---|
| `hubrenewal` | Spatial and partially pooled. Each province with cases has its own renewal process. The reproduction numbers share a national trend with pooled province deviations. Provinces import cases from each other through a gravity kernel and deplete their susceptibles. |
| `hubrenewalsp` | Single patch. One national renewal process, with each national trajectory split across provinces by their recent share of onsets. |
| `hubrenewalbvd` | The spatial model informed by the latest posterior estimates of the [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize) model. |
| `persistence` | The baseline: the last complete week carried forward. |

## What the repository does

- Downloads and tidies every data stream the hub publishes, with every vintage, and builds reconstructed and simulated vintages for testing ([Data](methods/data.md)).
- Scores every model against persistence on past vintages, then on every vintage after the model choice as a rolling held-out evaluation, re-scored as later data arrive ([Evaluation](methods/evaluation.md), [Backtest](backtest.md), [Held-out evaluation](heldout.md)).
- Checks how well each province is predicted from the others by leaving one province out.
- Writes and checks the hub's submission files and model metadata.
- Submits each week through a gated workflow ([How submissions work](submission/workflow.md)).
  It fits the main model, refits it once if it has not converged, and falls back to a simpler variant and then to persistence.
  Each candidate is checked for convergence, posterior predictive fit and file validity, and the result is opened as a pull request.
- Publishes the latest nowcast and forecast ([Nowcast](nowcast.md)).

## Pages

| Section | What it holds |
|---|---|
| [Getting started](getting-started.md) | Install, fit the model and write a hub file. |
| Methods | [Data](methods/data.md), [Model](methods/model.md) and [Evaluation](methods/evaluation.md), written as the methods of a paper. |
| [Data](data.md) | What the hub data show, vintage by vintage. |
| [Backtest](backtest.md) | Development scores against later data. |
| [Held-out evaluation](heldout.md) | Forecasts after the model choice, scored as later data arrive. |
| [Nowcast](nowcast.md) | The current nowcast and forecast from the latest vintage. |
| Submission | [How submissions work](submission/workflow.md), the [hub specification](submission/hub-spec.md), [Setup](submission/setup.md) and the [submitted rounds](submission/rounds/index.md). |
| Development | [Adding a model](development/adding-a-model.md) and the [CLI](development/cli.md). |
| [API reference](reference.md) | The Python package. |

## Data credit

Surveillance data: Institut National de Santé Publique (INSP) and the Incident Management Support Team (IMST), Democratic Republic of the Congo.
Processed by ACCIDDA and the Kinshasa School of Public Health, University of Kinshasa, and distributed through the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
