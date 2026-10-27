# ![BDBV hub forecasts](assets/logo.svg#only-light){ .title-logo width="248" height="44" } ![](assets/logo-dark.svg#only-dark){ .title-logo width="248" height="44" }

Nowcasts and forecasts of confirmed Bundibugyo virus disease cases for the BDBV Modeling Hub.
{ .page-subtitle }

This repository builds the foundations for weekly submissions to the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
The hub collects nowcasts and forecasts of weekly confirmed cases by onset week, by province and nationally, for the 2026 outbreak in the Democratic Republic of the Congo.
The plan is in [EpidemicDates issue #3](https://github.com/seabbs/EpidemicDates/issues/3).

So far it has:

- a data layer that downloads and tidies every stream the hub publishes, with every vintage;
- reconstructed and simulated vintages to extend testing;
- a persistence baseline and a model registry;
- a writer and checks for the hub's submission format and model metadata;
- a backtest scored like [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize), a rolling held-out evaluation and a comparison with hub forecasts;
- the maths of the planned renewal model.

## Pages

| Section | What it holds |
|---|---|
| [Getting started](getting-started.md) | Install, run and build the site. |
| Methods | [Data](methods/data.md), [Model](methods/model.md) and [Evaluation](methods/evaluation.md), written as the methods of a paper. |
| [Data](data.md) | What the hub data show, vintage by vintage. |
| [Backtest](backtest.md) | Development scores against later data. |
| [Held-out evaluation](heldout.md) | Forecasts after the model choice, scored as later data arrive. |
| [Nowcast](nowcast.md) | The current nowcast and forecast from the latest vintage. |
| Development | [Adding a model](development/adding-a-model.md), the [CLI](development/cli.md), [CI and submissions](development/submissions.md) and the [hub specification](hub-spec.md). |
| [API reference](reference.md) | The Python package. |

## Data credit

Surveillance data: Institut National de Santé Publique (INSP) and the Incident Management Support Team (IMST), Democratic Republic of the Congo.
Processed by ACCIDDA and the Kinshasa School of Public Health, University of Kinshasa, and distributed through the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
