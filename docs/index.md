# ![bdbv-hub-modelling](assets/logo.svg#only-light){ .title-logo width="336" height="44" } ![](assets/logo-dark.svg#only-dark){ .title-logo width="336" height="44" }

Nowcasts and forecasts of confirmed Bundibugyo virus disease cases for the BDBV Modeling Hub.
{ .page-subtitle }

The [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub) collects weekly nowcasts and forecasts of confirmed Bundibugyo virus disease cases in the Democratic Republic of the Congo.
This repository holds the models we submit to it, experimental models related to it, their backtests, and their evaluation on other case studies.
The random walk and the random walk nowcast are the baselines we evaluate every other model against, and give the hub a principled baseline.

The fine-tuned foundation project fine-tunes the Chronos-2 forecasting model on simulated outbreaks so that it can nowcast from reporting triangles ([paper plan](https://github.com/seabbs/bdbv-hub-modelling/issues/206)).
The [paper](paper/main.html) and its [supporting information](paper/si.html) describe it.

The joint-model-informed renewal project compares renewal models fitted on their own with renewal models informed by a larger joint model, BVDOutbreakSize, through approximate Markov melding.

## Quick links

<div class="quick-links" markdown>
[![](assets/logos/fine-tuned-foundation/logo-mark.svg#only-light){ width="96" height="96" }![](assets/logos/fine-tuned-foundation/logo-mark-dark.svg#only-dark){ width="96" height="96" } Fine-tuned foundation](foundation/index.md){ .quick-link }
[![](assets/logos/joint-model-informed-renewal/logo-mark.svg#only-light){ width="96" height="96" }![](assets/logos/joint-model-informed-renewal/logo-mark-dark.svg#only-dark){ width="96" height="96" } Joint-model-informed renewal](renewal/index.md){ .quick-link }
[![](assets/logos/hub-submission/logo-mark.svg#only-light){ width="96" height="96" }![](assets/logos/hub-submission/logo-mark-dark.svg#only-dark){ width="96" height="96" } Hub submission](hub/index.md){ .quick-link }
[![](assets/logos/nowcastutils/logo-mark.svg#only-light){ width="96" height="96" }![](assets/logos/nowcastutils/logo-mark-dark.svg#only-dark){ width="96" height="96" } nowcastutils](reference/data.md){ .quick-link }
[![](assets/logos/nowcasteval/logo-mark.svg#only-light){ width="96" height="96" }![](assets/logos/nowcasteval/logo-mark-dark.svg#only-dark){ width="96" height="96" } nowcasteval](methods/evaluation.md){ .quick-link }
[![](assets/logos/nowcastworkflow/logo-mark.svg#only-light){ width="96" height="96" }![](assets/logos/nowcastworkflow/logo-mark-dark.svg#only-dark){ width="96" height="96" } nowcastworkflow](development/cli.md){ .quick-link }
[![](assets/logos/nowcastbaselines/logo-mark.svg#only-light){ width="96" height="96" }![](assets/logos/nowcastbaselines/logo-mark-dark.svg#only-dark){ width="96" height="96" } nowcastbaselines](methods/baselines.md){ .quick-link }
[![](assets/logos/nowcastfoundation/logo-mark.svg#only-light){ width="96" height="96" }![](assets/logos/nowcastfoundation/logo-mark-dark.svg#only-dark){ width="96" height="96" } nowcastfoundation](methods/foundation.md){ .quick-link }
[![](assets/logos/nowcastrenewal/logo-mark.svg#only-light){ width="96" height="96" }![](assets/logos/nowcastrenewal/logo-mark-dark.svg#only-dark){ width="96" height="96" } nowcastrenewal](methods/model.md){ .quick-link }
[![](assets/logos/bdbvhub/logo-mark.svg#only-light){ width="96" height="96" }![](assets/logos/bdbvhub/logo-mark-dark.svg#only-dark){ width="96" height="96" } bdbvhub](submission/workflow.md){ .quick-link }
</div>

<div class="grid cards" markdown>

-   [![](assets/logos/fine-tuned-foundation/logo-mark.svg){ width="32" height="32" } **Fine-tuned foundation**](foundation/index.md)

    Chronos-2 fine-tuned on simulated outbreaks, the paper, its baselines and the German benchmark.

-   [![](assets/logos/joint-model-informed-renewal/logo-mark.svg){ width="32" height="32" } **Joint-model-informed renewal**](renewal/index.md)

    Renewal models on their own and informed by BVDOutbreakSize.

-   [![](assets/logos/hub-submission/logo-mark.svg){ width="32" height="32" } **Hub submission**](hub/index.md)

    The current nowcast, the backtest, the comparison with the hub and the submission pipeline.

</div>

## Data credit

Surveillance data: Institut National de Santé Publique (INSP) and the Incident Management Support Team (IMST), Democratic Republic of the Congo.
Processed by ACCIDDA and the Kinshasa School of Public Health, University of Kinshasa, and distributed through the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
