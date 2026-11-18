# ![bdbv-hub-modelling](assets/logo.svg#only-light){ .title-logo width="336" height="44" } ![](assets/logo-dark.svg#only-dark){ .title-logo width="336" height="44" }

Nowcasts and forecasts of confirmed Bundibugyo virus disease cases for the BDBV Modeling Hub.
{ .page-subtitle }

The [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub) collects weekly nowcasts and forecasts of confirmed Bundibugyo virus disease cases in the Democratic Republic of the Congo.
This repository holds the models we submit to it, experimental models related to it, their backtests, and their evaluation on other case studies.
The random walk and the random walk nowcast are the baselines we evaluate every other model against, and give the hub a principled baseline.

The fine-tuned foundation project fine-tunes the Chronos-2 forecasting model on simulated outbreaks so that it can nowcast from reporting triangles ([paper plan](https://github.com/seabbs/bdbv-hub-modelling/issues/206)).
The [paper](paper/main.html) and its [supporting information](paper/si.html) describe it.

The joint-model-informed renewal project compares renewal models fitted on their own with renewal models informed by a larger joint model, BVDOutbreakSize, through approximate Markov melding.

**Quick links:** [![bdbv-hub-modelling](assets/logos/bdbv-hub-modelling/logo-mark.svg){ width="32" height="32" title="bdbv-hub-modelling" }](index.md) [![Fine-tuned foundation](assets/logos/fine-tuned-foundation/logo-mark.svg){ width="32" height="32" title="Fine-tuned foundation" }](foundation/index.md) [![Joint-model-informed renewal](assets/logos/joint-model-informed-renewal/logo-mark.svg){ width="32" height="32" title="Joint-model-informed renewal" }](renewal/index.md) [![Hub submission](assets/logos/hub-submission/logo-mark.svg){ width="32" height="32" title="Hub submission" }](hub/index.md)

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
