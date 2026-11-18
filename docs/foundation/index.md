# Fine-tuned foundation

We fine-tune the Chronos-2 forecasting model on outbreaks simulated with epinowcast, so that it can nowcast from a reporting triangle laid out as series.
The fine-tuned model forecasts daily and weekly series in one joint block, for the country, the provinces and the health zones.
The [paper](../paper.md) and its supporting information describe the project, and this section holds the pages that go with it.

- [Paper](../paper.md): the main text, the supporting information and their PDFs.
- [Foundation models](../methods/foundation.md): the layouts, the release lags and the fine-tuning.
- [Using the fine-tuned Chronos-2](../methods/finetuned-chronos.md): loading a released adapter.
- [Baseline models](../methods/baselines.md): the random walk and random walk nowcast every model is compared with.
- [German nowcast benchmark](../germany.md): the paper's retrospective case study.

Next: [Paper](../paper.md).
