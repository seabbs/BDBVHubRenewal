# Held-out evaluation

Forecasts from vintages after each model choice, scored as later data arrive.
{ .page-subtitle }

The development set rolls forward with the model choices in `results/model-choice.yml`.
Development runs use every vintage up to the latest choice date.
Each real vintage after a choice is held out, forecast by the model chosen then and by persistence, and scored against every later vintage as reported.
With no choice recorded, nothing is held out.
The design is on the [Evaluation methods page](methods/evaluation.md#rolling-held-out-evaluation).

--8<-- "heldout-choices.md"

--8<-- "heldout-status.md"

--8<-- "heldout-scores.md"

--8<-- "heldout-forecasts.md"
