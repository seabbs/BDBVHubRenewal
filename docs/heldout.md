# Held-out evaluation

Forecasts from vintages after the model choice, scored as later data arrive.
{ .page-subtitle }

Every real vintage after the first date in `results/model-choice.yml` is held out.
Each held-out forecast is made by the model chosen at the time and scored against every later vintage, as reported then.
Persistence is scored alongside as the baseline.
The design is on the [Evaluation methods page](methods/evaluation.md#rolling-held-out-evaluation).

## Model choices

--8<-- "heldout-choices.md"

## Status

--8<-- "heldout-status.md"

## Scores

--8<-- "heldout-scores.md"

!!! note "Recent weeks against early truth"
    Every week is scored, so the latest weeks of a forecast are scored against a truth that is still filling in.
    A forecast of the final count looks too high against those weeks until later vintages arrive.
    Expect the scores at horizons −1 and 0 to change as the forecast is re-scored.

## Forecasts against later data

Each panel is one held-out vintage, labelled by its reference date.
The dashed line is the data at the time and the solid line the latest data.
The faint lines are the 100 sample trajectories, and the bands are 50% and 90% intervals around the median.

--8<-- "heldout-forecasts.md"
