# Backtest

Forecasts made at past vintages, scored on what each later release reported.
{ .page-subtitle }

Every model is scored on the count each later release reports, and complete weeks are a secondary check ([Evaluation](methods/evaluation.md#scoring)).
Weeks still being reported are shaded, with hollow points.

--8<-- "release-headline.md"

--8<-- "backtest-findings.md"

--8<-- "backtest-forecasts.md"

--8<-- "release-nowcasts.md"

## By horizon

Scores on the count each later release reported.
Horizons −4 to −1 are nowcasts, 0 is the reference week and 1 to 3 are forecasts.

--8<-- "scorechart-horizon.md"

--8<-- "backtest-summary.md"

## By province

Scores on the count each later release reported.

--8<-- "scorechart-location.md"

??? note "Tables"

    --8<-- "scores-location.md"

--8<-- "backtest-sampler.md"

## Scores by release

Each origin's forecasts scored on the counts each later release reported.

![Mean WIS of each origin's forecasts against each later release.](figures/scores-by-truth.svg#only-light)
![Mean WIS of each origin's forecasts against each later release.](figures/scores-by-truth-dark.svg#only-dark)

??? note "Tables"

    --8<-- "scores-by-truth.md"

??? note "Truth changes"

    --8<-- "truth-changes.md"

--8<-- "lopo-section.md"

--8<-- "zone-section.md"
