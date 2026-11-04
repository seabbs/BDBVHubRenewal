# Backtest

Forecasts made at past vintages, scored on what each later release reported.
{ .page-subtitle }

--8<-- "release-headline.md"

## By horizon

By weeks from the reference week ([horizon](methods/evaluation.md#reading-the-figures)), at the scale chosen.

--8<-- "scoresummary-horizon.md"

--8<-- "backtest-summary.md"

## By place

Scores at each place, at the scale chosen.
Pick a place to narrow the table.

--8<-- "scoresummary-place.md"

??? note "Scores by target, scale and place, over every origin"

    --8<-- "scores-location.md"

--8<-- "zone-section.md"

--8<-- "backtest-forecasts.md"

## Findings

--8<-- "backtest-findings.md"

--8<-- "release-nowcasts.md"

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
