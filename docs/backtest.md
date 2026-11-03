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

=== "WIS"

    ![Mean WIS by horizon and vintage kind.](figures/scores-wis-horizon.svg#only-light)
    ![Mean WIS by horizon and vintage kind.](figures/scores-wis-horizon-dark.svg#only-dark)

=== "Log WIS"

    ![Mean log-scale WIS by horizon and vintage kind.](figures/scores-log-wis-horizon.svg#only-light)
    ![Mean log-scale WIS by horizon and vintage kind.](figures/scores-log-wis-horizon-dark.svg#only-dark)

=== "90% coverage"

    ![90% interval coverage by horizon and vintage kind.](figures/scores-coverage-90-horizon.svg#only-light)
    ![90% interval coverage by horizon and vintage kind.](figures/scores-coverage-90-horizon-dark.svg#only-dark)

=== "Bias"

    ![Mean bias by horizon and vintage kind.](figures/scores-bias-horizon.svg#only-light)
    ![Mean bias by horizon and vintage kind.](figures/scores-bias-horizon-dark.svg#only-dark)

--8<-- "backtest-summary.md"

## By province

Scores on the count each later release reported.

=== "WIS"

    ![Mean WIS by province and vintage kind.](figures/scores-wis-location.svg#only-light)
    ![Mean WIS by province and vintage kind.](figures/scores-wis-location-dark.svg#only-dark)

=== "Log WIS"

    ![Mean log-scale WIS by province and vintage kind.](figures/scores-log-wis-location.svg#only-light)
    ![Mean log-scale WIS by province and vintage kind.](figures/scores-log-wis-location-dark.svg#only-dark)

=== "90% coverage"

    ![90% interval coverage by province and vintage kind.](figures/scores-coverage-90-location.svg#only-light)
    ![90% interval coverage by province and vintage kind.](figures/scores-coverage-90-location-dark.svg#only-dark)

=== "Bias"

    ![Mean bias by province and vintage kind.](figures/scores-bias-location.svg#only-light)
    ![Mean bias by province and vintage kind.](figures/scores-bias-location-dark.svg#only-dark)

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
