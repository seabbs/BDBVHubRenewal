# Backtest

Forecasts made at past vintages, scored against the latest data.
{ .page-subtitle }

Scores on complete weeks are the primary scores ([Evaluation](methods/evaluation.md#scoring)).
Weeks still being reported are shaded, with hollow points.

--8<-- "backtest-findings.md"

--8<-- "backtest-summary.md"

## Overall

--8<-- "backtest-truth.md"

--8<-- "scores-overall.md"

--8<-- "relative-section.md"

--8<-- "backtest-forecasts.md"

## By horizon

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

## By province

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

## Scores by truth vintage

Real forecasts scored against each later real vintage, as reported at each.

![Mean WIS of each real forecast vintage against each later truth vintage.](figures/scores-by-truth.svg#only-light)
![Mean WIS of each real forecast vintage against each later truth vintage.](figures/scores-by-truth-dark.svg#only-dark)

??? note "Tables"

    --8<-- "scores-by-truth.md"

??? note "Truth changes"

    --8<-- "truth-changes.md"

--8<-- "lopo-section.md"

--8<-- "recovery-section.md"

--8<-- "nowcast-recovery.md"

## Simulated scenarios

Three scenarios copy the real 2026-10-24 data and differ only in Rt after the data, which falls by a quarter, holds or rises by a quarter.
Forecasts are made from the four weekly simulated vintages up to 2026-10-24 and scored on complete weeks of the final simulated data.

--8<-- "scenario-figure.md"

??? note "Tables"

    --8<-- "scenario-scores.md"

Zone forecasts split each model's province samples across the zones of each province, and are compared with zone persistence.
