# Backtest

Forecasts made at past vintages, scored against later data.
{ .page-subtitle }

!!! info "A thin backtest"
    There are four real vintages, and only two give scorable forecasts in development: 2026-08-18 (reference date 2026-08-22) and 2026-10-09 (2026-10-17).
    Vintages after the model choice date, 2026-10-10, are held out and scored on the [Held-out evaluation](heldout.md) page.
    Weekly reconstructed vintages from 2026-06-06 to 2026-10-10 add more forecast dates, but they are built from the same latest data.
    Read these results as a check of the harness, not as evidence about models.

This page shows development results only.
Scores on complete weeks are the primary scores, with every week as reported and vintage-matched scores beside them.
How scoring works is on the [Evaluation methods page](methods/evaluation.md).
`task backtest` writes `results/backtest-scores.csv` (development, against the latest vintage), `results/backtest-scores-by-truth.csv` (real forecasts against every later vintage) and `results/recovery-scores.csv` (simulated).

## Overall

--8<-- "backtest-truth.md"

--8<-- "scores-overall.md"

## Persistence bias

--8<-- "persistence-bias.md"

This is what a flat baseline should do here.
Persistence copies the weeks it judges complete as published, and those weeks are still revised upwards after the vintage.
It also holds the last complete week flat, while the epidemic was growing for most of the period.
Both push its forecasts below the truth.
The maths is on the [Model page](methods/model.md#persistence-baseline).

## Forecasts against later data

Each panel is one reference date.
The dashed line is the data at the time, the solid line the latest data, the faint lines are the 100 sample trajectories, and the bands are 50% and 90% intervals around the median.

=== "National, real"

    ![Persistence forecasts against later data, national, real vintages.](figures/forecasts-real-cd.svg#only-light)
    ![Persistence forecasts against later data, national, real vintages.](figures/forecasts-real-cd-dark.svg#only-dark)

=== "Ituri, real"

    ![Persistence forecasts against later data, Ituri, real vintages.](figures/forecasts-real-cd-it.svg#only-light)
    ![Persistence forecasts against later data, Ituri, real vintages.](figures/forecasts-real-cd-it-dark.svg#only-dark)

=== "Nord Kivu, real"

    ![Persistence forecasts against later data, Nord Kivu, real vintages.](figures/forecasts-real-cd-nk.svg#only-light)
    ![Persistence forecasts against later data, Nord Kivu, real vintages.](figures/forecasts-real-cd-nk-dark.svg#only-dark)

=== "National, reconstructed"

    ![Persistence forecasts against later data, national, reconstructed vintages.](figures/forecasts-reconstructed-cd.svg#only-light)
    ![Persistence forecasts against later data, national, reconstructed vintages.](figures/forecasts-reconstructed-cd-dark.svg#only-dark)

=== "Ituri, reconstructed"

    ![Persistence forecasts against later data, Ituri, reconstructed vintages.](figures/forecasts-reconstructed-cd-it.svg#only-light)
    ![Persistence forecasts against later data, Ituri, reconstructed vintages.](figures/forecasts-reconstructed-cd-it-dark.svg#only-dark)

=== "Nord Kivu, reconstructed"

    ![Persistence forecasts against later data, Nord Kivu, reconstructed vintages.](figures/forecasts-reconstructed-cd-nk.svg#only-light)
    ![Persistence forecasts against later data, Nord Kivu, reconstructed vintages.](figures/forecasts-reconstructed-cd-nk-dark.svg#only-dark)

## By horizon

Horizons −4 to −1 are nowcasts of past weeks, 0 is the reference week and 1 to 3 are forecasts.
Panels split by vintage kind.

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

Persistence underpredicts the weeks it treats as complete, about horizons −4 to −2.
Its intervals cover better beyond the reference week, because the random walk spread grows quickly.

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

??? note "Scores by province"

    --8<-- "scores-location.md"

--8<-- "backtest-sampler.md"

## Relative skill

--8<-- "relative-section.md"

## Scores by truth vintage

Forecasts from real vintages are scored against every later real vintage, as reported at each.
The national truth for the weeks they cover:

--8<-- "truth-changes.md"

<figure class="doc-figure" markdown="span">
  ![Mean WIS of each real forecast vintage against each later truth vintage.](figures/scores-by-truth.svg#only-light)
  ![Mean WIS of each real forecast vintage against each later truth vintage.](figures/scores-by-truth-dark.svg#only-dark)
  <figcaption markdown="1">
    Mean WIS of the forecasts from each real vintage, scored against each later real vintage as reported.
  </figcaption>
</figure>

??? note "Scores by truth vintage"

    --8<-- "scores-by-truth.md"

--8<-- "lopo-section.md"

## Recovery on simulated vintages

Forecasts from vintages of a [simulated outbreak](methods/data.md#simulated-vintages) with a turning point, scored against the known final counts.

<figure class="doc-figure" markdown="span">
  ![True parameters of the simulated outbreak.](figures/simulation.svg#only-light)
  ![True parameters of the simulated outbreak.](figures/simulation-dark.svg#only-dark)
  <figcaption markdown="1">
    True parameters of the simulated outbreak: onsets, Rt, susceptible fraction, ascertainment, missing-onset share and mean delay.
  </figcaption>
</figure>

<figure class="doc-figure" markdown="span">
  ![Mean WIS by horizon on simulated vintages.](figures/recovery-wis-horizon.svg#only-light)
  ![Mean WIS by horizon on simulated vintages.](figures/recovery-wis-horizon-dark.svg#only-dark)
  <figcaption markdown="1">
    Mean WIS by horizon on simulated vintages.
    Rt recovery against the true values is on the [Model building](development/model-building.md) page.
  </figcaption>
</figure>

--8<-- "nowcast-recovery.md"
