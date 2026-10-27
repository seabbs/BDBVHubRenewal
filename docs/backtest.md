# Backtest

Forecasts made at past vintages, scored against later data.
{ .page-subtitle }

!!! info "A thin backtest"
    There are four real vintages, and only two give scorable forecasts in development: 2026-08-18 (reference date 2026-08-22) and 2026-10-09 (2026-10-17).
    The 2026-10-24 vintage is the truth, and 2026-10-17 (2026-10-24) is held out.
    Weekly reconstructed vintages from 2026-06-06 to 2026-10-10 add more forecast dates, but they are built from the same latest data.
    Read these results as a check of the harness, not as evidence about models.

This page shows development results only.
Held-out reference dates are listed in `config/backtest.toml` and never scored here.
`task backtest` writes `results/backtest-scores.csv` (development) and `results/recovery-scores.csv` (simulated).

## Overall

--8<-- "scores-overall.md"

Relative to persistence:

--8<-- "relative-skill.md"

Only persistence exists so far, so every ratio is 1.
The [relative WIS](#relative-to-persistence) plots fill in as models are registered.

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

Persistence underpredicts the weeks it treats as complete (horizons −4 and −3).
Those weeks are still revised upwards after the vintage.
Its intervals cover well beyond the reference week, because the random walk spread grows quickly.

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

## Relative to persistence

=== "By horizon"

    ![WIS relative to persistence by horizon.](figures/relative-wis-horizon.svg#only-light)
    ![WIS relative to persistence by horizon.](figures/relative-wis-horizon-dark.svg#only-dark)

=== "By province"

    ![WIS relative to persistence by province.](figures/relative-wis-location.svg#only-light)
    ![WIS relative to persistence by province.](figures/relative-wis-location-dark.svg#only-dark)

## Recovery on simulated vintages

Forecasts from vintages of a [simulated outbreak](methods.md#simulated-vintages) with a turning point, scored against the known final counts.

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
    The model phase adds parameter recovery checks against the true values above.
  </figcaption>
</figure>

## Held-out evaluation

The held-out reference date is **2026-10-24** (vintage 2026-10-17).
It is the latest reference date that can be scored at all.
With one date and only weeks up to 2026-10-10 scorable, the held-out set is very thin.

The procedure:

1. Develop and compare variants on the development set only.
2. Write the model choice record `results/model-choice.yml` (template in `config/model-choice.template.yml`) and commit it.
3. Run `task backtest-heldout` once. It refuses without the record, uses real vintages only and logs the run.
4. Results go to `results/heldout-scores.csv` and `results/heldout-run.yml`, with the run date and the hash of the choice record.
5. Sam reviews them before any submission decision.

--8<-- "heldout.md"
