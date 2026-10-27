# Comparison with hub forecasts

Our forecasts and those submitted to the hub, scored on the same truth and weeks.
{ .page-subtitle }

!!! warning "Not published"
    This page is built but kept off the site until publication is decided.
    `exclude_docs` in `mkdocs.yml` leaves it and its figures out.
    The display mode in `config/hub-comparison.toml` sets how other teams are shown.

--8<-- "comparison-display.md"

`bdbv-hub-renewal compare-hub --fetch` downloads the hub's `model-output/` and `model-metadata/` files for the models in `config/hub-comparison.toml`.
For each hub round, our registered models forecast from the matching vintage.
That is a real vintage where one serves the round, and otherwise a reconstructed one.
Our forecasts are labelled "retrospective" with the kind of vintage they used.
Every model is scored on the same truth, the latest real vintage as reported, and on the locations and weeks all of them covered.
The method is on the [Evaluation methods page](methods/evaluation.md#comparison-with-hub-forecasts).

## Teams and licences

Each team's licence is taken from its hub metadata.

--8<-- "comparison-teams.md"

## Scores

--8<-- "comparison-scores.md"

Rounds that cannot be scored yet, because no later real vintage exists:

--8<-- "comparison-pending.md"

## Forecasts side by side

Each panel is one hub round.
The solid line is the latest data.
The faint lines are each model's sample trajectories, and the bands are 50% and 90% intervals around the median.

--8<-- "comparison-figures.md"

## Files from other rounds

The hub's first round collected estimates of the cumulative number of symptomatic cases in the country.
That is a different quantity from weekly confirmed cases by onset week, so these files are listed but not scored.

--8<-- "comparison-other.md"

## Data and model credit

The hub asks for this credit wherever its data are used:

--8<-- "comparison-attribution.md"

Model outputs are credited to the teams above, as described in their metadata files.
