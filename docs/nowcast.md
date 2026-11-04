# Nowcast

The current nowcast and forecast from the latest vintage.
{ .page-subtitle }

--8<-- "nowcast-meta.md"

--8<-- "latest-submission.md"

Each model is shown from its latest origin ([reading the figures](methods/evaluation.md#reading-the-figures)).

## Nowcast

The weeks before the reference date, estimated for reporting delays.

--8<-- "nowcast-only-tabs.md"

## Nowcast and forecast

The same nowcast with the reference week and the weeks after it.

--8<-- "nowcast-tabs.md"

The submission pull request embeds [`plot_nowcast`][bdbvhubrenewal.reporting.plots.plot_nowcast].

--8<-- "zone-nowcast.md"

## Delay and spread

The growth to final of each onset day by its age at the latest release, and the observation model by weeks back.
A place using its parent's estimate is labelled with the parent.
How both are fitted is on the [Baseline models](methods/baselines.md#nowcast) page.

![Growth to final by age, and the dispersion of new reports and the removal rate by weeks back, at the latest release.](figures/delay-check.svg#only-light)
![Growth to final by age, and the dispersion of new reports and the removal rate by weeks back, at the latest release.](figures/delay-check-dark.svg#only-dark)
