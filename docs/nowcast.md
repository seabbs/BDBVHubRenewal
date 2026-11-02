# Nowcast

The current nowcast and forecast from the latest vintage.
{ .page-subtitle }

--8<-- "nowcast-meta.md"

--8<-- "latest-submission.md"

Each model is shown from its latest origin.
Lines are the data of the latest vintage (solid), with 100 sample trajectories, 50% and 90% bands and the median for each model.
The dotted line is the reference date and weeks still being reported are shaded, with hollow points.
The vertical axis ends at the top of the 90% band.

## Nowcast

The weeks before the reference date, estimated for reporting delays.

--8<-- "nowcast-only-tabs.md"

## Nowcast and forecast

The same nowcast with the reference week and the weeks after it.

--8<-- "nowcast-tabs.md"

## Delay and spread

The growth to final of each onset day by its age at the latest release, and the Skellam spread by weeks back.
A place using its parent's estimate is labelled with the parent.

![Growth to final by age and the Skellam spread by weeks back at the latest release.](figures/delay-check.svg#only-light)
![Growth to final by age and the Skellam spread by weeks back at the latest release.](figures/delay-check-dark.svg#only-dark)

The submission pull request embeds [`plot_nowcast`][bdbvhubrenewal.reporting.plots.plot_nowcast].
