# Nowcast

The current nowcast and forecast from the latest vintage.
{ .page-subtitle }

--8<-- "fit-store.md"

--8<-- "nowcast-meta.md"

--8<-- "latest-submission.md"

--8<-- "nowcast-picker.md"

## Nowcast

The weeks before the reference date, estimated for reporting delays.
Each model is shown from its latest origin ([reading the figures](methods/evaluation.md#reading-the-figures)).

??? note "Nowcast by model and province"

    --8<-- "nowcast-only-tabs.md"

## Nowcast and forecast

The same nowcast with the reference week and the weeks after it.

??? note "Nowcast and forecast by model and province"

    --8<-- "nowcast-tabs.md"

The submission pull request embeds [`plot_nowcast`][bdbvhubrenewal.reporting.plots.plot_nowcast].

--8<-- "zone-nowcast.md"

## Delay and spread

The growth to final of each onset day by its age at the latest release, and the observation model by weeks back.
A place using its parent's estimate is labelled with the parent.
How both are fitted is on the [Baseline models](methods/baselines.md#nowcast) page.

![Growth to final by age, and the dispersion of new reports and the removal rate by weeks back, at the latest release.](figures/delay-check.svg#only-light)
![Growth to final by age, and the dispersion of new reports and the removal rate by weeks back, at the latest release.](figures/delay-check-dark.svg#only-dark)

The renewal models fit the onset-to-report delay with the cases.
The figure gives each fit's delay at its latest origin, with 90% intervals: the share of onsets in the last week reported by each day since onset, and the median delay by onset week.
`renewal` and `renewalbvd` fit one delay for all provinces, and `renewalsp` one per province it fits ([Renewal models](methods/model.md)).

![Share of the last onset week reported by days since onset, and the median delay by onset week, for each renewal fit at its latest origin.](figures/renewal-delay.svg#only-light)
![Share of the last onset week reported by days since onset, and the median delay by onset week, for each renewal fit at its latest origin.](figures/renewal-delay-dark.svg#only-dark)
