# Data

The hub data, with all their vintages.
{ .page-subtitle }

Extraction and synthetic vintages are on the [Data methods page](methods/data.md).

## Vintages

--8<-- "vintages.md"

The 2026-08-18 vintage has no delay data.
The 2026-10-09 delay file exists only in the hub's git history.

## National coherence

The national (`CD`) row should equal the sum of the provinces.
The table counts the weeks where they differ, by vintage and stream.

--8<-- "national-coherence.md"

A gap is cases counted nationally but in no province.
The renewal model forecasts it as an unassigned component ([Forecasts](methods/model.md#forecasts)).

## Target by vintage

<figure class="doc-figure" markdown="span">
  ![Weekly confirmed cases by onset week for each province with cases, one line per vintage.](figures/target.svg#only-light)
  ![Weekly confirmed cases by onset week for each province with cases, one line per vintage.](figures/target-dark.svg#only-dark)
  <figcaption markdown="1">
    Weekly confirmed cases by onset week (observed plus imputed), by vintage.
    Ituri and Nord Kivu hold most cases.
  </figcaption>
</figure>

## Revisions of recent weeks

<figure class="doc-figure" markdown="span">
  ![Share of the latest national count reported in each earlier vintage, by weeks before the vintage.](figures/revisions.svg#only-light)
  ![Share of the latest national count reported in each earlier vintage, by weeks before the vintage.](figures/revisions-dark.svg#only-dark)
  <figcaption markdown="1">
    National counts in each vintage as a share of the latest vintage, by weeks from the onset week to the vintage's epi week.
    The last two weeks are far from complete.
    Weeks three or more back are still revised by 5 to 10%.
  </figcaption>
</figure>

## Missing onsets

The hub publishes, for every vintage, the cases in each province and onset week with an imputed onset.

<figure class="doc-figure" markdown="span">
  ![Share of each week's cases with an imputed onset, by province and vintage, with point size by the week's count.](figures/imputed-share.svg#only-light)
  ![Share of each week's cases with an imputed onset, by province and vintage, with point size by the week's count.](figures/imputed-share-dark.svg#only-dark)
  <figcaption markdown="1">
    Imputed share of cases by onset week, by vintage.
    Point size grows with the week's count.
    It varies over time and between provinces.
  </figcaption>
</figure>

The change between vintages approximates the cases reported without an onset.
It is net of cases that gain an onset later, so it can be negative.

<figure class="doc-figure" markdown="span">
  ![Known-onset and imputed-onset cases added between consecutive vintages, by province.](figures/newly-missing.svg#only-light)
  ![Known-onset and imputed-onset cases added between consecutive vintages, by province.](figures/newly-missing-dark.svg#only-dark)
  <figcaption markdown="1">
    Cases added between vintages, by onset status.
    In Nord Kivu between 2026-10-17 and 2026-10-24, imputed onsets fell by 175 while known onsets rose by 380, so many cases gained an onset after their first report.
  </figcaption>
</figure>

--8<-- "newly-missing.md"

## Delays

<figure class="doc-figure" markdown="span">
  ![Histogram of onset-to-report delays for each province with at least ten cases, latest vintage.](figures/delay-distribution.svg#only-light)
  ![Histogram of onset-to-report delays for each province with at least ten cases, latest vintage.](figures/delay-distribution-dark.svg#only-dark)
  <figcaption markdown="1">
    Onset-to-report delays by province in the latest vintage, cut at 40 days.
  </figcaption>
</figure>

--8<-- "delays.md"

A few delays are negative (down to −31 days) and a few very long (up to 193 days).

<figure class="doc-figure" markdown="span">
  ![Mean and median delay by onset week for the four largest provinces, with the number of cases per week below.](figures/delays-time.svg#only-light)
  ![Mean and median delay by onset week for the four largest provinces, with the number of cases per week below.](figures/delays-time-dark.svg#only-dark)
  <figcaption markdown="1">
    Mean (line) and median (points) delay by onset week, with cases per week below on a log scale.
    Delays were long early on and shortened by June.
    The fall in the last weeks is right truncation, since only short delays have been reported yet.
  </figcaption>
</figure>

## Reconstructed vintages

[Reconstructed vintages](methods/data.md#reconstructed-vintages) are checked against the real vintages of the same date.

<figure class="doc-figure" markdown="span">
  ![Real and reconstructed national counts by onset week for each real vintage.](figures/reconstruction.svg#only-light)
  ![Real and reconstructed national counts by onset week for each real vintage.](figures/reconstruction-dark.svg#only-dark)
  <figcaption markdown="1">
    Real (solid) and reconstructed (dashed) national counts.
    Older weeks are higher in the reconstruction, because later data added onsets that the real vintage did not yet have.
  </figcaption>
</figure>

--8<-- "reconstruction.md"

Over the last four weeks of each real vintage, differences are mostly within 10%.
The last, partial week is understated.

## Health zones

Health zone onsets are in `daily_zones.csv.gz`, and their weekly totals in `health_zones.csv.gz`.
Zone totals sum to the province totals in every vintage.
They set the zone shares that split the province forecasts (see [Health zones](methods/model.md#health-zones)).
