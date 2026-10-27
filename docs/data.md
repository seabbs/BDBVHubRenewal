# Data

What the hub data show, with all their vintages.
{ .page-subtitle }

How they are extracted and the synthetic vintages are on the [Data methods page](methods/data.md).

## Vintages

--8<-- "vintages.md"

The 2026-08-18 vintage has no delay data.
The 2026-10-09 delay file exists only in the hub's git history.

## National against the provinces

The national (`CD`) row should equal the sum of the provinces.
`bdbv-forecasts data` checks this for every vintage, week and stream (observed, imputed and total), writes `data/national-coherence.csv` and prints the result for the latest vintage.
`bdbv-forecasts coherence --vintage <as_of>` gives the same check for one vintage as JSON, which the [submission workflow](development/submissions.md#checks-and-thresholds) reports in each pull request.
The table counts the weeks where the two differ.

--8<-- "national-coherence.md"

**How it informs the model.**
The submitted national value is the sum of the province values for each sample.
A gap would mean cases counted nationally but in no province.
A model would then need an unassigned national component, which the writer adds to the sum (`to_hub_table(..., unassigned=...)`).
The current models have none.

## Target by vintage

<figure class="doc-figure" markdown="span">
  ![Weekly confirmed cases by onset week for each province with cases, one line per vintage.](figures/target.svg#only-light)
  ![Weekly confirmed cases by onset week for each province with cases, one line per vintage.](figures/target-dark.svg#only-dark)
  <figcaption markdown="1">
    Weekly confirmed cases by onset week (observed plus imputed), by vintage.
    Ituri and Nord Kivu hold most cases.
  </figcaption>
</figure>

**How it informs the model.**
The total is the target.
Each vintage is a partial view of it.

## How recent weeks fill in

<figure class="doc-figure" markdown="span">
  ![Share of the latest national count reported in each earlier vintage, by weeks before the vintage.](figures/revisions.svg#only-light)
  ![Share of the latest national count reported in each earlier vintage, by weeks before the vintage.](figures/revisions-dark.svg#only-dark)
  <figcaption markdown="1">
    National counts in each vintage as a share of the latest vintage, by weeks from the onset week to the vintage's epi week.
    The last two weeks are far from complete.
    Weeks three or more back are still revised by 5 to 10%.
  </figcaption>
</figure>

**How it informs the model.**
This is the weekly reporting triangle.
With the delay data it pins down right truncation.

## Missing onsets

The hub publishes, for every vintage, how many cases in each province and onset week had their onset imputed.
So the share of cases without an onset is measured, not latent.

<figure class="doc-figure" markdown="span">
  ![Share of each week's cases with an imputed onset, by province and vintage, with point size by the week's count.](figures/imputed-share.svg#only-light)
  ![Share of each week's cases with an imputed onset, by province and vintage, with point size by the week's count.](figures/imputed-share-dark.svg#only-dark)
  <figcaption markdown="1">
    Imputed share of cases by onset week, by vintage.
    Point size grows with the week's count.
    It varies over time and between provinces.
  </figcaption>
</figure>

The change between vintages gives the cases added without an onset.
This approximates missingness by report week.
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

**How it informs the model.**
The missing fraction $\pi_p(t)$ is fitted directly to these counts (see [Model](methods/model.md#missing-onsets)).
It is well identified.

## Delays

<figure class="doc-figure" markdown="span">
  ![Histogram of onset-to-report delays for each province with at least ten cases, latest vintage.](figures/delay-distribution.svg#only-light)
  ![Histogram of onset-to-report delays for each province with at least ten cases, latest vintage.](figures/delay-distribution-dark.svg#only-dark)
  <figcaption markdown="1">
    Onset-to-report delays by province in the latest vintage, cut at 40 days.
  </figcaption>
</figure>

--8<-- "delays.md"

<figure class="doc-figure" markdown="span">
  ![Mean and median delay by onset week for the four largest provinces, with the number of cases per week below.](figures/delays-time.svg#only-light)
  ![Mean and median delay by onset week for the four largest provinces, with the number of cases per week below.](figures/delays-time-dark.svg#only-dark)
  <figcaption markdown="1">
    Mean (line) and median (points) delay by onset week, with cases per week below on a log scale.
    Delays were long early on and shortened by June.
    The fall in the last weeks is right truncation: only short delays have been reported yet.
  </figcaption>
</figure>

**How it informs the model.**
The delay data estimate a delay that changes over time, pooled over provinces.
They also give the hub's own back-sampling distribution $\hat g_v$ at each vintage (see [Modelling the hub's imputation](methods/model.md#modelling-the-hubs-imputation)).
A few delays are negative (down to −31 days) and a few are very long (up to 193 days).

## Reconstructed vintages

[Reconstructed vintages](methods/data.md#reconstructed-vintages) rebuild what the data would have shown at past dates from the latest data.
Here they are checked against the real vintages of the same date.

<figure class="doc-figure" markdown="span">
  ![Real and reconstructed national counts by onset week for each real vintage.](figures/reconstruction.svg#only-light)
  ![Real and reconstructed national counts by onset week for each real vintage.](figures/reconstruction-dark.svg#only-dark)
  <figcaption markdown="1">
    Real (solid) and reconstructed (dashed) national counts.
    Older weeks are higher in the reconstruction, because later data added onsets that the real vintage did not yet have.
  </figcaption>
</figure>

--8<-- "reconstruction.md"

The table shows the last four weeks of each real vintage.
Differences are mostly within 10%.
The last, partial week is understated.

## Health zones

The hub accepts health zone forecasts, but provinces are the primary targets.
Health zone totals are kept in `health_zones.csv.gz` for later use.
