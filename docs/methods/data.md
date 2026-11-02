# Data

Extraction of the hub data, its vintages, and the synthetic vintages used in testing.
{ .page-subtitle }

What the data show is on the [Data results page](../data.md).

## Target and notation

The target is $y_{p}(w)$, the final count of confirmed cases in province $p$ with symptom onset in epi week $w$ (Sunday to Saturday, labelled by its Saturday).
It includes onsets the hub imputes for cases reported without one.
A vintage $v$ is the data published on date $T_v$.
It gives, for each $p$ and $w$, known onsets $K_{p,v}(w)$, imputed onsets $I_{p,v}(w)$ and their total $Y_{p,v}(w) = K_{p,v}(w) + I_{p,v}(w)$.
A forecast made from vintage $v$ serves the round with reference date $r_v$, the first Saturday whose submission window (Sunday to Tuesday, Kinshasa time) closes on or after $T_v$.
Horizon $h$ is the number of weeks from $r_v$ to $w$.
Backtest forecasts cover $h = -4, \dots, 3$.

## Extraction

| Hub file | What it holds | Vintages |
|---|---|---|
| `target-data/time-series.csv` | Weekly confirmed cases by onset week, by province and nationally: observed, imputed and total. Also health zone totals. | All, in one file |
| `auxiliary-data/daily-confirmed-case.csv` | Daily cases by onset day, observed and imputed, by province and health zone. | All, in one file |
| `auxiliary-data/weekly-reporting-delays-by-province.csv` | One row per known-onset case: province, onset week and onset-to-report delay. | Latest only; earlier ones from the archive folder and git history |
| `auxiliary-data/locations.csv` | Location codes, names and parents. | Static |
| `hub-config/*.json` | Rounds, tasks, output types and the metadata schema. | Copied at download |

The daily split into observed and imputed matches the weekly totals in every row, and the delay rows match the observed totals of their vintage.
The hub does not publish deaths, suspected cases, counts by report date, the imputation code or populations.
Province populations are the Institut National de la Statistique's 2019 figures (*Annuaire statistique RDC 2020*), and province centres the WorldPop-weighted centres of BVDOutbreakSize.

## Vintages

Each hub release is a vintage, labelled by its `as_of` date and of kind `real` or `simulated`.
Recent weeks are revised upwards in later vintages, so each forecast is made from one vintage and scored against a later one.

## Release snapshots

`auxiliary-data/daily-confirmed-case.csv` holds, for each release, the daily onsets known at that release.
The count of a day is its known plus imputed onsets, and the counts of a week sum to the hub's weekly target.
A release has an `as_of` date and a data cut-off, its latest onset day over all places.
For the four real releases the `as_of` dates are 2026-08-18, 2026-10-09, 2026-10-17 and 2026-10-24, and the cut-offs are 2026-08-18, 2026-10-07, 2026-10-16 and 2026-10-24.
A release observes the days up to its cut-off, and a day in range with no row counts 0.
The age of day $t$ at release $r$ is the cut-off minus $t$, the onset clock.
Counts of a day are compared across releases on this clock.
A release with a later release gives one pair of releases.
Its step is the difference of the two cut-offs in days, 50, 9 and 8 days for the real pairs.

## Delay and missingness data

The hub imputes an onset for each case reported without one by subtracting a delay drawn from the pooled published delays ([hub issue 61](https://github.com/InsightNet-US/BDBV-Modeling-Hub/issues/61)).
So the share of cases without an onset is measured at every vintage.
The 2026-08-18 vintage has no delay file.

## Simulated vintages

Simulated vintages come from the model's own renewal with known parameters:

- expected onsets from the model's renewal and generation interval, with a national Rt that turns from 1.6 to 0.85 around day 120, fixed province offsets, gravity-kernel importation ($\varepsilon = 0.01$) and susceptible depletion;
- Poisson confirmed cases, with ascertainment rising from 0.5 to 0.75;
- a missing-onset share by report day rising from 5% to 25% and falling to 12%;
- a double censored Gamma delay whose mean falls from 12 to 6 days;
- a lag from report to publication, 85% uniform on 0 to 3 days and 15% uniform on 4 to 60 days, as late entries in the real delay files;
- 30% of cases reported with no onset get one 7 to 42 days after publication, so imputed counts of past weeks fall and known counts rise, as between real vintages.

Vintages are published as the hub does, with missing onsets imputed from the known delays reported so far.
Daily releases are published on Tuesdays, Wednesdays, Fridays and Saturdays, each cut off the lag the real releases of that weekday had.
Their weekly sums are the weekly target.
The truth is every confirmed case by its true onset week.

## BVDOutbreakSize estimates

`renewalbvd` uses the Rt estimates in the public `results-<run>` releases of [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize) (MIT licence, [Zenodo 10.5281/zenodo.20312758](https://doi.org/10.5281/zenodo.20312758)).
These are the national $R_T$ draws, the patch summaries from 23 October and the release's data cut-off.
Each forecast uses the newest release that existed at its cut-off, with data ending by the forecast date, whose fit converged.
The joint fit's worst R-hat must be at most 1.1 (`[bvd] max_rhat` in `config/backtest.toml`), and effective sample sizes are ignored.
The R-hat is read from the fit diagnostics table in the release's rendered site (`site.zip`), and a release without a readable table is passed over for the next older one.
Each fit records the release it used, its R-hat and why newer releases were skipped.
Their use is on the [Renewal models page](model.md#bvdoutbreaksize-informed).
